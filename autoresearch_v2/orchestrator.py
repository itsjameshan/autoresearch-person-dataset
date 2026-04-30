"""
orchestrator.py — 主驱动器，替代 ollama_runner.py

单线程主循环驱动器，协调 Researcher / Curator / Triage 三个 Agent，
通过 SQLite 状态库交流，不用 free-text 互相调用。

终止条件:
  - CDS >= target_cds (达到目标)
  - 连续5轮 CDS 提升 < 0.5% (平台期)
  - 总预算耗尽
  - 累计连续失败 >= 3 次 (系统性问题)
  - 人工 stop 信号

运行方式:
  python -m autoresearch_v2.orchestrator
  python -m autoresearch_v2.orchestrator --config config.json
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Optional

from autoresearch_v2.db import StateDB
from autoresearch_v2.budget import BudgetGuard, BudgetExceeded
from autoresearch_v2.hitl import HITLGate
from autoresearch_v2.agents.researcher import ResearcherAgent
from autoresearch_v2.agents.curator import CuratorAgent
from autoresearch_v2.agents.triage import TriageAgent
from autoresearch_v2.tools.train_dispatcher import TrainDispatcher, TrainingFailed
from autoresearch_v2.tools.eval_dispatcher import EvalDispatcher


TARGET_CDS = 0.85
PRECISION_GATE = 0.90
RECALL_GATE = 0.85
MAX_CONSECUTIVE_FAILS = 3
STAGNATION_WINDOW = 5
STAGNATION_MIN_DELTA = 0.005
COOLDOWN_SEC = 30

DEFAULT_DATA_YAML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "person_dataset", "person.yaml"
))


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "📋", "WARN": "⚠️", "CRIT": "🚨",
        "OK": "✅", "SUGGEST": "💡", "AGENT": "🤖",
    }.get(level, "📋")
    print(f"[{ts}] [ORCHESTRATOR:{level}] {prefix} {msg}")
    sys.stdout.flush()


class Orchestrator:
    def __init__(self, config: dict = None):
        config = config or {}

        self.state = StateDB()
        self.budget = BudgetGuard(
            self.state,
            default_gpu_limit=config.get("gpu_minutes_limit", 0),
            default_dollar_limit=config.get("dollar_limit", 0),
        )
        self.hitl = HITLGate(
            self.state,
            flask_port=config.get("hitl_port", 8765),
        )

        self.researcher = ResearcherAgent(
            llm_backend=config.get("llm_backend", "auto"),
            anthropic_model=config.get("anthropic_model", "claude-sonnet-4-20250514"),
            ollama_model=config.get("ollama_model", "gemma3:4b"),
            ollama_url=config.get("ollama_url", "http://localhost:11434"),
        )

        self.curator = CuratorAgent(
            state=self.state,
            llm_backend=config.get("llm_backend", "auto"),
            anthropic_model=config.get("anthropic_model", "claude-sonnet-4-20250514"),
            ollama_model=config.get("ollama_model", "gemma3:4b"),
            ollama_url=config.get("ollama_url", "http://localhost:11434"),
        )

        self.triage = TriageAgent(
            state=self.state,
            llm_backend=config.get("llm_backend", "auto"),
            anthropic_model=config.get("triage_anthropic_model", "claude-haiku-4-20250414"),
            ollama_model=config.get("ollama_model", "gemma3:4b"),
            ollama_url=config.get("ollama_url", "http://localhost:11434"),
        )

        self.train_dispatcher = TrainDispatcher(state=self.state)
        self.eval_dispatcher = EvalDispatcher(
            state=self.state,
            data_yaml=config.get("data_yaml", DEFAULT_DATA_YAML),
            imgsz=config.get("imgsz", 1280),
        )

        self.target_cds = config.get("target_cds", TARGET_CDS)
        self.max_consecutive_fails = config.get("max_consecutive_fails", MAX_CONSECUTIVE_FAILS)
        self.stagnation_window = config.get("stagnation_window", STAGNATION_WINDOW)
        self.stagnation_min_delta = config.get("stagnation_min_delta", STAGNATION_MIN_DELTA)
        self.cooldown_sec = config.get("cooldown_sec", COOLDOWN_SEC)
        self.max_iterations = config.get("max_iterations", 50)

        self.consecutive_fails = 0
        self.iteration = 0
        self.stop_signal_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "state", "STOP"
        )

    def _generate_run_id(self) -> str:
        return f"run_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _check_stop_signal(self) -> bool:
        return os.path.exists(self.stop_signal_file)

    def _check_stagnation(self) -> bool:
        recent = self.state.get_recent_experiments(self.stagnation_window + 1)
        if len(recent) < self.stagnation_window:
            return False

        cds_values = [r["metrics_json"].get("cds", 0) for r in recent
                      if r.get("metrics_json")]
        if len(cds_values) < self.stagnation_window:
            return False

        max_cds = max(cds_values)
        min_cds = min(cds_values)
        return (max_cds - min_cds) < self.stagnation_min_delta

    def _check_quality_gates(self, metrics: dict) -> tuple[bool, str]:
        p = metrics.get("precision", 0)
        r = metrics.get("recall", 0)
        if p >= PRECISION_GATE and r >= RECALL_GATE:
            return True, f"PASS (P={p:.4f}, R={r:.4f})"
        issues = []
        if p < PRECISION_GATE:
            issues.append(f"precision={p:.4f}<{PRECISION_GATE}")
        if r < RECALL_GATE:
            issues.append(f"recall={r:.4f}<{RECALL_GATE}")
        return False, f"FAIL ({', '.join(issues)})"

    def _handle_triage(self, error_info: dict, log_tail: str = "",
                       config_diff: dict = None) -> dict:
        diagnosis = self.triage.diagnose(error_info, log_tail, config_diff)
        action = diagnosis.get("recommended_action", "escalate_human")

        if action == "rollback":
            log(f"Triage建议回滚: {diagnosis.get('root_cause_hypothesis', '')}", "WARN")
            return diagnosis

        if action == "retry_with_fix":
            fix = diagnosis.get("fix_diff", {})
            if fix:
                log(f"Triage建议修复: {fix}", "SUGGEST")
                from autoresearch_v2.tools.train_dispatcher import apply_config_diff
                apply_config_diff(fix)
            return diagnosis

        if action == "escalate_human":
            log(f"Triage升级人工: {diagnosis.get('root_cause_hypothesis', '')}", "CRIT")
            return diagnosis

        return diagnosis

    def run(self):
        log("=" * 60)
        log("  AutoResearch v2 — 多Agent自演化系统启动")
        log("=" * 60)
        log(f"目标 CDS: {self.target_cds}")
        log(f"最大迭代: {self.max_iterations}")
        log(f"连续失败上限: {self.max_consecutive_fails}")

        if self.budget.default_gpu_limit > 0 or self.budget.default_dollar_limit > 0:
            self.hitl.start_flask_server()

        while self.iteration < self.max_iterations:
            self.iteration += 1
            run_id = self._generate_run_id()

            log(f"\n{'='*60}")
            log(f"  迭代 {self.iteration}/{self.max_iterations} | run_id={run_id}")
            log(f"{'='*60}")

            if self._check_stop_signal():
                log("收到人工停止信号", "WARN")
                break

            snapshot = self.state.get_recent_experiments(n=10)
            best_metrics = self.state.get_best_metrics() or {}
            remaining_budget = self.budget.check_remaining()
            top_issues = self.state.get_top_data_issues(n=3)

            recent_failures = []
            for exp in snapshot:
                if exp.get("status") == "failed":
                    recent_failures.append({
                        "run_id": exp.get("run_id", ""),
                        "root_cause": "training_failed",
                    })

            decision = self.researcher.decide(
                recent_experiments=snapshot,
                best_metrics=best_metrics,
                remaining_budget=remaining_budget,
                top_data_issues=top_issues,
                recent_failures=recent_failures,
            )

            log(f"Researcher决策: action={decision.get('action_type')}", "AGENT")
            log(f"  rationale: {decision.get('rationale', '')}", "AGENT")

            self.state.log_decision(
                run_id=run_id,
                made_by_agent="researcher",
                action_type=decision.get("action_type", "unknown"),
                rationale=decision.get("rationale", ""),
                expected_improvement=json.dumps(
                    decision.get("expected_metric_delta", {}), ensure_ascii=False
                ),
                estimated_cost=decision.get("estimated_cost_usd", 0),
            )

            if decision.get("action_type") == "stop":
                log("Researcher决定停止迭代", "OK")
                break

            if decision.get("needs_hitl", False):
                gate_id = self.hitl.request(
                    run_id, "researcher",
                    prompt_text=(
                        f"Action: {decision.get('action_type')}\n"
                        f"Rationale: {decision.get('rationale')}\n"
                        f"Estimated cost: ${decision.get('estimated_cost_usd', 0):.2f}"
                    ),
                )
                log(f"需要人工审批 (gate_id={gate_id})，等待中...", "WARN")
                approved = self.hitl.wait_for_decision(gate_id, timeout=86400)
                if not approved:
                    log("人工审批被拒绝，跳过本次迭代", "WARN")
                    continue
                log("人工审批通过", "OK")

            try:
                self.budget.reserve(
                    run_id,
                    estimated_gpu_minutes=decision.get("estimated_gpu_minutes", 60),
                    estimated_cost_usd=decision.get("estimated_cost_usd", 1.0),
                )
            except BudgetExceeded as e:
                log(f"预算不足: {e}", "CRIT")
                break

            try:
                best_pt = self.train_dispatcher.dispatch(run_id, decision)
                log(f"训练完成: model={best_pt}", "OK")
                self.consecutive_fails = 0

            except TrainingFailed as e:
                self.consecutive_fails += 1
                log(f"训练失败 (连续{self.consecutive_fails}次): {e}", "CRIT")

                diagnosis = self._handle_triage(
                    error_info={"run_id": run_id, "error": str(e)},
                    log_tail=e.log_tail,
                    config_diff=decision.get("config_diff"),
                )

                self.budget.release_reserved(run_id)

                if self.consecutive_fails >= self.max_consecutive_fails:
                    log(f"连续{self.max_consecutive_fails}次失败，停止迭代", "CRIT")
                    break

                time.sleep(self.cooldown_sec)
                continue

            eval_result = self.eval_dispatcher.run(
                best_pt, run_id=run_id, extract_fp_fn=True
            )

            metrics = eval_result.get("metrics", {})
            fp_fn_data = eval_result.get("fp_fn_data", {})

            if not metrics:
                log("评估失败", "CRIT")
                self.consecutive_fails += 1
                if self.consecutive_fails >= self.max_consecutive_fails:
                    break
                continue

            cds = metrics.get("cds", 0)
            log(f"评估结果: CDS={cds:.4f} | mAP50={metrics.get('mAP50',0):.4f} "
                f"| P={metrics.get('precision',0):.4f} | R={metrics.get('recall',0):.4f}")

            gates_pass, gates_msg = self._check_quality_gates(metrics)
            log(f"Quality Gates: {gates_msg}")

            if cds >= self.target_cds and gates_pass:
                log(f"🎉 目标达成！CDS={cds:.4f} >= {self.target_cds}", "OK")
                break

            if self._check_stagnation():
                log(f"检测到平台期 (连续{self.stagnation_window}轮CDS提升<{self.stagnation_min_delta})", "WARN")

            if fp_fn_data:
                log("触发 Curator 异步分析...", "AGENT")
                try:
                    curator_report = self.curator.analyze(fp_fn_data, run_id)
                    if curator_report.get("worst_failure_modes"):
                        for fm in curator_report["worst_failure_modes"][:3]:
                            log(f"  Curator发现: {fm.get('pattern', '')} "
                                f"(影响: {fm.get('estimated_impact', '')})", "SUGGEST")
                except Exception as e:
                    log(f"Curator分析失败: {e}", "WARN")

            actual_gpu_min = 0
            exp = self.state.get_experiment(run_id)
            if exp:
                actual_gpu_min = exp.get("gpu_minutes", 0)
            self.budget.commit_actual(run_id, actual_gpu_min, actual_gpu_min * 0.01)

            time.sleep(self.cooldown_sec)

        self._print_summary()

    def _print_summary(self):
        log("\n" + "=" * 60)
        log("  AutoResearch v2 — 迭代结束")
        log("=" * 60)

        best = self.state.get_best_metrics()
        if best:
            log(f"最佳 CDS: {best.get('cds', 0):.4f}")
            log(f"最佳 mAP50: {best.get('mAP50', 0):.4f}")
            log(f"最佳 Precision: {best.get('precision', 0):.4f}")
            log(f"最佳 Recall: {best.get('recall', 0):.4f}")
            log(f"最佳 Small Obj Recall: {best.get('small_obj_recall', 0):.4f}")
            log(f"最佳 Counting MAE: {best.get('counting_mae', 0):.2f}")
            log(f"最佳 Inference: {best.get('inference_ms', 0):.1f}ms")

        total = self.state.get_experiment_count()
        log(f"总实验数: {total}")
        log(f"总迭代数: {self.iteration}")

        issues = self.state.get_unresolved_issues()
        if issues:
            log(f"未解决数据问题: {len(issues)}")

        log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AutoResearch v2 Orchestrator")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (JSON)")
    parser.add_argument("--target-cds", type=float, default=TARGET_CDS)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--llm-backend", type=str, default="auto",
                        choices=["auto", "anthropic", "ollama"])
    parser.add_argument("--ollama-model", type=str, default="gemma3:4b")
    parser.add_argument("--anthropic-model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--data-yaml", type=str, default=DEFAULT_DATA_YAML)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--gpu-limit", type=float, default=0,
                        help="GPU分钟预算 (0=无限制)")
    parser.add_argument("--dollar-limit", type=float, default=0,
                        help="美元预算 (0=无限制)")
    parser.add_argument("--hitl-port", type=int, default=8765)
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SEC)
    args = parser.parse_args()

    config = {
        "target_cds": args.target_cds,
        "max_iterations": args.max_iterations,
        "llm_backend": args.llm_backend,
        "ollama_model": args.ollama_model,
        "anthropic_model": args.anthropic_model,
        "data_yaml": args.data_yaml,
        "imgsz": args.imgsz,
        "gpu_minutes_limit": args.gpu_limit,
        "dollar_limit": args.dollar_limit,
        "hitl_port": args.hitl_port,
        "cooldown_sec": args.cooldown,
    }

    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            config.update(file_config)
        except Exception as e:
            print(f"Warning: failed to load config file: {e}", file=sys.stderr)

    orchestrator = Orchestrator(config)
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        log("收到中断信号，正在停止...", "WARN")
        orchestrator._print_summary()
    finally:
        orchestrator.state.close()


if __name__ == "__main__":
    main()
