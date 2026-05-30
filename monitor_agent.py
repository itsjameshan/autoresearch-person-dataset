"""
monitor_agent.py — AutoResearch 持续监督智能体

职责:
  1. 定期读取 AutoResearch v2 的运行状态（SQLite state.db + activity_events.jsonl + reports/）
  2. 分析每轮实验结果、趋势、异常
  3. 生成 handoff.md — 结构化状态传递文档（喂给其他 Agent）
  4. 生成 blog.md — 叙事风格研究进展博客（喂给人 + Agent）
  5. 支持持续模式（轮询）和一次性模式

运行方式:
  python monitor_agent.py                           # 一次性生成 + auto commit
  python monitor_agent.py --watch                   # 持续监督模式，每30秒轮询 + auto commit
  python monitor_agent.py --watch --auto-push       # 持续监督 + 每轮自动 push
  python monitor_agent.py --watch --interval 60     # 自定义轮询间隔
  python monitor_agent.py --handoff-only            # 只生成 handoff.md
  python monitor_agent.py --blog-only               # 只生成 blog.md
  python monitor_agent.py --no-auto-commit          # 禁用自动 git commit
"""

import argparse
import json
import os
import sys
import time
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Any, Optional

try:
    import sqlite3
except ImportError:
    sqlite3 = None

try:
    import numpy as np
except ImportError:
    np = None


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

V2_STATE_DB = BASE_DIR / "autoresearch_v2" / "state" / "state.db"
V2_EVENTS_FILE = BASE_DIR / "activity_events.jsonl"
REPORTS_DIR = BASE_DIR / "reports"

LEGACY_RESULTS = BASE_DIR / "results.tsv"
LEGACY_STATUS = BASE_DIR / "status.md"
LEGACY_SUPERVISOR_REPORT = BASE_DIR / "supervisor_report.md"
LEGACY_SUPERVISOR_AUDIT = BASE_DIR / "supervisor_audit.jsonl"
LEGACY_PROGRAM = BASE_DIR / "program.md"

HANDOFF_MD = BASE_DIR / "handoff.md"
BLOG_MD = BASE_DIR / "blog.md"

DEFAULT_POLL_INTERVAL = 30

TARGET_CDS = 0.85
PRECISION_GATE = 0.90
RECALL_GATE = 0.85
STAGNATION_THRESHOLD = 5
STAGNATION_MIN_DELTA = 0.005
CRASH_THRESHOLD = 3


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icons = {"INFO": "📋", "WARN": "⚠️", "CRIT": "🚨", "OK": "✅", "GEN": "📝"}
    icon = icons.get(level, "📋")
    print(f"[{ts}] [MONITOR:{level}] {icon} {msg}")
    sys.stdout.flush()


class DataReader:
    """统一数据读取层 — 从 v2 state.db + events + reports + legacy 文件读取"""

    def __init__(self):
        self._v2_available = V2_STATE_DB.exists()
        self._events_available = V2_EVENTS_FILE.exists()

    def read_v2_experiments(self) -> list[dict]:
        if not self._v2_available:
            return []
        try:
            conn = sqlite3.connect(str(V2_STATE_DB))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY started_at DESC"
            ).fetchall()
            conn.close()
            results = []
            for row in rows:
                d = dict(row)
                try:
                    d["config_json"] = json.loads(d.get("config_json", "{}"))
                except (json.JSONDecodeError, TypeError):
                    d["config_json"] = {}
                try:
                    d["metrics_json"] = json.loads(d.get("metrics_json", "{}"))
                except (json.JSONDecodeError, TypeError):
                    d["metrics_json"] = {}
                results.append(d)
            return results
        except Exception as e:
            log(f"读取 v2 state.db 失败: {e}", "WARN")
            return []

    def read_v2_decisions(self) -> list[dict]:
        if not self._v2_available:
            return []
        try:
            conn = sqlite3.connect(str(V2_STATE_DB))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            log(f"读取 v2 decisions 失败: {e}", "WARN")
            return []

    def read_v2_data_issues(self) -> list[dict]:
        if not self._v2_available:
            return []
        try:
            conn = sqlite3.connect(str(V2_STATE_DB))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM data_issues ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            log(f"读取 v2 data_issues 失败: {e}", "WARN")
            return []

    def read_v2_budget(self) -> list[dict]:
        if not self._v2_available:
            return []
        try:
            conn = sqlite3.connect(str(V2_STATE_DB))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM budget ORDER BY period DESC").fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            log(f"读取 v2 budget 失败: {e}", "WARN")
            return []

    def read_v2_hitl_gates(self) -> list[dict]:
        if not self._v2_available:
            return []
        try:
            conn = sqlite3.connect(str(V2_STATE_DB))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM hitl_gates ORDER BY requested_at DESC"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            log(f"读取 v2 hitl_gates 失败: {e}", "WARN")
            return []

    def read_events(self, limit: int = 200) -> list[dict]:
        if not self._events_available:
            return []
        try:
            events = []
            with open(V2_EVENTS_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            for line in lines[start:]:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return events
        except Exception as e:
            log(f"读取 events 失败: {e}", "WARN")
            return []

    def read_legacy_results(self) -> list[dict]:
        if not LEGACY_RESULTS.exists():
            return []
        try:
            with open(LEGACY_RESULTS, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            if len(lines) < 2:
                return []
            header = lines[0].strip().split("\t")
            results = []
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                entry = {}
                for i, h in enumerate(header):
                    if i < len(parts):
                        try:
                            entry[h] = float(parts[i])
                        except (ValueError, IndexError):
                            entry[h] = parts[i]
                results.append(entry)
            return results
        except Exception as e:
            log(f"读取 legacy results.tsv 失败: {e}", "WARN")
            return []

    def read_legacy_audit(self) -> list[dict]:
        if not LEGACY_SUPERVISOR_AUDIT.exists():
            return []
        try:
            records = []
            with open(LEGACY_SUPERVISOR_AUDIT, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return records
        except Exception as e:
            log(f"读取 audit log 失败: {e}", "WARN")
            return []

    def read_legacy_supervisor_report(self) -> str:
        if not LEGACY_SUPERVISOR_REPORT.exists():
            return ""
        try:
            return LEGACY_SUPERVISOR_REPORT.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def read_dataset_reports(self) -> dict:
        result = {}
        md_path = REPORTS_DIR / "dataset_report.md"
        json_path = REPORTS_DIR / "dataset_report.json"
        if md_path.exists():
            result["md"] = md_path.read_text(encoding="utf-8", errors="replace")
        if json_path.exists():
            try:
                result["json"] = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    def read_program(self) -> str:
        if LEGACY_PROGRAM.exists():
            return LEGACY_PROGRAM.read_text(encoding="utf-8", errors="replace")
        return ""


class MetricsAnalyzer:
    """指标趋势分析器"""

    def __init__(self):
        self.cds_history = []
        self.map50_history = []
        self.precision_history = []
        self.recall_history = []
        self.loss_history = []

    def feed(self, metrics: dict):
        cds = metrics.get("cds", metrics.get("CDS", 0))
        map50 = metrics.get("mAP50", metrics.get("map50", 0))
        prec = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)
        box_loss = metrics.get("box_loss", None)
        self.cds_history.append(cds)
        self.map50_history.append(map50)
        self.precision_history.append(prec)
        self.recall_history.append(rec)
        if box_loss is not None:
            self.loss_history.append(box_loss)

    def trend(self, values: list, window: int = 5) -> str:
        if len(values) < 2:
            return "insufficient_data"
        recent = values[-min(window, len(values)):]
        older = values[:-len(recent)] if len(values) > len(recent) else values[:1]
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else recent_avg
        diff = recent_avg - older_avg
        if diff > 0.005:
            return "improving"
        elif diff < -0.005:
            return "declining"
        return "stable"

    def is_stagnant(self, window: int = STAGNATION_THRESHOLD) -> bool:
        if len(self.cds_history) < window:
            return False
        recent = self.cds_history[-window:]
        return (max(recent) - min(recent)) < STAGNATION_MIN_DELTA

    def detect_overfitting(self) -> tuple[bool, str]:
        if len(self.map50_history) < 4 or len(self.cds_history) < 4:
            return False, ""
        if (self.map50_history[-1] > self.map50_history[-2] >
                self.map50_history[-3]) and self.cds_history[-1] < self.cds_history[-2]:
            return True, "mAP50 上升但 CDS 下降，疑似过拟合"
        return False, ""

    def detect_metric_anomalies(self, metrics: dict) -> list[str]:
        warnings = []
        p = metrics.get("precision", 0)
        r = metrics.get("recall", 0)
        m50 = metrics.get("mAP50", 0)
        m5095 = metrics.get("mAP50-95", 0)
        if p > 0.99 and r < 0.5:
            warnings.append("precision 异常高但 recall 极低，可能存在置信度阈值作弊")
        if m50 > 0.95 and m5095 < 0.2:
            warnings.append("mAP50 与 mAP50-95 差距过大，检测框定位质量差")
        if metrics.get("inference_ms", 0) < 1.0 and m50 > 0:
            warnings.append("推理时间异常短，可能未正确测量")
        return warnings


class StateAnalyzer:
    """综合状态分析器"""

    def __init__(self, reader: DataReader):
        self.reader = reader
        self.metrics_analyzer = MetricsAnalyzer()

    def gather(self) -> dict:
        v2_exps = self.reader.read_v2_experiments()
        v2_decisions = self.reader.read_v2_decisions()
        v2_issues = self.reader.read_v2_data_issues()
        v2_budget = self.reader.read_v2_budget()
        v2_hitl = self.reader.read_v2_hitl_gates()
        events = self.reader.read_events(limit=300)
        legacy_results = self.reader.read_legacy_results()
        legacy_audit = self.reader.read_legacy_audit()
        legacy_supervisor = self.reader.read_legacy_supervisor_report()
        dataset_reports = self.reader.read_dataset_reports()
        program = self.reader.read_program()

        use_v2 = len(v2_exps) > 0
        if use_v2:
            all_experiments = v2_exps
        else:
            all_experiments = legacy_results

        has_legacy = bool(legacy_results)
        data_source = "v2" if use_v2 else ("legacy" if has_legacy else "empty")

        completed = [e for e in all_experiments
                     if self._get_status(e) in ("completed", "KEEP", "DISCARD")]
        failed = [e for e in all_experiments
                  if self._get_status(e) in ("failed", "CRASH", "EVAL_FAIL")]
        pending = [e for e in all_experiments
                   if self._get_status(e) in ("pending",)]

        for exp in completed:
            metrics = self._get_metrics(exp)
            if metrics:
                self.metrics_analyzer.feed(metrics)

        best, best_idx = self._find_best(completed)

        stagnation = self.metrics_analyzer.is_stagnant()
        overfit, overfit_msg = self.metrics_analyzer.detect_overfitting()

        consecutive_fails = 0
        for e in reversed(all_experiments):
            if self._get_status(e) in ("failed", "CRASH", "EVAL_FAIL"):
                consecutive_fails += 1
            else:
                break

        recent_cds = [self._get_metrics(e).get("cds", self._get_metrics(e).get("CDS", 0))
                      for e in completed[-10:] if self._get_metrics(e)]

        return {
            "data_source": data_source,
            "total_experiments": len(all_experiments),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "pending_count": len(pending),
            "consecutive_fails": consecutive_fails,
            "best_metrics": best,
            "best_run_id": best_idx,
            "stagnation": stagnation,
            "overfitting": {"detected": overfit, "message": overfit_msg},
            "cds_trend": self.metrics_analyzer.trend(self.metrics_analyzer.cds_history),
            "map50_trend": self.metrics_analyzer.trend(self.metrics_analyzer.map50_history),
            "precision_trend": self.metrics_analyzer.trend(self.metrics_analyzer.precision_history),
            "recall_trend": self.metrics_analyzer.trend(self.metrics_analyzer.recall_history),
            "recent_cds": recent_cds,
            "cds_history": self.metrics_analyzer.cds_history,
            "v2_decisions": v2_decisions,
            "v2_data_issues": v2_issues,
            "v2_budget": v2_budget,
            "v2_hitl_gates": v2_hitl,
            "events": events,
            "legacy_audit": legacy_audit,
            "legacy_supervisor_report": legacy_supervisor,
            "dataset_reports": dataset_reports,
            "experiments": all_experiments,
            "completed_experiments": completed,
            "failed_experiments": failed,
        }

    def _get_status(self, exp: dict) -> str:
        return str(exp.get("status", exp.get("_status", "")))

    def _get_metrics(self, exp: dict) -> dict:
        m = exp.get("metrics_json", {})
        if m:
            return m
        metrics = {}
        for k in ("CDS", "cds", "mAP50", "mAP50-95", "precision", "recall",
                   "small_obj_recall", "counting_mae", "inference_ms",
                   "box_loss", "cls_loss", "dfl_loss"):
            if k in exp:
                metrics[k] = exp[k]
        return metrics

    def _find_best(self, completed: list) -> tuple[dict, str]:
        best = {}
        best_cds = -1.0
        best_run = ""
        for exp in completed:
            m = self._get_metrics(exp)
            cds = m.get("cds", m.get("CDS", 0))
            if isinstance(cds, (int, float)) and cds > best_cds:
                best_cds = cds
                best = m
                best_run = exp.get("run_id", exp.get("exp", ""))
        return best, best_run


class HandoffGenerator:
    """生成 handoff.md — 结构化 Agent 间状态传递文档"""

    def generate(self, state: dict) -> str:
        lines = []
        lines.append("# AutoResearch Handoff — Agent 状态传递")
        lines.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> 数据源: {state.get('data_source', 'unknown')}")
        lines.append("")

        lines.append("## 1. 会话状态 (Session Status)\n")
        lines.append(f"- 总实验数: {state['total_experiments']}")
        lines.append(f"- 已完成: {state['completed_count']}")
        lines.append(f"- 失败: {state['failed_count']}")
        lines.append(f"- 运行中: {state['pending_count']}")
        lines.append(f"- 连续失败: {state['consecutive_fails']}")
        lines.append("")

        bm = state.get("best_metrics", {})
        if bm:
            lines.append("## 2. 最佳指标 (Best Metrics)\n")
            cds = bm.get("cds", bm.get("CDS", 0))
            lines.append(f"- **CDS**: {cds:.4f} (目标: {TARGET_CDS})")
            lines.append(f"- mAP50: {bm.get('mAP50', 0):.4f}")
            m5095 = bm.get("mAP50-95", 0)
            if m5095:
                lines.append(f"- mAP50-95: {m5095:.4f}")
            lines.append(f"- Precision: {bm.get('precision', 0):.4f}")
            lines.append(f"- Recall: {bm.get('recall', 0):.4f}")
            small_r = bm.get("small_obj_recall", 0)
            if small_r:
                lines.append(f"- Small Object Recall: {small_r:.4f}")
            cmae = bm.get("counting_mae", 0)
            if cmae:
                lines.append(f"- Counting MAE: {cmae:.4f}")
            inf = bm.get("inference_ms", 0)
            if inf:
                lines.append(f"- Inference: {inf:.1f}ms")
            lines.append(f"- 来源实验: `{state.get('best_run_id', '?')}`")
            lines.append("")

            lines.append("## 3. Quality Gates\n")
            p = bm.get("precision", 0)
            r = bm.get("recall", 0)
            cds_v = bm.get("cds", bm.get("CDS", 0))
            p_ok = "✅" if p >= PRECISION_GATE else "❌"
            r_ok = "✅" if r >= RECALL_GATE else "❌"
            cds_ok = "✅" if cds_v >= TARGET_CDS else "❌"
            lines.append(f"| Gate | 阈值 | 当前值 | 状态 |")
            lines.append(f"|------|------|--------|------|")
            lines.append(f"| CDS | ≥{TARGET_CDS} | {cds_v:.4f} | {cds_ok} |")
            lines.append(f"| Precision | ≥{PRECISION_GATE} | {p:.4f} | {p_ok} |")
            lines.append(f"| Recall | ≥{RECALL_GATE} | {r:.4f} | {r_ok} |")
            lines.append("")

        lines.append("## 4. 趋势分析 (Trends)\n")
        lines.append(f"- CDS 趋势: {state.get('cds_trend', '?')}")
        lines.append(f"- mAP50 趋势: {state.get('map50_trend', '?')}")
        lines.append(f"- Precision 趋势: {state.get('precision_trend', '?')}")
        lines.append(f"- Recall 趋势: {state.get('recall_trend', '?')}")
        stag = state.get("stagnation", False)
        lines.append(f"- 停滞检测: {'🚨 是' if stag else '✅ 否'}")
        overfit = state.get("overfitting", {})
        if overfit.get("detected"):
            lines.append(f"- 过拟合告警: ⚠️ {overfit.get('message', '')}")
        lines.append("")

        cds_hist = state.get("cds_history", [])
        if cds_hist:
            lines.append("### CDS 历史序列\n")
            lines.append("```json")
            lines.append(json.dumps([round(v, 4) for v in cds_hist[-20:]], ensure_ascii=False))
            lines.append("```\n")

        lines.append("## 5. Agent 决策摘要 (Recent Decisions)\n")
        decisions = state.get("v2_decisions", [])
        if decisions:
            lines.append("| 时间 | Agent | Action | Rationale |")
            lines.append("|------|-------|--------|-----------|")
            for d in decisions[:10]:
                ts = str(d.get("created_at", ""))[:19]
                agent = d.get("made_by_agent", "?")
                action = d.get("action_type", "?")
                rationale = (d.get("rationale") or "")[:60]
                lines.append(f"| {ts} | {agent} | {action} | {rationale} |")
            lines.append("")
        else:
            lines.append("(无决策记录)\n")

        lines.append("## 6. 数据问题 (Data Issues)\n")
        issues = state.get("v2_data_issues", [])
        unresolved = [i for i in issues if not i.get("resolved_in_run")]
        if unresolved:
            for issue in unresolved[:10]:
                sev = issue.get("severity", "medium")
                desc = issue.get("description", "")[:80]
                lines.append(f"- [{sev}] {desc}")
        else:
            lines.append("(无未解决的数据问题)")
        lines.append("")

        lines.append("## 7. 预算状态 (Budget)\n")
        budget = state.get("v2_budget", [])
        if budget:
            for b in budget[:3]:
                period = b.get("period", "?")
                used = b.get("gpu_minutes_used", 0)
                limit = b.get("gpu_minutes_limit", 0)
                remain = max(0, limit - used) if limit > 0 else float("inf")
                lines.append(f"- 期间 `{period}`: GPU {used:.1f}/{limit:.1f}min "
                             f"(剩余 {remain:.1f}min)")
        else:
            lines.append("(无预算限制)")
        lines.append("")

        lines.append("## 8. 最近实验列表 (Recent Experiments)\n")
        lines.append("| Run ID | Status | CDS | mAP50 | P | R |")
        lines.append("|--------|--------|-----|-------|---|---|")
        for exp in state.get("completed_experiments", [])[-10:]:
            rid = str(exp.get("run_id", exp.get("exp", "?")))[:20]
            lines.append(
                f"| {rid} | {self._short_status(exp)} | "
                f"{self._get_metric_val(exp, 'cds'):.4f} | "
                f"{self._get_metric_val(exp, 'mAP50'):.4f} | "
                f"{self._get_metric_val(exp, 'precision'):.3f} | "
                f"{self._get_metric_val(exp, 'recall'):.3f} |"
            )
        lines.append("")

        lines.append("## 9. HITL 审批状态\n")
        hitl = state.get("v2_hitl_gates", [])
        pending_hitl = [h for h in hitl if not h.get("decision")]
        if pending_hitl:
            for h in pending_hitl:
                lines.append(f"- ⏳ gate_id={h.get('gate_id')} "
                             f"agent={h.get('requestor_agent')} 等待审批")
        else:
            lines.append("(无待审批的 HITL 请求)")
        lines.append("")

        unresolved_count = len(unresolved)
        lines.append("## 10. 给下一个 Agent 的指令\n")
        lines.append("> 请依据以上状态做出决策。关键检查点：")
        lines.append("> 1. 如果连续失败 ≥3 次，优先排查训练环境/数据问题")
        lines.append(f"> 2. 如果停滞为 True (当前: {stag})，建议切换策略（换模型规模 / 调整学习率 / 数据增强）")
        lines.append(f"> 3. 如果有未解决的数据问题 (当前: {unresolved_count} 个)，先修数据再调参")
        lines.append(f"> 4. Quality Gates 未通过项是下一轮优化的优先级目标")
        lines.append("")

        return "\n".join(lines)

    def _get_metric_val(self, exp: dict, key: str) -> float:
        m = exp.get("metrics_json", {})
        if m:
            val = m.get(key, m.get(key.upper(), 0))
            if isinstance(val, (int, float)):
                return float(val)
        for k in (key, key.upper(), key.upper().replace("-", "_")):
            if k in exp:
                try:
                    return float(exp[k])
                except (ValueError, TypeError):
                    pass
        return 0.0

    def _short_status(self, exp: dict) -> str:
        s = str(exp.get("status", exp.get("_status", "?"))).lower()
        if s in ("completed", "keep"):
            return "KEEP"
        if s in ("failed", "crash"):
            return "FAIL"
        if s in ("discard",):
            return "DISCARD"
        if s == "pending":
            return "PENDING"
        return s[:8].upper()


class BlogGenerator:
    """生成 blog.md — 叙事风格研究进展博客"""

    def generate(self, state: dict) -> str:
        lines = []
        now = datetime.now()
        lines.append(f"# AutoResearch 研究日志 — {now.strftime('%Y年%m月%d日')}\n")

        total = state["total_experiments"]
        completed = state["completed_count"]
        failed = state["failed_count"]
        bm = state.get("best_metrics", {})
        cds_val = bm.get("cds", bm.get("CDS", 0))

        lines.append("## 📊 今日概览\n")
        lines.append(f"截至 {now.strftime('%H:%M')}，AutoResearch 系统已自动完成 **{completed}** 轮实验"
                     f"（共 {total} 轮，失败 {failed} 轮）。\n")

        if cds_val > 0:
            progress_pct = min(100, (cds_val / TARGET_CDS) * 100)
            lines.append(f"### 🎯 目标进度")
            lines.append(f"- 当前最佳 CDS: **{cds_val:.4f}** / 目标 {TARGET_CDS} ({progress_pct:.1f}%)")
            p = bm.get("precision", 0)
            r = bm.get("recall", 0)
            lines.append(f"- Precision: {p:.4f} (门槛: {PRECISION_GATE})")
            lines.append(f"- Recall: {r:.4f} (门槛: {RECALL_GATE})")
            lines.append(f"- mAP50: {bm.get('mAP50', 0):.4f}")
            m5095 = bm.get("mAP50-95", 0)
            if m5095:
                lines.append(f"- mAP50-95: {m5095:.4f}")
            small_r = bm.get("small_obj_recall", 0)
            if small_r:
                lines.append(f"- 小目标召回率: {small_r:.4f}")
            cmae = bm.get("counting_mae", 0)
            if cmae:
                lines.append(f"- 计数 MAE: {cmae:.2f}")
            lines.append("")

        lines.append("## 📈 CDS 趋势\n")
        cds_hist = state.get("cds_history", [])
        if cds_hist and len(cds_hist) >= 2:
            first = cds_hist[0]
            last = cds_hist[-1]
            best_cds_hist = max(cds_hist)
            delta = last - first
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            lines.append(f"- 初始 CDS: {first:.4f} → 当前最佳: {last:.4f} (变化: {delta_str})")
            lines.append(f"- 历史最佳: {best_cds_hist:.4f}")

            trend = state.get("cds_trend", "stable")
            trend_map = {"improving": "📈 持续上升中", "declining": "📉 呈下降趋势", "stable": "➡️ 趋于平稳"}
            lines.append(f"- 趋势: {trend_map.get(trend, trend)}")
            lines.append("")

            ba = self._sparkline(cds_hist[-20:])
            lines.append(f"```\nCDS 走势: {ba}\n```\n")

        lines.append("## 🧪 本轮实验详情\n")
        recent = state.get("completed_experiments", [])[-5:]
        if recent:
            lines.append("| # | Status | CDS | mAP50 | P | R |")
            lines.append("|---|--------|-----|-------|---|---|")
            for i, exp in enumerate(recent):
                s = self._status_label(exp)
                cds = self._get_metric(exp, "cds")
                m50 = self._get_metric(exp, "mAP50")
                prec = self._get_metric(exp, "precision")
                rec = self._get_metric(exp, "recall")
                lines.append(f"| {len(state['completed_experiments']) - len(recent) + i + 1} | {s} | {cds:.4f} | {m50:.4f} | {prec:.3f} | {rec:.3f} |")
            lines.append("")
        else:
            lines.append("(暂无完成实验)\n")

        lines.append("## 🤖 Agent 行为摘要\n")
        decisions = state.get("v2_decisions", [])
        if decisions:
            actions = {}
            for d in decisions[:20]:
                a = d.get("action_type", "unknown")
                actions[a] = actions.get(a, 0) + 1
            lines.append("### 决策分布\n")
            for action, count in sorted(actions.items(), key=lambda x: -x[1]):
                lines.append(f"- **{action}**: {count} 次")
            lines.append("")

            lines.append("### 最近决策\n")
            for d in decisions[:5]:
                ts = str(d.get("created_at", ""))[:19]
                agent = d.get("made_by_agent", "?")
                action = d.get("action_type", "?")
                rationale = (d.get("rationale") or "")[:100]
                lines.append(f"> [{ts}] **{agent}** → `{action}`")
                if rationale:
                    lines.append(f"> _{rationale}_")
                lines.append("")
        else:
            lines.append("(无 Agent 决策记录)\n")

        stag = state.get("stagnation", False)
        consec_fails = state.get("consecutive_fails", 0)
        if stag or consec_fails >= CRASH_THRESHOLD:
            lines.append("## ⚠️ 需要关注的问题\n")
            if stag:
                lines.append(f"- 🛑 **停滞告警**: 连续 {STAGNATION_THRESHOLD} 轮 CDS 无明显提升，建议切换策略")
            if consec_fails >= CRASH_THRESHOLD:
                lines.append(f"- 💥 **连续失败**: 最近 {consec_fails} 轮训练失败，需排查环境/数据")
            overfit = state.get("overfitting", {})
            if overfit.get("detected"):
                lines.append(f"- 📉 **过拟合**: {overfit.get('message')}")
            lines.append("")

        lines.append("## 📋 数据质量\n")
        issues = state.get("v2_data_issues", [])
        if issues:
            unresolved = [i for i in issues if not i.get("resolved_in_run")]
            lines.append(f"- 总问题: {len(issues)} (未解决: {len(unresolved)})")
            if unresolved:
                lines.append("\n### 未解决问题:\n")
                for issue in unresolved[:5]:
                    sev = issue.get("severity", "medium")
                    desc = issue.get("description", "")[:100]
                    lines.append(f"- [{sev}] {desc}")
        else:
            lines.append("- 无记录的数据质量问题")
        lines.append("")

        budget = state.get("v2_budget", [])
        if budget:
            lines.append("## 💰 预算使用\n")
            for b in budget[:3]:
                used = b.get("gpu_minutes_used", 0)
                limit = b.get("gpu_minutes_limit", 0)
                pct = (used / limit * 100) if limit > 0 else 0
                lines.append(f"- GPU: {used:.1f}/{limit:.1f} 分钟 ({pct:.1f}%)")
            lines.append("")

        events = state.get("events", [])
        if events:
            crash_events = [e for e in events if e.get("event") == "crash"]
            halt_events = [e for e in events if e.get("event") == "halt"]
            if crash_events or halt_events:
                lines.append("## 🚨 关键事件\n")
                for e in crash_events[-3:]:
                    lines.append(f"- 💥 CRASH: {e.get('reason', '?')}")
                for e in halt_events[-3:]:
                    lines.append(f"- 🛑 HALT: {e.get('reason', '?')}")
                lines.append("")

        lines.append("## 💡 下一步建议\n")
        stag = state.get("stagnation", False)
        bm = state.get("best_metrics", {})

        has_data = bool(bm) and cds_val > 0

        if not has_data:
            lines.append("1. **等待首轮完成**: 尚无已完成的实验，待第一轮完成后将生成针对性建议")
        elif stag:
            lines.append("1. **打破停滞**: 当前 CDS 不再提升，建议尝试:")
            lines.append("   - 切换 YOLO12 模型规模（如 n→s 或 s→m）")
            lines.append("   - 提高输入分辨率 (imgsz: 1280→1600)")
            lines.append("   - 检查验证集是否代表真实部署场景")
        else:
            lines.append("1. **继续优化**: CDS 仍在提升，建议继续当前策略")

        if has_data:
            p = bm.get("precision", 0)
            r = bm.get("recall", 0)
            if p < PRECISION_GATE and r >= RECALL_GATE:
                lines.append("2. **提升 Precision**: recall 已达标但 precision 不足，建议提高置信度阈值")
            elif r < RECALL_GATE and p >= PRECISION_GATE:
                lines.append("2. **提升 Recall**: precision 已达标但 recall 不足，建议增加小目标检测层")
            elif p < PRECISION_GATE and r < RECALL_GATE:
                lines.append("2. **P/R 均未达标**: 建议从数据质量入手，检查标注一致性")

            if bm.get("small_obj_recall", 0) < 0.5:
                lines.append("3. **小目标优化**: 小目标召回率偏低，考虑 SAHI 切片推理或更高分辨率")
        lines.append("")

        lines.append("---")
        lines.append(f"*此报告由 monitor_agent.py 自动生成于 {now.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append(f"*数据源: {state.get('data_source', 'unknown')}*")

        return "\n".join(lines)

    def _sparkline(self, values: list) -> str:
        if len(values) < 2:
            return str(values)[:80]
        ticks = "▁▂▃▄▅▆▇█"
        vmin = min(values)
        vmax = max(values)
        if vmax - vmin < 0.0001:
            return "█" * len(values)
        chars = []
        for v in values:
            idx = int((v - vmin) / (vmax - vmin) * (len(ticks) - 1))
            idx = max(0, min(len(ticks) - 1, idx))
            chars.append(ticks[idx])
        return "".join(chars)

    def _get_metric(self, exp: dict, key: str) -> float:
        m = exp.get("metrics_json", {})
        if m:
            val = m.get(key, m.get(key.upper(), 0))
            if isinstance(val, (int, float)):
                return float(val)
        for k in (key, key.upper()):
            if k in exp:
                try:
                    return float(exp[k])
                except (ValueError, TypeError):
                    pass
        return 0.0

    def _status_label(self, exp: dict) -> str:
        s = str(exp.get("status", exp.get("_status", "?"))).lower()
        if s in ("completed", "keep"):
            return "✅ KEEP"
        if s == "discard":
            return "🗑️ DISCARD"
        if s in ("failed", "crash"):
            return "💥 FAIL"
        return s[:10]


class MonitorAgent:
    """AutoResearch 持续监督智能体"""

    def __init__(self, *, auto_commit: bool = True, auto_push: bool = False):
        self.reader = DataReader()
        self.analyzer = StateAnalyzer(self.reader)
        self.handoff_gen = HandoffGenerator()
        self.blog_gen = BlogGenerator()
        self._last_handoff_hash = ""
        self._last_blog_hash = ""
        self._auto_commit = auto_commit
        self._auto_push = auto_push
        self._round = 0

    def _git_add(self, *paths: Path) -> bool:
        try:
            result = subprocess.run(
                ["git", "add"] + [str(p) for p in paths],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(BASE_DIR),
            )
            return result.returncode == 0
        except Exception as e:
            log(f"git add 失败: {e}", "WARN")
            return False

    def _git_commit(self, message: str) -> bool:
        try:
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(BASE_DIR),
            )
            if result.returncode == 0:
                return True
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                return True
            log(f"git commit 失败: {result.stderr.strip()}", "WARN")
            return False
        except Exception as e:
            log(f"git commit 失败: {e}", "WARN")
            return False

    def _git_push(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "push"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(BASE_DIR),
            )
            if result.returncode == 0:
                return True
            log(f"git push 失败: {result.stderr.strip()}", "WARN")
            return False
        except Exception as e:
            log(f"git push 失败: {e}", "WARN")
            return False

    def _auto_git_snapshot(self, state: dict, files_changed: list[str], round_num: int = 0):
        if not self._auto_commit:
            return
        if not files_changed:
            log("无文件变更，跳过 git commit", "INFO")
            return

        paths = [BASE_DIR / f for f in files_changed]
        if not self._git_add(*paths):
            return

        bm = state.get("best_metrics", {})
        cds = bm.get("cds", bm.get("CDS", 0))
        total = state.get("total_experiments", 0)
        completed = state.get("completed_count", 0)
        trend = state.get("cds_trend", "?")
        stag = "STAGNANT" if state.get("stagnation") else ""

        if round_num:
            msg = (
                f"[monitor] round#{round_num} | "
                f"exp={completed}/{total} | "
                f"CDS={cds:.4f} | "
                f"trend={trend}"
                + (f" | {stag}" if stag else "")
            )
        else:
            msg = (
                f"[monitor] oneshot | "
                f"exp={completed}/{total} | "
                f"CDS={cds:.4f} | "
                f"trend={trend}"
                + (f" | {stag}" if stag else "")
            )

        if self._git_commit(msg):
            log(f"git commit: {msg}", "OK")
            if self._auto_push:
                if self._git_push():
                    log("git push 成功", "OK")
        else:
            log("git commit 跳过（无变更或失败）", "INFO")

    def run_once(self, *, gen_handoff: bool = True, gen_blog: bool = True) -> dict:
        log("开始收集 AutoResearch 状态...")
        state = self.analyzer.gather()
        log(f"数据源: {state['data_source']}, "
            f"实验: {state['completed_count']}/{state['total_experiments']}, "
            f"最佳 CDS: {state['best_metrics'].get('cds', state['best_metrics'].get('CDS', 0)):.4f}")

        result = {"state": state}
        files_changed = []

        if gen_handoff:
            handoff_content = self.handoff_gen.generate(state)
            new_hash = hashlib.md5(handoff_content.encode()).hexdigest()
            if new_hash != self._last_handoff_hash:
                HANDOFF_MD.write_text(handoff_content, encoding="utf-8")
                log(f"handoff.md 已更新 → {HANDOFF_MD}", "GEN")
                self._last_handoff_hash = new_hash
                result["handoff_updated"] = True
                files_changed.append("handoff.md")
            else:
                log("handoff.md 内容无变化，跳过写入", "INFO")
                result["handoff_updated"] = False
            result["handoff"] = handoff_content

        if gen_blog:
            blog_content = self.blog_gen.generate(state)
            new_hash = hashlib.md5(blog_content.encode()).hexdigest()
            if new_hash != self._last_blog_hash:
                BLOG_MD.write_text(blog_content, encoding="utf-8")
                log(f"blog.md 已更新 → {BLOG_MD}", "GEN")
                self._last_blog_hash = new_hash
                result["blog_updated"] = True
                files_changed.append("blog.md")
            else:
                log("blog.md 内容无变化，跳过写入", "INFO")
                result["blog_updated"] = False
            result["blog"] = blog_content

        if files_changed:
            self._auto_git_snapshot(state, files_changed, self._round)

        return result

    def watch(self, poll_interval: int = DEFAULT_POLL_INTERVAL):
        log(f"启动持续监督模式 (间隔 {poll_interval}s)")
        log(f"监听目标: {V2_STATE_DB} / {V2_EVENTS_FILE}")
        log(f"自动提交: {'✅' if self._auto_commit else '❌'} | 自动推送: {'✅' if self._auto_push else '❌'}")
        log("按 Ctrl+C 停止")
        log("")

        try:
            while True:
                self._round += 1
                log(f"── 第 {self._round} 次轮询 ──")
                result = self.run_once()
                state = result["state"]

                stag = state.get("stagnation", False)
                consec = state.get("consecutive_fails", 0)
                if stag:
                    log(f"⚠️ 停滞检测: CDS 已 {STAGNATION_THRESHOLD} 轮无明显提升", "WARN")
                if consec >= CRASH_THRESHOLD:
                    log(f"🚨 连续失败: {consec} 轮训练失败", "CRIT")

                log(f"等待 {poll_interval}s ...")
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            log("收到中断信号，生成最终报告...")
            self._round += 1
            self.run_once()
            log("监督智能体已停止", "OK")


def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch 持续监督智能体 — 生成 handoff.md + blog.md"
    )
    parser.add_argument("--watch", action="store_true",
                        help="持续监督模式（轮询）")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help=f"轮询间隔秒数 (默认: {DEFAULT_POLL_INTERVAL})")
    parser.add_argument("--handoff-only", action="store_true",
                        help="只生成 handoff.md，不生成 blog.md")
    parser.add_argument("--blog-only", action="store_true",
                        help="只生成 blog.md，不生成 handoff.md")
    parser.add_argument("--auto-push", action="store_true",
                        help="每次轮询后自动 git push（需配合 --watch 使用）")
    parser.add_argument("--no-auto-commit", action="store_true",
                        help="禁用自动 git commit（默认启用）")
    parser.add_argument("--quiet", action="store_true",
                        help="安静模式，仅输出关键信息")
    args = parser.parse_args()

    agent = MonitorAgent(
        auto_commit=not args.no_auto_commit,
        auto_push=args.auto_push,
    )

    if args.watch:
        agent.watch(poll_interval=args.interval)
    else:
        gen_handoff = not args.blog_only
        gen_blog = not args.handoff_only
        result = agent.run_once(gen_handoff=gen_handoff, gen_blog=gen_blog)
        state = result["state"]

        print()
        print("=" * 60)
        print("  AutoResearch Monitor — 一次性分析完成")
        print("=" * 60)
        bm = state["best_metrics"]
        cds = bm.get("cds", bm.get("CDS", 0))
        print(f"  数据源:      {state['data_source']}")
        print(f"  总实验:      {state['total_experiments']}")
        print(f"  最佳 CDS:    {cds:.4f}")
        print(f"  CDS 趋势:    {state['cds_trend']}")
        print(f"  停滞:        {state['stagnation']}")
        if result.get("handoff_updated"):
            print(f"  handoff.md → {HANDOFF_MD}")
        if result.get("blog_updated"):
            print(f"  blog.md    → {BLOG_MD}")
        print("=" * 60)


if __name__ == "__main__":
    main()