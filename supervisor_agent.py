"""
supervisor_agent.py — 监督智能体：审核、监控、守护训练智能体

职责:
  1. 实时监控训练进程状态和资源使用
  2. 审核每次实验的 KEEP/DISCARD 决策
  3. 验证指标真实性（检测过拟合、异常波动）
  4. 异常检测（崩溃循环、资源泄漏、停滞）
  5. 安全守卫（硬件安全、代码安全）
  6. 智能建议（停滞时推荐新方向）
  7. 生成审计报告

运行方式:
  python supervisor_agent.py                    # 启动监督（监控模式）
  python supervisor_agent.py --audit            # 对已有结果做一次性审计
  python supervisor_agent.py --report           # 生成审计报告
  python supervisor_agent.py --watchdog         # 守护模式（可中断训练）
"""

import os
import re
import sys
import json
import time
import math
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


RESULTS_FILE = "results.tsv"
METRICS_FILE = "last_metrics.json"
STATUS_FILE = "status.md"
LOG_FILE = "run.log"
TRAIN_SCRIPT = "train.py"
AUDIT_LOG = "supervisor_audit.jsonl"
AUDIT_REPORT = "supervisor_report.md"
PROGRAM_FILE = "program.md"

POLL_INTERVAL = 10
HISTORY_WINDOW = 20

CRASH_THRESHOLD = 3
STAGNATION_THRESHOLD = 6
VRAM_WARN = 0.85
VRAM_CRIT = 0.95
RAM_WARN = 0.80
RAM_CRIT = 0.92
DISK_MIN_GB = 5

TARGET_CDS = 0.85
PRECISION_GATE = 0.90
RECALL_GATE = 0.85

CDS_IMPROVE_MIN = 0.001


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {"INFO": "📋", "WARN": "⚠️", "CRIT": "🚨", "AUDIT": "🔍", "OK": "✅", "SUGGEST": "💡"}.get(level, "📋")
    print(f"[{ts}] [SUPERVISOR:{level}] {prefix} {msg}")
    sys.stdout.flush()


def write_audit_record(record):
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"审计日志写入失败: {e}", "WARN")


class MetricsAnalyzer:
    """指标趋势分析器"""

    def __init__(self):
        self.history = deque(maxlen=HISTORY_WINDOW)

    def add(self, metrics):
        self.history.append(metrics)

    def trend(self, key, window=None):
        if len(self.history) < 2:
            return "insufficient_data"
        data = list(self.history)
        if window:
            data = data[-window:]
        values = [m.get(key, 0) for m in data if m.get(key) is not None]
        if len(values) < 2:
            return "insufficient_data"
        recent_avg = sum(values[-3:]) / len(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = sum(values[:-3]) / len(values[:-3]) if len(values) > 3 else values[0]
        diff = recent_avg - older_avg
        if diff > 0.01:
            return "improving"
        elif diff < -0.01:
            return "declining"
        return "stagnant"

    def volatility(self, key):
        if np is None or len(self.history) < 3:
            return 0.0
        values = [m.get(key, 0) for m in self.history if m.get(key) is not None]
        if len(values) < 3:
            return 0.0
        return float(np.std(values))

    def detect_overfitting(self):
        if len(self.history) < 4:
            return False, ""
        recent = list(self.history)[-4:]
        map50_vals = [m.get("mAP50", 0) for m in recent]
        map5095_vals = [m.get("mAP50-95", 0) for m in recent]
        if len(map50_vals) >= 4:
            if map50_vals[-1] > map50_vals[-2] > map50_vals[-3] and map5095_vals[-1] < map5095_vals[-2]:
                return True, "mAP50上升但mAP50-95下降，疑似过拟合"
        return False, ""

    def detect_metric_cheating(self, metrics):
        warnings = []
        if metrics.get("precision", 0) > 0.99 and metrics.get("recall", 0) < 0.5:
            warnings.append("precision异常高但recall极低，可能通过极低置信度阈值作弊")
        if metrics.get("mAP50", 0) > 0.95 and metrics.get("mAP50-95", 0) < 0.2:
            warnings.append("mAP50与mAP50-95差距过大，检测框定位质量差")
        if metrics.get("inference_ms", 0) < 1.0 and metrics.get("mAP50", 0) > 0:
            warnings.append("推理时间异常短，可能未正确测量")
        if metrics.get("counting_mae", 0) < 0.01 and metrics.get("mAP50", 0) > 0.3:
            warnings.append("计数MAE异常低，需验证评估数据集是否正确")
        return warnings


class DecisionAuditor:
    """决策审核器"""

    def __init__(self):
        self.decisions = []

    def audit_keep_decision(self, metrics, best_metrics, exp_num):
        issues = []
        cds = metrics.get("CDS", metrics.get("cds", 0))
        best_cds = best_metrics.get("CDS", best_metrics.get("cds", 0))

        if cds <= best_cds:
            issues.append(f"CDS({cds:.4f})未超过最佳({best_cds:.4f})，KEEP决策可能有误")

        if metrics.get("precision", 0) < PRECISION_GATE:
            issues.append(f"precision({metrics.get('precision', 0):.4f})未达标(<{PRECISION_GATE})")
        if metrics.get("recall", 0) < RECALL_GATE:
            issues.append(f"recall({metrics.get('recall', 0):.4f})未达标(<{RECALL_GATE})")

        if metrics.get("counting_mae", 0) > 5.0:
            issues.append(f"计数MAE({metrics.get('counting_mae', 0):.2f})过高")

        if metrics.get("inference_ms", 0) > 115:
            issues.append(f"推理延迟({metrics.get('inference_ms', 0):.1f}ms)超过部署门槛")

        verdict = "APPROVED" if not issues else "FLAGGED"
        record = {
            "timestamp": datetime.now().isoformat(),
            "exp_num": exp_num,
            "action": "KEEP",
            "verdict": verdict,
            "cds": cds,
            "issues": issues,
        }
        self.decisions.append(record)
        write_audit_record(record)
        return verdict, issues

    def audit_discard_decision(self, metrics, best_metrics, exp_num):
        issues = []
        cds = metrics.get("CDS", metrics.get("cds", 0))
        best_cds = best_metrics.get("CDS", best_metrics.get("cds", 0))

        if cds > best_cds - 0.005 and cds < best_cds:
            issues.append(f"CDS({cds:.4f})接近最佳({best_cds:.4f})，DISCARD可能过于激进")

        precision_trend = metrics.get("precision", 0) - best_metrics.get("precision", 0)
        recall_trend = metrics.get("recall", 0) - best_metrics.get("recall", 0)
        if precision_trend > 0.02 or recall_trend > 0.02:
            issues.append(f"precision/recall有提升但CDS未提升，可能CDS权重需调整")

        verdict = "APPROVED" if not issues else "FLAGGED"
        record = {
            "timestamp": datetime.now().isoformat(),
            "exp_num": exp_num,
            "action": "DISCARD",
            "verdict": verdict,
            "cds": cds,
            "issues": issues,
        }
        self.decisions.append(record)
        write_audit_record(record)
        return verdict, issues


class SafetyGuard:
    """安全守卫"""

    def __init__(self):
        self.violations = []

    def check_hardware(self):
        issues = []
        if psutil is None:
            return issues

        ram = psutil.virtual_memory()
        ram_pct = ram.used / ram.total
        if ram_pct > RAM_CRIT:
            issues.append(("CRIT", f"RAM使用率{ram_pct:.1%}超过临界值{RAM_CRIT:.0%}"))
        elif ram_pct > RAM_WARN:
            issues.append(("WARN", f"RAM使用率{ram_pct:.1%}超过警告值{RAM_WARN:.0%}"))

        disk = psutil.disk_usage(".")
        free_gb = disk.free / (1024**3)
        if free_gb < DISK_MIN_GB:
            issues.append(("CRIT", f"磁盘剩余{free_gb:.1f}GB不足{DISK_MIN_GB}GB"))

        if torch is not None and torch.cuda.is_available():
            try:
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_used = torch.cuda.memory_allocated(0)
                vram_pct = vram_used / vram_total
                if vram_pct > VRAM_CRIT:
                    issues.append(("CRIT", f"VRAM使用率{vram_pct:.1%}超过临界值{VRAM_CRIT:.0%}"))
                elif vram_pct > VRAM_WARN:
                    issues.append(("WARN", f"VRAM使用率{vram_pct:.1%}超过警告值{VRAM_WARN:.0%}"))
            except Exception:
                pass

        for level, msg in issues:
            self.violations.append({"timestamp": datetime.now().isoformat(), "level": level, "msg": msg})
            write_audit_record({"type": "safety", "level": level, "msg": msg, "timestamp": datetime.now().isoformat()})

        return issues

    def check_train_script_safety(self):
        issues = []
        try:
            with open(TRAIN_SCRIPT, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            dangerous_patterns = [
                (r"import\s+os.*?os\.system\s*\(", "检测到os.system调用，可能执行危险命令"),
                (r"import\s+subprocess", "检测到subprocess导入，训练脚本不应调用外部进程"),
                (r"shutil\.rmtree", "检测到递归删除目录操作"),
                (r"open\s*\(.+['\"]w['\"]", "检测到文件写入操作（除训练输出外）"),
            ]
            for pattern, desc in dangerous_patterns:
                if re.search(pattern, content, re.DOTALL):
                    issues.append(desc)

            if "evaluate.py" in content and "import" in content:
                pass

        except FileNotFoundError:
            issues.append(f"训练脚本{TRAIN_SCRIPT}不存在")
        except Exception as e:
            issues.append(f"检查训练脚本时出错: {e}")

        return issues

    def should_emergency_stop(self):
        crit_count = sum(1 for v in self.violations if v.get("level") == "CRIT")
        if crit_count >= 3:
            return True, "连续3次以上严重安全违规，建议紧急停止"
        return False, ""


class AnomalyDetector:
    """异常检测器"""

    def __init__(self):
        self.crash_count = 0
        self.eval_fail_count = 0
        self.last_exp_time = None
        self.exp_intervals = deque(maxlen=10)

    def record_experiment(self, status):
        now = time.time()
        if self.last_exp_time is not None:
            self.exp_intervals.append(now - self.last_exp_time)
        self.last_exp_time = now

        if status == "CRASH":
            self.crash_count += 1
        elif status == "EVAL_FAIL":
            self.eval_fail_count += 1
        else:
            self.crash_count = 0
            self.eval_fail_count = 0

    def detect_crash_loop(self):
        if self.crash_count >= CRASH_THRESHOLD:
            return True, f"连续{self.crash_count}次崩溃，疑似崩溃循环"
        return False, ""

    def detect_eval_loop(self):
        if self.eval_fail_count >= CRASH_THRESHOLD:
            return True, f"连续{self.eval_fail_count}次评估失败"
        return False, ""

    def detect_stuck(self, exp_history):
        if len(exp_history) < STAGNATION_THRESHOLD:
            return False, ""
        recent = exp_history[-STAGNATION_THRESHOLD:]
        cds_values = [e.get("CDS", e.get("cds", 0)) for e in recent]
        if any(v <= 0 for v in cds_values):
            return False, ""
        if len(set([round(v, 3) for v in cds_values])) == 1:
            return True, f"连续{STAGNATION_THRESHOLD}次CDS完全相同，疑似卡死"
        max_cds = max(cds_values)
        min_cds = min(cds_values)
        if max_cds - min_cds < 0.002:
            return True, f"连续{STAGNATION_THRESHOLD}次CDS波动<0.002，疑似停滞"
        return False, ""

    def detect_time_anomaly(self):
        if len(self.exp_intervals) < 3:
            return False, ""
        avg_interval = sum(self.exp_intervals) / len(self.exp_intervals)
        if avg_interval < 30:
            return True, f"实验平均间隔{avg_interval:.0f}秒过短，可能未真正训练"
        if avg_interval > 7200:
            return True, f"实验平均间隔{avg_interval:.0f}秒过长，可能卡住"
        return False, ""


class SuggestionEngine:
    """智能建议引擎"""

    def __init__(self):
        self.suggestions_given = set()

    def suggest(self, metrics_analyzer, best_metrics, exp_num, no_improve):
        suggestions = []

        if no_improve >= 4:
            suggestions.append(("stagnation", "连续多次无提升，建议：1)切换模型架构 2)调整数据增强策略 3)检查数据质量"))

        if best_metrics.get("precision", 0) < PRECISION_GATE and best_metrics.get("recall", 0) >= RECALL_GATE:
            suggestions.append(("precision_low", "precision未达标但recall已达标，建议：1)提高置信度阈值 2)增加BOX损失权重 3)减少误检"))

        if best_metrics.get("recall", 0) < RECALL_GATE and best_metrics.get("precision", 0) >= PRECISION_GATE:
            suggestions.append(("recall_low", "recall未达标但precision已达标，建议：1)降低置信度阈值 2)增加小目标检测层 3)增强数据增强"))

        if best_metrics.get("small_obj_recall", 0) < 0.5:
            suggestions.append(("small_obj", "小目标召回率低，建议：1)使用更大imgsz 2)增加P2检测层 3)使用SAHI切片推理"))

        if best_metrics.get("counting_mae", 0) > 3.0:
            suggestions.append(("counting", "计数MAE高，建议：1)优化NMS参数 2)处理密集区域重叠 3)调整置信度阈值"))

        trend = metrics_analyzer.trend("cds")
        if trend == "declining":
            suggestions.append(("declining", "CDS呈下降趋势，建议：1)回退到最佳配置 2)减小学习率 3)增加正则化"))

        overfit, msg = metrics_analyzer.detect_overfitting()
        if overfit:
            suggestions.append(("overfitting", f"过拟合警告: {msg}，建议：1)增加dropout 2)减少epoch 3)增强数据增强"))

        cheating_warnings = metrics_analyzer.detect_metric_cheating(best_metrics)
        for w in cheating_warnings:
            suggestions.append(("cheating", w))

        new_suggestions = []
        for key, msg in suggestions:
            if key not in self.suggestions_given:
                self.suggestions_given.add(key)
                new_suggestions.append((key, msg))

        return new_suggestions


class ReportGenerator:
    """审计报告生成器"""

    def __init__(self, supervisor):
        self.supervisor = supervisor

    def generate_report(self):
        lines = []
        lines.append("# 监督智能体审计报告")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        lines.append("## 1. 实验概览\n")
        exp_count = len(self.supervisor.exp_history)
        keep_count = sum(1 for e in self.supervisor.exp_history if e.get("_status") == "KEEP")
        discard_count = sum(1 for e in self.supervisor.exp_history if e.get("_status") == "DISCARD")
        crash_count = sum(1 for e in self.supervisor.exp_history if e.get("_status") in ("CRASH", "EVAL_FAIL"))
        lines.append(f"- 总实验数: {exp_count}")
        lines.append(f"- 保留(KEEP): {keep_count}")
        lines.append(f"- 丢弃(DISCARD): {discard_count}")
        lines.append(f"- 崩溃/失败: {crash_count}")
        if exp_count > 0:
            lines.append(f"- 成功率: {(keep_count / exp_count) * 100:.1f}%\n")

        lines.append("## 2. 最佳指标\n")
        bm = self.supervisor.best_metrics
        if bm:
            for k, v in bm.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

            lines.append("## 3. Quality Gates\n")
            p = bm.get("precision", 0)
            r = bm.get("recall", 0)
            lines.append(f"- Precision >= {PRECISION_GATE}: {'✅ PASS' if p >= PRECISION_GATE else '❌ FAIL'} ({p:.4f})")
            lines.append(f"- Recall >= {RECALL_GATE}: {'✅ PASS' if r >= RECALL_GATE else '❌ FAIL'} ({r:.4f})")
            lines.append(f"- CDS >= {TARGET_CDS}: {'✅ PASS' if bm.get('CDS', bm.get('cds', 0)) >= TARGET_CDS else '❌ FAIL'} ({bm.get('CDS', bm.get('cds', 0)):.4f})")
            lines.append("")

        lines.append("## 4. 趋势分析\n")
        ma = self.supervisor.metrics_analyzer
        for key in ["cds", "mAP50", "mAP50-95", "precision", "recall", "small_obj_recall", "counting_mae"]:
            t = ma.trend(key)
            v = ma.volatility(key)
            lines.append(f"- {key}: 趋势={t}, 波动={v:.4f}")
        lines.append("")

        lines.append("## 5. 决策审核\n")
        flagged = [d for d in self.supervisor.decision_auditor.decisions if d.get("verdict") == "FLAGGED"]
        approved = [d for d in self.supervisor.decision_auditor.decisions if d.get("verdict") == "APPROVED"]
        lines.append(f"- 审核通过: {len(approved)}")
        lines.append(f"- 标记问题: {len(flagged)}")
        if flagged:
            lines.append("\n### 标记的决策:\n")
            for d in flagged[-10:]:
                lines.append(f"- 实验#{d.get('exp_num', '?')} {d.get('action', '?')}: {d.get('issues', [])}")
        lines.append("")

        lines.append("## 6. 安全事件\n")
        safety_events = self.supervisor.safety_guard.violations
        if safety_events:
            for v in safety_events[-20:]:
                lines.append(f"- [{v.get('level', '?')}] {v.get('msg', '?')}")
        else:
            lines.append("- 无安全事件")
        lines.append("")

        lines.append("## 7. 异常检测\n")
        ad = self.supervisor.anomaly_detector
        crash_loop, crash_msg = ad.detect_crash_loop()
        eval_loop, eval_msg = ad.detect_eval_loop()
        stuck, stuck_msg = ad.detect_stuck(self.supervisor.exp_history)
        time_anom, time_msg = ad.detect_time_anomaly()
        lines.append(f"- 崩溃循环: {'🚨 ' + crash_msg if crash_loop else '✅ 正常'}")
        lines.append(f"- 评估失败循环: {'🚨 ' + eval_msg if eval_loop else '✅ 正常'}")
        lines.append(f"- 停滞检测: {'⚠️ ' + stuck_msg if stuck else '✅ 正常'}")
        lines.append(f"- 时间异常: {'⚠️ ' + time_msg if time_anom else '✅ 正常'}")
        lines.append("")

        lines.append("## 8. 建议\n")
        suggestions = self.supervisor.suggestion_engine.suggest(
            ma, bm or {}, exp_count,
            self.supervisor.consecutive_no_improve
        )
        if suggestions:
            for key, msg in suggestions:
                lines.append(f"- [{key}] {msg}")
        else:
            lines.append("- 暂无新建议")
        lines.append("")

        lines.append("## 9. 实验历史\n")
        lines.append("| # | CDS | mAP50 | Precision | Recall | Status |")
        lines.append("|---|-----|-------|-----------|--------|--------|")
        for i, e in enumerate(self.supervisor.exp_history[-30:]):
            cds_val = e.get("CDS", e.get("cds", 0))
            m50 = e.get("mAP50", 0)
            p = e.get("precision", 0)
            r = e.get("recall", 0)
            s = e.get("_status", "?")
            lines.append(f"| {i} | {cds_val:.4f} | {m50:.4f} | {p:.4f} | {r:.4f} | {s} |")
        lines.append("")

        report = "\n".join(lines)
        try:
            with open(AUDIT_REPORT, "w", encoding="utf-8") as f:
                f.write(report)
            log(f"审计报告已写入 {AUDIT_REPORT}", "OK")
        except Exception as e:
            log(f"写入报告失败: {e}", "WARN")

        return report


class SupervisorAgent:
    """监督智能体主类"""

    def __init__(self, poll_interval=POLL_INTERVAL):
        self.metrics_analyzer = MetricsAnalyzer()
        self.decision_auditor = DecisionAuditor()
        self.safety_guard = SafetyGuard()
        self.anomaly_detector = AnomalyDetector()
        self.suggestion_engine = SuggestionEngine()
        self.report_generator = ReportGenerator(self)

        self.exp_history = []
        self.best_metrics = {}
        self.best_cds = -1.0
        self.consecutive_no_improve = 0
        self.total_experiments = 0
        self.running = True
        self._last_results_size = 0
        self._last_metrics_mtime = 0
        self.poll_interval = poll_interval

    def load_existing_results(self):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            if not lines:
                return

            first_line = lines[0].strip()
            has_header = any(c.isalpha() and c not in "._" for c in first_line.split("\t")[0])

            if has_header:
                header = first_line.split("\t")
                data_lines = lines[1:]
            else:
                num_cols = len(first_line.split("\t"))
                if num_cols <= 6:
                    header = ["exp", "mAP50", "Recall", "Precision", "score", "status"]
                else:
                    header = ["exp", "date", "description", "CDS", "mAP50", "mAP50-95",
                              "precision", "recall", "small_obj_recall", "counting_mae",
                              "latency_ms", "status"]
                data_lines = lines

            for line in data_lines:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                metrics = {}
                for i, h in enumerate(header):
                    if i < len(parts):
                        try:
                            metrics[h] = float(parts[i])
                        except (ValueError, IndexError):
                            metrics[h] = parts[i]
                if "status" in metrics:
                    metrics["_status"] = str(metrics.pop("status"))
                elif len(parts) > len(header) - 1:
                    metrics["_status"] = parts[-1]

                self.exp_history.append(metrics)
                cds = metrics.get("CDS", metrics.get("cds", metrics.get("score", 0)))
                if isinstance(cds, (int, float)) and cds > self.best_cds:
                    self.best_cds = cds
                    self.best_metrics = metrics.copy()

                self.metrics_analyzer.add(metrics)

            self.total_experiments = len(self.exp_history)
            self._last_results_size = len(lines)
            log(f"已加载 {self.total_experiments} 条历史记录，最佳CDS={self.best_cds:.4f}", "INFO")

        except FileNotFoundError:
            log("未找到历史结果文件，从零开始监控", "INFO")
        except Exception as e:
            log(f"加载历史结果出错: {e}", "WARN")

    def load_latest_metrics(self):
        try:
            mtime = os.path.getmtime(METRICS_FILE)
            if mtime <= self._last_metrics_mtime:
                return None
            self._last_metrics_mtime = mtime

            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            cds = metrics.get("CDS", metrics.get("cds", 0))
            log(f"检测到新评估结果: CDS={cds:.4f}", "AUDIT")
            return metrics
        except FileNotFoundError:
            return None
        except Exception as e:
            log(f"读取指标文件出错: {e}", "WARN")
            return None

    def audit_new_result(self, metrics, status="UNKNOWN"):
        cds = metrics.get("CDS", metrics.get("cds", 0))
        metrics["_status"] = status
        self.exp_history.append(metrics)
        self.total_experiments += 1
        self.metrics_analyzer.add(metrics)

        self.anomaly_detector.record_experiment(status)

        if status == "KEEP":
            if cds > self.best_cds + CDS_IMPROVE_MIN:
                verdict, issues = self.decision_auditor.audit_keep_decision(
                    metrics, self.best_metrics, self.total_experiments
                )
                if verdict == "FLAGGED":
                    for issue in issues:
                        log(f"KEEP决策问题: {issue}", "WARN")
                else:
                    log(f"KEEP决策审核通过: CDS={cds:.4f}", "OK")

                self.best_cds = cds
                self.best_metrics = metrics.copy()
                self.consecutive_no_improve = 0
            else:
                log(f"KEEP但CDS未显著提升: {cds:.4f} vs {self.best_cds:.4f}", "WARN")

        elif status == "DISCARD":
            verdict, issues = self.decision_auditor.audit_discard_decision(
                metrics, self.best_metrics, self.total_experiments
            )
            if verdict == "FLAGGED":
                for issue in issues:
                    log(f"DISCARD决策问题: {issue}", "WARN")
            self.consecutive_no_improve += 1

        elif status in ("CRASH", "EVAL_FAIL"):
            log(f"实验失败: {status}", "CRIT")
            self.consecutive_no_improve += 1

        cheating_warnings = self.metrics_analyzer.detect_metric_cheating(metrics)
        for w in cheating_warnings:
            log(f"指标异常: {w}", "WARN")

        overfit, msg = self.metrics_analyzer.detect_overfitting()
        if overfit:
            log(f"过拟合警告: {msg}", "WARN")

        suggestions = self.suggestion_engine.suggest(
            self.metrics_analyzer, self.best_metrics,
            self.total_experiments, self.consecutive_no_improve
        )
        for key, msg in suggestions:
            log(f"建议[{key}]: {msg}", "SUGGEST")

    def check_safety(self):
        issues = self.safety_guard.check_hardware()
        for level, msg in issues:
            if level == "CRIT":
                log(f"安全告警: {msg}", "CRIT")
            else:
                log(f"安全警告: {msg}", "WARN")

        should_stop, reason = self.safety_guard.should_emergency_stop()
        if should_stop:
            log(f"紧急停止建议: {reason}", "CRIT")
            return True

        code_issues = self.safety_guard.check_train_script_safety()
        for issue in code_issues:
            log(f"代码安全: {issue}", "WARN")

        return False

    def check_anomalies(self):
        crash_loop, msg = self.anomaly_detector.detect_crash_loop()
        if crash_loop:
            log(f"异常: {msg}", "CRIT")

        eval_loop, msg = self.anomaly_detector.detect_eval_loop()
        if eval_loop:
            log(f"异常: {msg}", "CRIT")

        stuck, msg = self.anomaly_detector.detect_stuck(self.exp_history)
        if stuck:
            log(f"异常: {msg}", "WARN")

        time_anom, msg = self.anomaly_detector.detect_time_anomaly()
        if time_anom:
            log(f"异常: {msg}", "WARN")

    def monitor_loop(self):
        log("=" * 60, "INFO")
        log("    监督智能体启动 — 监控模式", "INFO")
        log("=" * 60, "INFO")

        self.load_existing_results()

        log(f"历史实验: {self.total_experiments}", "INFO")
        log(f"最佳CDS: {self.best_cds:.4f}", "INFO")
        log(f"轮询间隔: {self.poll_interval}秒", "INFO")
        log("", "INFO")

        while self.running:
            try:
                emergency = self.check_safety()
                if emergency:
                    log("建议紧急停止训练智能体！", "CRIT")

                self.check_anomalies()

                metrics = self.load_latest_metrics()
                if metrics:
                    self.audit_new_result(metrics)

                if self.total_experiments > 0 and self.total_experiments % 5 == 0:
                    self.report_generator.generate_report()

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                log("收到中断信号，生成最终报告...", "INFO")
                self.running = False
            except Exception as e:
                log(f"监控循环出错: {e}", "WARN")
                time.sleep(self.poll_interval)

        self.report_generator.generate_report()
        log("监督智能体已停止", "INFO")

    def run_audit(self):
        log("=" * 60, "INFO")
        log("    监督智能体 — 一次性审计模式", "INFO")
        log("=" * 60, "INFO")

        self.load_existing_results()

        for metrics in self.exp_history:
            status = metrics.get("_status", "UNKNOWN")
            self.metrics_analyzer.add(metrics)
            self.anomaly_detector.record_experiment(status)

        log(f"审计 {self.total_experiments} 条实验记录", "AUDIT")

        self.check_safety()
        self.check_anomalies()

        suggestions = self.suggestion_engine.suggest(
            self.metrics_analyzer, self.best_metrics,
            self.total_experiments, self.consecutive_no_improve
        )
        for key, msg in suggestions:
            log(f"建议[{key}]: {msg}", "SUGGEST")

        report = self.report_generator.generate_report()
        print("\n" + report)
        return report

    def run_report_only(self):
        log("生成审计报告...", "INFO")
        self.load_existing_results()
        for metrics in self.exp_history:
            self.metrics_analyzer.add(metrics)
        report = self.report_generator.generate_report()
        print("\n" + report)
        return report

    def run_watchdog(self):
        log("=" * 60, "INFO")
        log("    监督智能体 — 守护模式（可中断训练）", "INFO")
        log("=" * 60, "INFO")

        self.load_existing_results()

        while self.running:
            try:
                emergency = self.check_safety()
                if emergency:
                    log("🚨 安全违规达到临界，尝试停止训练进程...", "CRIT")
                    self._kill_training_processes()

                self.check_anomalies()

                crash_loop, msg = self.anomaly_detector.detect_crash_loop()
                if crash_loop:
                    # 不要在这里杀正在运行的训练：活着的 train.py 并不是"正在崩溃"的那个，
                    # 把它在 DataLoader 子进程 spawn 阶段杀掉只会制造新的崩溃
                    # (_pickle.UnpicklingError: pickle data was truncated)。真正的崩溃循环
                    # 由 Orchestrator 自己的连续失败熔断处理，这里只上报异常。
                    log(f"⚠️ {msg}（不终止正在运行的训练，交由 Orchestrator 失败熔断处理）", "WARN")

                metrics = self.load_latest_metrics()
                if metrics:
                    self.audit_new_result(metrics)

                if self.total_experiments > 0 and self.total_experiments % 3 == 0:
                    self.report_generator.generate_report()

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                log("收到中断信号，生成最终报告...", "INFO")
                self.running = False
            except Exception as e:
                log(f"守护循环出错: {e}", "WARN")
                time.sleep(self.poll_interval)

        self.report_generator.generate_report()
        log("监督智能体已停止", "INFO")

    def _kill_training_processes(self):
        if psutil is None:
            log("psutil不可用，无法终止进程", "WARN")
            return
        try:
            current_pid = os.getpid()
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if "train.py" in cmdline and proc.info["pid"] != current_pid:
                        log(f"终止训练进程: PID={proc.info['pid']}", "CRIT")
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except psutil.TimeoutExpired:
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            log(f"终止进程出错: {e}", "WARN")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="监督智能体 — 审核、监控、守护训练智能体")
    parser.add_argument("--audit", action="store_true", help="对已有结果做一次性审计")
    parser.add_argument("--report", action="store_true", help="仅生成审计报告")
    parser.add_argument("--watchdog", action="store_true", help="守护模式（可中断训练）")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="轮询间隔（秒）")
    args = parser.parse_args()

    poll_interval = args.interval

    supervisor = SupervisorAgent(poll_interval=poll_interval)

    if args.audit:
        supervisor.run_audit()
    elif args.report:
        supervisor.run_report_only()
    elif args.watchdog:
        supervisor.run_watchdog()
    else:
        supervisor.monitor_loop()


if __name__ == "__main__":
    main()
