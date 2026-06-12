"""
architecture_supervisor.py — 架构合规监督智能体 (v2)

职责:
  1. 读取 architecture_v2.md 作为"架构宪法"
  2. 监督 Researcher / Curator / Triage 三个智能体是否按 v2 架构设计执行
  3. 校验智能体输出是否符合 JSON Schema 合约
  4. 校验 Researcher 的 config_patch 是否锁定到 28 变量白名单 + MODEL 路径约束
  5. 校验 Orchestrator 流程是否遵循 v2 架构（双阶段评估、新终止条件等）
  6. 校验代码结构是否与 v2 架构文档一致
  7. 校验预算守门 / HITL 门是否正确触发
  8. 校验 LLM 后端是否为纯 Ollama（不加 anthropic）
  9. 生成合规报告，标记违规项

运行方式:
  python -m autoresearch_v2.agents.architecture_supervisor
  python -m autoresearch_v2.agents.architecture_supervisor --check code
  python -m autoresearch_v2.agents.architecture_supervisor --check decisions
  python -m autoresearch_v2.agents.architecture_supervisor --check flow
  python -m autoresearch_v2.agents.architecture_supervisor --check all
  python -m autoresearch_v2.agents.architecture_supervisor --watch
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

ARCHITECTURE_V2_MD = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "architecture_v2.md"
))
V2_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
))
PROJECT_ROOT = os.path.normpath(os.path.join(V2_ROOT, ".."))

COMPLIANCE_LOG = os.path.join(V2_ROOT, "state", "compliance_audit.jsonl")

TRAIN_PY_WHITELIST_VARS = [
    "MODEL", "DATA_YAML", "EPOCHS", "BATCH", "IMGSZ", "LR0", "LRF",
    "MOMENTUM", "WEIGHT_DECAY", "WARMUP_EPOCHS", "WARMUP_MOMENTUM",
    "BOX", "CLS", "DFL", "POSE", "KOBJ", "LABEL_SMOOTHING",
    "MOSAIC", "MIXUP", "COPY_PASTE", "DEGREES", "TRANSLATE",
    "SCALE", "SHEAR", "FLIPUD", "FLIPLR", "HSV_H", "HSV_S", "HSV_V",
    "ERASING", "PATIENCE", "SAVE_PERIOD", "WORKERS", "AMP",
]


def _log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "📋", "WARN": "⚠️", "CRIT": "🚨",
        "OK": "✅", "AUDIT": "🔍", "SUGGEST": "💡",
    }.get(level, "📋")
    print(f"[{ts}] [ARCH-SUPERVISOR:{level}] {prefix} {msg}")
    sys.stdout.flush()


def _write_compliance_record(record: dict):
    os.makedirs(os.path.dirname(COMPLIANCE_LOG), exist_ok=True)
    try:
        with open(COMPLIANCE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        _log(f"合规日志写入失败: {e}", "WARN")


class Violation:
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

    def __init__(self, rule_id: str, category: str, severity: str,
                 message: str, detail: str = "", fix_suggestion: str = ""):
        self.rule_id = rule_id
        self.category = category
        self.severity = severity
        self.message = message
        self.detail = detail
        self.fix_suggestion = fix_suggestion
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
            "fix_suggestion": self.fix_suggestion,
            "timestamp": self.timestamp,
        }


ARCHITECTURE_RULES = {
    "ARCH-001": {
        "name": "三智能体架构",
        "description": "系统必须且只能包含 Researcher / Data Curator / Failure Triage 三个智能体",
        "severity": Violation.CRITICAL,
        "category": "agent_structure",
    },
    "ARCH-002": {
        "name": "智能体模型分配(v2)",
        "description": "v2: 全部本地 Ollama — Researcher=qwen2.5:14b/32b, Curator=qwen2-vl:7b/llava:13b, Triage=qwen2.5:7b/phi3:14b",
        "severity": Violation.WARNING,
        "category": "agent_structure",
    },
    "ARCH-003": {
        "name": "合约优先",
        "description": "所有智能体输出必须强制 JSON Schema，智能体之间通过数据库交流，不用 free-text",
        "severity": Violation.CRITICAL,
        "category": "contract",
    },
    "ARCH-004": {
        "name": "禁止NAS",
        "description": "Researcher 的 action_type 枚举中不能包含架构修改选项",
        "severity": Violation.CRITICAL,
        "category": "agent_constraint",
    },
    "ARCH-005": {
        "name": "预算双重守门",
        "description": "agent 自我估算 + 系统硬限双重守门，estimated_cost > budget_remaining*0.3 时强制 needs_hitl=true",
        "severity": Violation.CRITICAL,
        "category": "budget",
    },
    "ARCH-006": {
        "name": "HITL门触发",
        "description": "超过阈值的决策必须经过人工审批",
        "severity": Violation.WARNING,
        "category": "hitl",
    },
    "ARCH-007": {
        "name": "终止条件(v2)",
        "description": "v2: CDS达标/平台期/预算耗尽/连续失败/人工stop/pipeline-tile背离/LLM失败率>30%",
        "severity": Violation.CRITICAL,
        "category": "orchestrator",
    },
    "ARCH-008": {
        "name": "HPO用Optuna串行TPE",
        "description": "v2: 超参搜索必须用 Optuna 串行 TPE（单 GPU 约束）",
        "severity": Violation.WARNING,
        "category": "deterministic_tools",
    },
    "ARCH-009": {
        "name": "Triage置信度门",
        "description": "confidence < 0.7 一律 escalate_human",
        "severity": Violation.CRITICAL,
        "category": "triage",
    },
    "ARCH-010": {
        "name": "Curator图片限制",
        "description": "每次调用最多 K=20 张图",
        "severity": Violation.WARNING,
        "category": "curator",
    },
    "ARCH-011": {
        "name": "状态层6张表(v2)",
        "description": "SQLite 必须包含 experiments/decisions/data_issues/budget/artifacts/hitl_gates 6张表，experiments 用 git_sha PK",
        "severity": Violation.CRITICAL,
        "category": "state_layer",
    },
    "ARCH-012": {
        "name": "Orchestrator单线程",
        "description": "Orchestrator 是单线程主循环驱动器",
        "severity": Violation.WARNING,
        "category": "orchestrator",
    },
    "ARCH-013": {
        "name": "智能体不直接调train.py",
        "description": "Researcher 不直接调 train.py，而是返回结构化指令给 Orchestrator 执行",
        "severity": Violation.CRITICAL,
        "category": "agent_constraint",
    },
    "ARCH-014": {
        "name": "指标回归检测用规则",
        "description": "指标回归检测用简单规则，不需要 LLM 判断",
        "severity": Violation.WARNING,
        "category": "deterministic_tools",
    },
    "ARCH-015": {
        "name": "Curator输出落data_issues",
        "description": "Curator 输出必须写入 data_issues 表，低置信度 issue 必须人工 review",
        "severity": Violation.WARNING,
        "category": "curator",
    },
    "ARCH-016": {
        "name": "文件结构合规(v2)",
        "description": "v2: 文件结构必须与架构文档定义一致（含 llm_client.py/_json_extract.py/dashboard.py/events.py）",
        "severity": Violation.WARNING,
        "category": "file_structure",
    },
    "ARCH-017": {
        "name": "数据质量校验必跑",
        "description": "每次 add_training_data 后必须跑数据质量校验",
        "severity": Violation.WARNING,
        "category": "deterministic_tools",
    },
    "ARCH-018": {
        "name": "决策审计日志不可变",
        "description": "decisions 表是不可变审计日志，不能 UPDATE/DELETE",
        "severity": Violation.CRITICAL,
        "category": "state_layer",
    },
    "ARCH-019": {
        "name": "平台期检测",
        "description": "连续5轮CDS提升<0.5%触发平台期检测",
        "severity": Violation.WARNING,
        "category": "orchestrator",
    },
    "ARCH-020": {
        "name": "连续失败上限",
        "description": "累计连续失败>=3次必须停止迭代",
        "severity": Violation.CRITICAL,
        "category": "orchestrator",
    },
    "ARCH-021": {
        "name": "Researcher config_patch 白名单",
        "description": "v2: config_patch/config_diff 只允许 train.py 的 28 个白名单变量",
        "severity": Violation.CRITICAL,
        "category": "agent_constraint",
    },
    "ARCH-022": {
        "name": "MODEL路径强约束",
        "description": "v2: MODEL 必须形如 person_dataset/*.pt，禁止裸名触发网络下载",
        "severity": Violation.CRITICAL,
        "category": "agent_constraint",
    },
    "ARCH-023": {
        "name": "BATCH上限约束",
        "description": "v2: BATCH <= 16 @ imgsz=1280（单 GPU 显存约束）",
        "severity": Violation.WARNING,
        "category": "agent_constraint",
    },
    "ARCH-024": {
        "name": "双阶段评估",
        "description": "v2: 每轮迭代必须跑 tile-mode + pipeline-mode 两套评估",
        "severity": Violation.CRITICAL,
        "category": "orchestrator",
    },
    "ARCH-025": {
        "name": "双写legacy文件",
        "description": "v2: SQLite 为主存储，同时写 results.tsv/last_metrics.json/status.md/suggestions.md 维持兼容",
        "severity": Violation.WARNING,
        "category": "state_layer",
    },
    "ARCH-026": {
        "name": "纯Ollama后端",
        "description": "v2: LLM 全部本地 Ollama，不加 anthropic SDK，requirements.txt 不含 anthropic",
        "severity": Violation.WARNING,
        "category": "agent_structure",
    },
    "ARCH-027": {
        "name": "Curator风险预案",
        "description": "v2: 本地视觉模型结果必须标注 confidence，低置信度 issue 必须人工 review 后才进入数据采集队列",
        "severity": Violation.WARNING,
        "category": "curator",
    },
    "ARCH-028": {
        "name": "Optuna串行约束",
        "description": "v2: 单 GPU 串行约束，Optuna 必须用串行 TPE，不并行 trial",
        "severity": Violation.WARNING,
        "category": "deterministic_tools",
    },
    "ARCH-029": {
        "name": "pipeline-tile背离检测",
        "description": "v2: pipeline_metrics 与 tile_metrics 严重背离(>10%差距3轮)时触发 HITL",
        "severity": Violation.WARNING,
        "category": "orchestrator",
    },
    "ARCH-030": {
        "name": "LLM失败率halt",
        "description": "v2: LLM 调用失败率 > 30% 时必须 halt",
        "severity": Violation.CRITICAL,
        "category": "orchestrator",
    },
    "ARCH-031": {
        "name": "evaluate.py FP/FN导出",
        "description": "v2: evaluate.py 必须支持 --export-fpfn 参数导出 per-image FP/FN 数据",
        "severity": Violation.WARNING,
        "category": "deterministic_tools",
    },
    "ARCH-032": {
        "name": "ollama_runner并存",
        "description": "v2: ollama_runner.py 完整保留在原位，v2 系统在子目录独立运行",
        "severity": Violation.INFO,
        "category": "file_structure",
    },
}


class AgentOutputValidator:
    """校验智能体输出是否符合 v2 架构合约"""

    def __init__(self, state=None):
        self.state = state
        self.violations = []
        self._schemas = {}
        self._load_schemas()

    def _load_schemas(self):
        schema_dir = os.path.join(V2_ROOT, "schemas")
        if not os.path.exists(schema_dir):
            return
        for fname in os.listdir(schema_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(schema_dir, fname), "r", encoding="utf-8") as f:
                        self._schemas[fname] = json.load(f)
                except Exception:
                    pass

    def validate_decision(self, decision: dict, agent_name: str) -> list[Violation]:
        violations = []

        if not isinstance(decision, dict):
            v = Violation(
                "ARCH-003", "contract", Violation.CRITICAL,
                f"{agent_name} 输出不是 JSON dict",
                f"实际类型: {type(decision).__name__}",
                "确保智能体输出结构化 JSON"
            )
            violations.append(v)
            return violations

        if agent_name == "researcher":
            violations.extend(self._validate_researcher_output(decision))
        elif agent_name == "curator":
            violations.extend(self._validate_curator_output(decision))
        elif agent_name == "triage":
            violations.extend(self._validate_triage_output(decision))

        return violations

    def _validate_researcher_output(self, decision: dict) -> list[Violation]:
        violations = []
        schema = self._schemas.get("researcher_decision.json", {})
        valid_actions = schema.get("properties", {}).get("action_type", {}).get("enum", [])

        if not valid_actions:
            valid_actions = [
                "patch_train_config", "hpo_sweep", "add_training_data",
                "change_tiling", "freeze_backbone", "adjust_augmentation",
                "adjust_loss_weights", "change_model_size", "stop",
            ]

        action = decision.get("action_type", "")
        if valid_actions and action not in valid_actions:
            v = Violation(
                "ARCH-004", "agent_constraint", Violation.CRITICAL,
                f"Researcher 输出了非法 action_type: {action}",
                f"合法值: {valid_actions}",
                "检查 Researcher prompt 是否被篡改，或 LLM 产生了幻觉"
            )
            violations.append(v)

        nas_keywords = ["nas", "architecture_change", "modify_backbone", "change_architecture"]
        if action.lower() in nas_keywords:
            v = Violation(
                "ARCH-004", "agent_constraint", Violation.CRITICAL,
                "Researcher 试图执行 NAS（架构搜索），架构设计禁止此操作",
                f"action_type={action}",
                "从 action_type 枚举中移除 NAS 相关选项"
            )
            violations.append(v)

        required_fields = schema.get("required", [])
        if not required_fields:
            required_fields = ["action_type", "rationale", "estimated_gpu_minutes", "needs_hitl"]
        for field in required_fields:
            if field not in decision:
                v = Violation(
                    "ARCH-003", "contract", Violation.CRITICAL,
                    f"Researcher 输出缺少必填字段: {field}",
                    f"decision keys: {list(decision.keys())}",
                    "确保 LLM 输出完整 JSON"
                )
                violations.append(v)

        config_patch = decision.get("config_patch", decision.get("config_diff", {}))
        if config_patch and isinstance(config_patch, dict):
            violations.extend(self._validate_config_patch(config_patch))

        if "needs_hitl" in decision and decision["needs_hitl"] is None:
            v = Violation(
                "ARCH-003", "contract", Violation.WARNING,
                "Researcher needs_hitl 字段为 None，应为布尔值",
            )
            violations.append(v)

        return violations

    def _validate_config_patch(self, config_patch: dict) -> list[Violation]:
        violations = []
        whitelist_set = set(TRAIN_PY_WHITELIST_VARS)

        for key in config_patch:
            if key not in whitelist_set:
                v = Violation(
                    "ARCH-021", "agent_constraint", Violation.CRITICAL,
                    f"Researcher config_patch 包含非白名单变量: {key}",
                    f"白名单变量: {sorted(whitelist_set)}",
                    f"移除 {key}，只允许 train.py 的 28 个白名单变量"
                )
                violations.append(v)

            if key == "MODEL":
                model_val = config_patch[key]
                if isinstance(model_val, str):
                    if not model_val.startswith("person_dataset/"):
                        v = Violation(
                            "ARCH-022", "agent_constraint", Violation.CRITICAL,
                            f"MODEL 路径违反强约束: {model_val}",
                            "v2: MODEL 必须形如 person_dataset/*.pt，禁止裸名触发网络下载",
                            f"改为 person_dataset/{model_val} 或 person_dataset/<filename>.pt"
                        )
                        violations.append(v)
                    if not model_val.endswith(".pt"):
                        v = Violation(
                            "ARCH-022", "agent_constraint", Violation.WARNING,
                            f"MODEL 路径不以 .pt 结尾: {model_val}",
                        )
                        violations.append(v)

            if key == "BATCH":
                try:
                    batch_val = int(config_patch[key])
                    if batch_val > 16:
                        v = Violation(
                            "ARCH-023", "agent_constraint", Violation.WARNING,
                            f"BATCH={batch_val} 超过上限 16 (imgsz=1280 单GPU约束)",
                            "v2: BATCH <= 16 @ imgsz=1280",
                            "将 BATCH 降为 16 或以下"
                        )
                        violations.append(v)
                except (TypeError, ValueError):
                    pass

        return violations

    def _validate_curator_output(self, report: dict) -> list[Violation]:
        violations = []
        schema = self._schemas.get("curator_report.json", {})
        required_fields = schema.get("required", [])
        if not required_fields:
            required_fields = ["worst_failure_modes", "suspected_label_errors", "underrepresented_scenarios"]

        for field in required_fields:
            if field not in report:
                v = Violation(
                    "ARCH-003", "contract", Violation.CRITICAL,
                    f"Curator 输出缺少必填字段: {field}",
                )
                violations.append(v)

        failure_modes = report.get("worst_failure_modes", [])
        if not isinstance(failure_modes, list):
            v = Violation(
                "ARCH-003", "contract", Violation.CRITICAL,
                "Curator worst_failure_modes 不是数组",
                f"实际类型: {type(failure_modes).__name__}",
            )
            violations.append(v)

        return violations

    def _validate_triage_output(self, diagnosis: dict) -> list[Violation]:
        violations = []
        schema = self._schemas.get("triage_diagnosis.json", {})
        valid_actions = schema.get("properties", {}).get("recommended_action", {}).get("enum", [])
        if not valid_actions:
            valid_actions = ["rollback", "retry_with_fix", "escalate_human"]

        action = diagnosis.get("recommended_action", "")
        if valid_actions and action not in valid_actions:
            v = Violation(
                "ARCH-003", "contract", Violation.CRITICAL,
                f"Triage 输出了非法 recommended_action: {action}",
                f"合法值: {valid_actions}",
            )
            violations.append(v)

        confidence = diagnosis.get("confidence", 0)
        try:
            conf_val = float(confidence)
        except (TypeError, ValueError):
            conf_val = -1

        if not (0.0 <= conf_val <= 1.0):
            v = Violation(
                "ARCH-003", "contract", Violation.CRITICAL,
                f"Triage confidence 超出范围: {confidence}",
            )
            violations.append(v)
        elif conf_val < 0.7 and action != "escalate_human":
            v = Violation(
                "ARCH-009", "triage", Violation.CRITICAL,
                f"Triage confidence={conf_val:.2f}<0.7 但未 escalate_human (action={action})",
                "架构规定: confidence < 0.7 一律 escalate",
                "在 Triage 输出校验中强制将低置信度决策升级为 escalate_human"
            )
            violations.append(v)

        return violations

    def validate_all_decisions_from_db(self) -> list[Violation]:
        if not self.state:
            return []

        violations = []
        decisions = self.state.get_decisions(limit=200)

        for d in decisions:
            agent = d.get("made_by_agent", "")
            action = d.get("action_type", "")
            rationale = d.get("rationale", "")

            if agent == "researcher":
                valid_actions = [
                    "patch_train_config", "hpo_sweep", "add_training_data",
                    "change_tiling", "freeze_backbone", "adjust_augmentation",
                    "adjust_loss_weights", "change_model_size", "stop",
                    "quick_diagnosis",
                ]
                if action not in valid_actions:
                    v = Violation(
                        "ARCH-004", "agent_constraint", Violation.CRITICAL,
                        f"Researcher 记录了非法 action: {action}",
                        f"run_id={d.get('run_id','')}, decision_id={d.get('decision_id','')}",
                    )
                    violations.append(v)

            elif agent == "triage":
                if action not in ("rollback", "retry_with_fix", "escalate_human",
                                  "quick_diagnosis"):
                    v = Violation(
                        "ARCH-003", "contract", Violation.WARNING,
                        f"Triage 记录了非常规 action: {action}",
                    )
                    violations.append(v)

            if not rationale or len(rationale.strip()) < 5:
                v = Violation(
                    "ARCH-003", "contract", Violation.WARNING,
                    f"{agent} 的决策 rationale 过短或为空",
                    f"decision_id={d.get('decision_id','')}",
                    "每个决策必须有充分的理由说明"
                )
                violations.append(v)

        return violations


class OrchestratorFlowValidator:
    """校验 Orchestrator 流程是否遵循 v2 架构规定"""

    def __init__(self, state=None):
        self.state = state
        self.violations = []

    def _read_orchestrator_code(self) -> str:
        orch_path = os.path.join(V2_ROOT, "orchestrator.py")
        if not os.path.exists(orch_path):
            return ""
        with open(orch_path, "r", encoding="utf-8") as f:
            return f.read()

    def validate_terminations(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            v = Violation("ARCH-007", "orchestrator", Violation.CRITICAL, "orchestrator.py 不存在")
            violations.append(v)
            return violations

        required_terminations = {
            "CDS达标": [r"target_cds", r"CDS.*>=", r"cds.*>=.*target"],
            "平台期": [r"stagnat", r"STAGNATION", r"平台期"],
            "预算耗尽": [r"BudgetExceeded", r"budget.*exceed", r"预算"],
            "连续失败": [r"consecutive_fail", r"MAX_CONSECUTIVE_FAILS", r"连续.*失败"],
            "人工stop": [r"stop_signal", r"STOP", r"人工.*停止"],
            "pipeline-tile背离": [r"pipeline.*tile.*背离", r"tile.*pipeline.*diverg", r"pipeline_metrics.*tile_metrics"],
            "LLM失败率halt": [r"llm.*fail.*rate", r"LLM.*失败率", r"halt.*llm", r"llm_failure"],
        }

        for name, patterns in required_terminations.items():
            found = any(re.search(p, code, re.IGNORECASE) for p in patterns)
            if not found:
                severity = Violation.CRITICAL if name in ("CDS达标", "预算耗尽", "连续失败", "人工stop") else Violation.WARNING
                v = Violation(
                    "ARCH-007", "orchestrator", severity,
                    f"Orchestrator 缺少终止条件: {name}",
                    "v2 架构要求终止条件全部实现",
                    f"在 orchestrator.py 中实现 {name} 终止条件"
                )
                violations.append(v)

        return violations

    def validate_budget_gate(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            return violations

        if "budget.reserve" not in code and "budget.check_remaining" not in code:
            v = Violation(
                "ARCH-005", "budget", Violation.CRITICAL,
                "Orchestrator 未调用 budget.reserve() 进行预算守门",
                "架构要求双重守门: agent自我估算 + 系统硬限",
                "在 dispatch 前调用 budget.reserve()"
            )
            violations.append(v)

        if "BudgetExceeded" not in code:
            v = Violation(
                "ARCH-005", "budget", Violation.CRITICAL,
                "Orchestrator 未处理 BudgetExceeded 异常",
                "预算超限时应停止迭代",
                "try/except BudgetExceeded 并 break"
            )
            violations.append(v)

        return violations

    def validate_hitl_gate(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            return violations

        if "needs_hitl" not in code:
            v = Violation(
                "ARCH-006", "hitl", Violation.WARNING,
                "Orchestrator 未检查 needs_hitl 标志",
                "架构要求超过阈值的决策必须人工审批",
            )
            violations.append(v)

        if "hitl" not in code.lower():
            v = Violation(
                "ARCH-006", "hitl", Violation.CRITICAL,
                "Orchestrator 未集成 HITL 审批门",
            )
            violations.append(v)

        return violations

    def validate_flow_sequence(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            return violations

        flow_steps = [
            ("读取实验快照", r"get_recent_experiments|read_recent"),
            ("Researcher决策", r"researcher\.decide"),
            ("记录决策", r"log_decision"),
            ("预算预留", r"budget\.reserve"),
            ("训练调度", r"train_dispatcher\.dispatch|dispatch\(decision\)"),
            ("评估", r"eval_dispatcher\.run|evaluate"),
            ("预算提交", r"budget\.commit_actual"),
        ]

        for step_name, pattern in flow_steps:
            if not re.search(pattern, code):
                v = Violation(
                    "ARCH-012", "orchestrator", Violation.WARNING,
                    f"Orchestrator 流程缺少步骤: {step_name}",
                    "架构定义的主循环应包含此步骤",
                )
                violations.append(v)

        return violations

    def validate_stagnation_detection(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            return violations

        if "stagnat" not in code.lower():
            v = Violation(
                "ARCH-019", "orchestrator", Violation.WARNING,
                "Orchestrator 未实现平台期检测",
                "架构要求: 连续5轮CDS提升<0.5%触发平台期",
            )
            violations.append(v)

        if "consecutive_fail" not in code.lower():
            v = Violation(
                "ARCH-020", "orchestrator", Violation.CRITICAL,
                "Orchestrator 未实现连续失败计数",
                "架构要求: 累计连续失败>=3次停止迭代",
            )
            violations.append(v)

        return violations

    def validate_dual_evaluation(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            return violations

        has_tile_eval = bool(re.search(r"tile.*eval|eval.*tile", code, re.IGNORECASE))
        has_pipeline_eval = bool(re.search(r"pipeline.*eval|eval.*pipeline", code, re.IGNORECASE))

        if not has_pipeline_eval:
            eval_disp_path = os.path.join(V2_ROOT, "tools", "eval_dispatcher.py")
            if os.path.exists(eval_disp_path):
                with open(eval_disp_path, "r", encoding="utf-8") as f:
                    eval_code = f.read()
                has_pipeline_eval = bool(re.search(r"pipeline", eval_code, re.IGNORECASE))

        if not has_pipeline_eval:
            v = Violation(
                "ARCH-024", "orchestrator", Violation.WARNING,
                "未检测到 pipeline-mode 评估",
                "v2: 每轮迭代必须跑 tile-mode + pipeline-mode 两套评估",
                "在 eval_dispatcher 或 orchestrator 中添加 pipeline 评估步骤"
            )
            violations.append(v)

        return violations

    def validate_pipeline_tile_divergence(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            return violations

        if not re.search(r"pipeline.*tile.*diverg|tile.*pipeline.*背离|pipeline_metrics.*tile_metrics", code, re.IGNORECASE):
            v = Violation(
                "ARCH-029", "orchestrator", Violation.WARNING,
                "未检测到 pipeline-tile 背离检测逻辑",
                "v2: pipeline_metrics 与 tile_metrics 严重背离(>10%差距3轮)时触发 HITL",
                "添加 pipeline_metrics vs tile_metrics 背离检测"
            )
            violations.append(v)

        return violations

    def validate_llm_failure_halt(self) -> list[Violation]:
        violations = []
        code = self._read_orchestrator_code()
        if not code:
            return violations

        if not re.search(r"llm.*fail|llm_failure|失败率.*halt|halt.*llm", code, re.IGNORECASE):
            v = Violation(
                "ARCH-030", "orchestrator", Violation.WARNING,
                "未检测到 LLM 失败率 halt 逻辑",
                "v2: LLM 调用失败率 > 30% 时必须 halt",
                "添加 LLM 调用失败率统计和 halt 机制"
            )
            violations.append(v)

        return violations

    def validate_all(self) -> list[Violation]:
        violations = []
        violations.extend(self.validate_terminations())
        violations.extend(self.validate_budget_gate())
        violations.extend(self.validate_hitl_gate())
        violations.extend(self.validate_flow_sequence())
        violations.extend(self.validate_stagnation_detection())
        violations.extend(self.validate_dual_evaluation())
        violations.extend(self.validate_pipeline_tile_divergence())
        violations.extend(self.validate_llm_failure_halt())
        return violations


class CodeStructureValidator:
    """校验代码结构是否与 v2 架构文档一致"""

    EXPECTED_FILES_V2 = {
        "orchestrator.py": "主驱动，替代 ollama_runner.py",
        "db.py": "SQLite schema + DAO + 双写 legacy",
        "budget.py": "预算守门",
        "hitl.py": "人工审批门",
        "_json_extract.py": "v2: 共享 JSON 提取工具",
        "events.py": "v2: 活动事件流",
        "dashboard.py": "v2: 只读仪表盘",
        "agents/__init__.py": "agents 包",
        "agents/researcher.py": "主决策 agent（含 28 变量白名单校验）",
        "agents/curator.py": "数据策展 agent (vision)",
        "agents/triage.py": "故障定级 agent",
        "tools/__init__.py": "tools 包",
        "tools/optuna_runner.py": "HPO 搜索（串行 TPE）",
        "tools/train_dispatcher.py": "包装 train.py",
        "tools/eval_dispatcher.py": "包装 evaluate.py + FP/FN 提取",
        "schemas/researcher_decision.json": "Researcher 输出 schema",
        "schemas/curator_report.json": "Curator 输出 schema",
        "schemas/triage_diagnosis.json": "Triage 输出 schema",
        "prompts/researcher.md": "Researcher prompt",
        "prompts/curator.md": "Curator prompt",
        "prompts/triage.md": "Triage prompt",
    }

    PRESERVED_FILES = [
        "train.py",
        "evaluate.py",
        "data_quality.py",
        "build_pipeline_val.py",
        "ollama_runner.py",
    ]

    def __init__(self):
        self.violations = []

    def validate_file_structure(self) -> list[Violation]:
        violations = []

        for rel_path, desc in self.EXPECTED_FILES_V2.items():
            full_path = os.path.join(V2_ROOT, rel_path)
            if not os.path.exists(full_path):
                v = Violation(
                    "ARCH-016", "file_structure", Violation.WARNING,
                    f"缺少架构要求的文件: {rel_path}",
                    f"用途: {desc}",
                    f"创建 {rel_path}"
                )
                violations.append(v)

        return violations

    def validate_preserved_files(self) -> list[Violation]:
        violations = []

        for fname in self.PRESERVED_FILES:
            fpath = os.path.join(PROJECT_ROOT, fname)
            if not os.path.exists(fpath):
                v = Violation(
                    "ARCH-016", "file_structure", Violation.WARNING,
                    f"架构要求保留的文件不存在: {fname}",
                    "这些文件应保持不动，不被 autoresearch_v2 修改",
                )
                violations.append(v)

        return violations

    def validate_agent_count(self) -> list[Violation]:
        violations = []
        agents_dir = os.path.join(V2_ROOT, "agents")

        if not os.path.exists(agents_dir):
            v = Violation("ARCH-001", "agent_structure", Violation.CRITICAL, "agents/ 目录不存在")
            violations.append(v)
            return violations

        agent_files = [f for f in os.listdir(agents_dir)
                       if f.endswith(".py") and f != "__init__.py"
                       and not f.startswith("_") and not f.startswith("architecture_supervisor")]

        expected_agents = {"researcher.py", "curator.py", "triage.py"}
        actual_agents = set(agent_files)

        extra = actual_agents - expected_agents
        if extra:
            for f in sorted(extra):
                v = Violation(
                    "ARCH-001", "agent_structure", Violation.WARNING,
                    f"发现架构未定义的智能体文件: agents/{f}",
                    "架构只允许 Researcher / Curator / Triage 三个智能体",
                    f"确认 {f} 是否为辅助工具而非独立智能体"
                )
                violations.append(v)

        missing = expected_agents - actual_agents
        if missing:
            for f in sorted(missing):
                v = Violation(
                    "ARCH-001", "agent_structure", Violation.CRITICAL,
                    f"缺少架构要求的智能体: agents/{f}",
                )
                violations.append(v)

        return violations

    def validate_no_direct_train_call(self) -> list[Violation]:
        violations = []

        agent_files = ["researcher.py", "curator.py", "triage.py"]
        for fname in agent_files:
            fpath = os.path.join(V2_ROOT, "agents", fname)
            if not os.path.exists(fpath):
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()

            if re.search(r"subprocess\.(run|call|Popen).*train\.py", code):
                v = Violation(
                    "ARCH-013", "agent_constraint", Violation.CRITICAL,
                    f"agents/{fname} 直接调用了 train.py",
                    "架构规定: 智能体不直接调 train.py，返回结构化指令给 Orchestrator",
                    "移除直接调用，改为返回 config_diff"
                )
                violations.append(v)

            if re.search(r"import\s+train", code) and "train_dispatcher" not in code:
                v = Violation(
                    "ARCH-013", "agent_constraint", Violation.WARNING,
                    f"agents/{fname} 导入了 train 模块",
                    "智能体不应直接依赖训练逻辑",
                )
                violations.append(v)

        return violations

    def validate_decisions_table_immutable(self) -> list[Violation]:
        violations = []
        db_path = os.path.join(V2_ROOT, "db.py")

        if not os.path.exists(db_path):
            return violations

        with open(db_path, "r", encoding="utf-8") as f:
            code = f.read()

        if re.search(r"UPDATE\s+decisions\s+SET", code, re.IGNORECASE):
            v = Violation(
                "ARCH-018", "state_layer", Violation.CRITICAL,
                "db.py 包含 UPDATE decisions SET 语句",
                "decisions 表是不可变审计日志，不能修改",
                "移除对 decisions 表的 UPDATE 操作"
            )
            violations.append(v)

        if re.search(r"DELETE\s+FROM\s+decisions", code, re.IGNORECASE):
            v = Violation(
                "ARCH-018", "state_layer", Violation.CRITICAL,
                "db.py 包含 DELETE FROM decisions 语句",
                "decisions 表是不可变审计日志，不能删除",
                "移除对 decisions 表的 DELETE 操作"
            )
            violations.append(v)

        return violations

    def validate_optuna_usage(self) -> list[Violation]:
        violations = []
        optuna_path = os.path.join(V2_ROOT, "tools", "optuna_runner.py")

        if not os.path.exists(optuna_path):
            v = Violation("ARCH-008", "deterministic_tools", Violation.WARNING, "tools/optuna_runner.py 不存在")
            violations.append(v)
            return violations

        with open(optuna_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "import optuna" not in code:
            v = Violation("ARCH-008", "deterministic_tools", Violation.CRITICAL, "optuna_runner.py 未导入 optuna")
            violations.append(v)

        if "suggest_float" not in code and "suggest_int" not in code:
            v = Violation("ARCH-008", "deterministic_tools", Violation.WARNING, "optuna_runner.py 未使用 Optuna 的 suggest API")
            violations.append(v)

        return violations

    def validate_curator_constraints(self) -> list[Violation]:
        violations = []
        curator_path = os.path.join(V2_ROOT, "agents", "curator.py")

        if not os.path.exists(curator_path):
            return violations

        with open(curator_path, "r", encoding="utf-8") as f:
            code = f.read()

        max_images_match = re.search(r"max_images[^=]*=\s*(\d+)", code)
        if max_images_match:
            max_images = int(max_images_match.group(1))
            if max_images > 20:
                v = Violation(
                    "ARCH-010", "curator", Violation.WARNING,
                    f"Curator max_images={max_images} 超过架构限制 K=20",
                    "架构规定每次调用最多 K=20 张图",
                    "将 max_images 降为 20"
                )
                violations.append(v)
        else:
            v = Violation(
                "ARCH-010", "curator", Violation.WARNING,
                "Curator 未设置 max_images 限制",
                "架构规定每次调用最多 K=20 张图",
                "添加 max_images_per_call=20 参数"
            )
            violations.append(v)

        if "insert_data_issue" not in code and "data_issues" not in code:
            v = Violation(
                "ARCH-015", "curator", Violation.WARNING,
                "Curator 未将输出写入 data_issues 表",
                "架构规定 Curator 输出必须落入 data_issues 表",
            )
            violations.append(v)

        return violations

    def validate_state_tables(self) -> list[Violation]:
        violations = []
        db_path = os.path.join(V2_ROOT, "db.py")

        if not os.path.exists(db_path):
            v = Violation("ARCH-011", "state_layer", Violation.CRITICAL, "db.py 不存在")
            violations.append(v)
            return violations

        with open(db_path, "r", encoding="utf-8") as f:
            code = f.read()

        required_tables = [
            "experiments", "decisions", "data_issues",
            "budget", "artifacts", "hitl_gates"
        ]

        for table in required_tables:
            if f"CREATE TABLE IF NOT EXISTS {table}" not in code:
                v = Violation(
                    "ARCH-011", "state_layer", Violation.CRITICAL,
                    f"db.py 缺少 {table} 表定义",
                    f"架构要求6张核心表，缺少: {table}",
                )
                violations.append(v)

        if "WAL" not in code:
            v = Violation(
                "ARCH-011", "state_layer", Violation.WARNING,
                "db.py 未启用 WAL 模式",
                "架构要求 SQLite 使用 WAL 模式",
                "添加 conn.execute('PRAGMA journal_mode=WAL')"
            )
            violations.append(v)

        return violations

    def validate_ollama_only(self) -> list[Violation]:
        violations = []
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")

        if os.path.exists(req_path):
            with open(req_path, "r", encoding="utf-8") as f:
                req_content = f.read().lower()
            if "anthropic" in req_content:
                v = Violation(
                    "ARCH-026", "agent_structure", Violation.WARNING,
                    "requirements.txt 包含 anthropic 依赖",
                    "v2: LLM 全部本地 Ollama，不加 anthropic SDK",
                    "从 requirements.txt 移除 anthropic"
                )
                violations.append(v)

        for agent_file in ["researcher.py", "curator.py", "triage.py"]:
            fpath = os.path.join(V2_ROOT, "agents", agent_file)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
            if "import anthropic" in code:
                v = Violation(
                    "ARCH-026", "agent_structure", Violation.WARNING,
                    f"agents/{agent_file} 导入了 anthropic SDK",
                    "v2: LLM 全部本地 Ollama，不应依赖 anthropic",
                    "移除 anthropic 导入，改用 Ollama HTTP 客户端"
                )
                violations.append(v)

        return violations

    def validate_dual_write_legacy(self) -> list[Violation]:
        violations = []
        db_path = os.path.join(V2_ROOT, "db.py")

        if not os.path.exists(db_path):
            return violations

        with open(db_path, "r", encoding="utf-8") as f:
            code = f.read()

        legacy_writes = {
            "results.tsv": [r"results\.tsv"],
            "last_metrics.json": [r"last_metrics\.json"],
            "status.md": [r"status\.md"],
        }

        for fname, patterns in legacy_writes.items():
            if not any(re.search(p, code) for p in patterns):
                v = Violation(
                    "ARCH-025", "state_layer", Violation.WARNING,
                    f"db.py 未检测到对 {fname} 的双写",
                    "v2: SQLite 为主存储，同时写 legacy 文件维持兼容",
                    f"在 db.py 中添加对 {fname} 的双写逻辑"
                )
                violations.append(v)

        return violations

    def validate_evaluate_fpfn_export(self) -> list[Violation]:
        violations = []
        eval_path = os.path.join(PROJECT_ROOT, "evaluate.py")

        if not os.path.exists(eval_path):
            return violations

        with open(eval_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "export-fpfn" not in code and "export_fpfn" not in code:
            v = Violation(
                "ARCH-031", "deterministic_tools", Violation.WARNING,
                "evaluate.py 未支持 --export-fpfn 参数",
                "v2: evaluate.py 必须支持 FP/FN 导出给 Curator 使用",
                "在 evaluate.py 中添加 --export-fpfn DIR 参数"
            )
            violations.append(v)

        return violations

    def validate_ollama_runner_coexistence(self) -> list[Violation]:
        violations = []
        ollama_path = os.path.join(PROJECT_ROOT, "ollama_runner.py")

        if not os.path.exists(ollama_path):
            v = Violation(
                "ARCH-032", "file_structure", Violation.INFO,
                "ollama_runner.py 不存在（v2 架构要求保留作为 legacy 对照基线）",
            )
            violations.append(v)

        return violations

    def validate_optuna_serial(self) -> list[Violation]:
        violations = []
        optuna_path = os.path.join(V2_ROOT, "tools", "optuna_runner.py")

        if not os.path.exists(optuna_path):
            return violations

        with open(optuna_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "n_jobs" in code and re.search(r"n_jobs\s*[=>]\s*[2-9]", code):
            v = Violation(
                "ARCH-028", "deterministic_tools", Violation.WARNING,
                "optuna_runner.py 设置了 n_jobs > 1",
                "v2: 单 GPU 串行约束，Optuna 必须用串行 TPE",
                "移除 n_jobs 参数或设为 n_jobs=1"
            )
            violations.append(v)

        return violations

    def validate_all(self) -> list[Violation]:
        violations = []
        violations.extend(self.validate_file_structure())
        violations.extend(self.validate_preserved_files())
        violations.extend(self.validate_agent_count())
        violations.extend(self.validate_no_direct_train_call())
        violations.extend(self.validate_decisions_table_immutable())
        violations.extend(self.validate_optuna_usage())
        violations.extend(self.validate_curator_constraints())
        violations.extend(self.validate_state_tables())
        violations.extend(self.validate_ollama_only())
        violations.extend(self.validate_dual_write_legacy())
        violations.extend(self.validate_evaluate_fpfn_export())
        violations.extend(self.validate_ollama_runner_coexistence())
        violations.extend(self.validate_optuna_serial())
        return violations


class ComplianceReporter:
    """合规报告生成器"""

    def __init__(self):
        self.violations = []

    def add_violations(self, violations: list[Violation]):
        self.violations.extend(violations)

    def generate_report(self) -> str:
        lines = []
        lines.append("# 架构合规监督报告 (v2)")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        total = len(self.violations)
        critical = sum(1 for v in self.violations if v.severity == Violation.CRITICAL)
        warning = sum(1 for v in self.violations if v.severity == Violation.WARNING)
        info = sum(1 for v in self.violations if v.severity == Violation.INFO)

        lines.append("## 总体评估\n")
        if critical == 0 and warning == 0:
            lines.append("**状态: ✅ 完全合规** — 所有检查项均通过\n")
        elif critical == 0:
            lines.append(f"**状态: ⚠️ 基本合规** — {warning} 个警告\n")
        else:
            lines.append(f"**状态: 🚨 不合规** — {critical} 个严重违规, {warning} 个警告\n")

        lines.append(f"| 级别 | 数量 |")
        lines.append(f"|------|------|")
        lines.append(f"| 🚨 CRITICAL | {critical} |")
        lines.append(f"| ⚠️ WARNING | {warning} |")
        lines.append(f"| 📋 INFO | {info} |")
        lines.append(f"| **总计** | **{total}** |\n")

        by_category = {}
        for v in self.violations:
            by_category.setdefault(v.category, []).append(v)

        lines.append("## 按类别统计\n")
        lines.append("| 类别 | CRITICAL | WARNING | INFO |")
        lines.append("|------|----------|---------|------|")
        for cat in sorted(by_category.keys()):
            cat_violations = by_category[cat]
            c = sum(1 for v in cat_violations if v.severity == Violation.CRITICAL)
            w = sum(1 for v in cat_violations if v.severity == Violation.WARNING)
            i = sum(1 for v in cat_violations if v.severity == Violation.INFO)
            lines.append(f"| {cat} | {c} | {w} | {i} |")
        lines.append("")

        if self.violations:
            lines.append("## 违规详情\n")
            for v in sorted(self.violations, key=lambda x: (
                    0 if x.severity == Violation.CRITICAL else 1 if x.severity == Violation.WARNING else 2,
                    x.rule_id)):
                icon = "🚨" if v.severity == Violation.CRITICAL else "⚠️" if v.severity == Violation.WARNING else "📋"
                rule_info = ARCHITECTURE_RULES.get(v.rule_id, {})
                rule_name = rule_info.get("name", v.rule_id)

                lines.append(f"### {icon} [{v.rule_id}] {rule_name}\n")
                lines.append(f"- **严重级别**: {v.severity}")
                lines.append(f"- **类别**: {v.category}")
                lines.append(f"- **违规描述**: {v.message}")
                if v.detail:
                    lines.append(f"- **详情**: {v.detail}")
                if v.fix_suggestion:
                    lines.append(f"- **修复建议**: {v.fix_suggestion}")
                lines.append("")

        lines.append("## 架构规则检查清单 (v2)\n")
        lines.append("| 规则ID | 规则名称 | 状态 |")
        lines.append("|--------|----------|------|")

        violated_rules = {v.rule_id for v in self.violations}
        critical_rules = {v.rule_id for v in self.violations if v.severity == Violation.CRITICAL}
        warning_rules = {v.rule_id for v in self.violations if v.severity == Violation.WARNING}

        for rule_id, rule_info in sorted(ARCHITECTURE_RULES.items()):
            if rule_id in critical_rules:
                status = "🚨 CRITICAL"
            elif rule_id in warning_rules:
                status = "⚠️ WARNING"
            else:
                status = "✅ PASS"
            lines.append(f"| {rule_id} | {rule_info['name']} | {status} |")
        lines.append("")

        report = "\n".join(lines)

        report_path = os.path.join(V2_ROOT, "state", "architecture_compliance_report.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            _log(f"合规报告已写入 {report_path}", "OK")
        except Exception as e:
            _log(f"写入报告失败: {e}", "WARN")

        return report


class ArchitectureSupervisorAgent:
    """架构合规监督智能体主类 (v2)"""

    def __init__(self, state=None):
        self.state = state
        self.agent_validator = AgentOutputValidator(state=state)
        self.flow_validator = OrchestratorFlowValidator(state=state)
        self.code_validator = CodeStructureValidator()
        self.reporter = ComplianceReporter()

    def check_code_structure(self) -> list[Violation]:
        _log("检查代码结构合规性...", "AUDIT")
        violations = self.code_validator.validate_all()
        for v in violations:
            icon = "🚨" if v.severity == Violation.CRITICAL else "⚠️"
            _log(f"{icon} [{v.rule_id}] {v.message}", "CRIT" if v.severity == Violation.CRITICAL else "WARN")
            _write_compliance_record(v.to_dict())
        if not violations:
            _log("代码结构合规检查全部通过", "OK")
        return violations

    def check_decisions(self) -> list[Violation]:
        _log("检查智能体决策合规性...", "AUDIT")
        violations = self.agent_validator.validate_all_decisions_from_db()
        for v in violations:
            icon = "🚨" if v.severity == Violation.CRITICAL else "⚠️"
            _log(f"{icon} [{v.rule_id}] {v.message}", "CRIT" if v.severity == Violation.CRITICAL else "WARN")
            _write_compliance_record(v.to_dict())
        if not violations:
            _log("智能体决策合规检查全部通过", "OK")
        return violations

    def check_flow(self) -> list[Violation]:
        _log("检查 Orchestrator 流程合规性...", "AUDIT")
        violations = self.flow_validator.validate_all()
        for v in violations:
            icon = "🚨" if v.severity == Violation.CRITICAL else "⚠️"
            _log(f"{icon} [{v.rule_id}] {v.message}", "CRIT" if v.severity == Violation.CRITICAL else "WARN")
            _write_compliance_record(v.to_dict())
        if not violations:
            _log("Orchestrator 流程合规检查全部通过", "OK")
        return violations

    def check_all(self) -> str:
        _log("=" * 60)
        _log("  架构合规监督智能体 (v2) — 全面检查")
        _log("=" * 60)

        all_violations = []

        code_violations = self.check_code_structure()
        all_violations.extend(code_violations)

        decision_violations = self.check_decisions()
        all_violations.extend(decision_violations)

        flow_violations = self.check_flow()
        all_violations.extend(flow_violations)

        self.reporter.add_violations(all_violations)
        report = self.reporter.generate_report()

        critical_count = sum(1 for v in all_violations if v.severity == Violation.CRITICAL)
        warning_count = sum(1 for v in all_violations if v.severity == Violation.WARNING)

        _log("")
        _log("=" * 60)
        if critical_count == 0 and warning_count == 0:
            _log("  ✅ 全面合规 — 所有架构规则检查通过", "OK")
        elif critical_count == 0:
            _log(f"  ⚠️ 基本合规 — {warning_count} 个警告", "WARN")
        else:
            _log(f"  🚨 不合规 — {critical_count} 个严重违规, {warning_count} 个警告", "CRIT")
        _log("=" * 60)

        return report

    def validate_agent_output(self, output: dict, agent_name: str) -> list[Violation]:
        violations = self.agent_validator.validate_decision(output, agent_name)
        for v in violations:
            icon = "🚨" if v.severity == Violation.CRITICAL else "⚠️"
            _log(f"{icon} [{v.rule_id}] {v.message}", "CRIT" if v.severity == Violation.CRITICAL else "WARN")
            _write_compliance_record(v.to_dict())
        return violations

    def watch(self, interval: int = 60):
        _log("=" * 60)
        _log("  架构合规监督智能体 (v2) — 持续监控模式")
        _log(f"  检查间隔: {interval}秒")
        _log("=" * 60)

        try:
            while True:
                self.reporter = ComplianceReporter()
                self.check_all()
                _log(f"下次检查: {interval}秒后", "INFO")
                time.sleep(interval)
        except KeyboardInterrupt:
            _log("收到中断信号，停止监控", "INFO")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="架构合规监督智能体 (v2)")
    parser.add_argument("--check", type=str, default="all",
                        choices=["code", "decisions", "flow", "all"],
                        help="检查类型")
    parser.add_argument("--watch", action="store_true",
                        help="持续监控模式")
    parser.add_argument("--interval", type=int, default=60,
                        help="监控间隔（秒）")
    args = parser.parse_args()

    state = None
    try:
        from autoresearch_v2.db import StateDB
        state = StateDB()
    except Exception:
        _log("无法连接状态数据库，跳过决策检查", "WARN")

    supervisor = ArchitectureSupervisorAgent(state=state)

    if args.watch:
        supervisor.watch(interval=args.interval)
    elif args.check == "code":
        violations = supervisor.check_code_structure()
        supervisor.reporter.add_violations(violations)
        report = supervisor.reporter.generate_report()
        print("\n" + report)
    elif args.check == "decisions":
        violations = supervisor.check_decisions()
        supervisor.reporter.add_violations(violations)
        report = supervisor.reporter.generate_report()
        print("\n" + report)
    elif args.check == "flow":
        violations = supervisor.check_flow()
        supervisor.reporter.add_violations(violations)
        report = supervisor.reporter.generate_report()
        print("\n" + report)
    else:
        report = supervisor.check_all()
        print("\n" + report)

    if state:
        state.close()


if __name__ == "__main__":
    main()
