# 架构合规监督报告 (v2)

生成时间: 2026-04-30 17:31:15

## 总体评估

**状态: ⚠️ 基本合规** — 12 个警告

| 级别 | 数量 |
|------|------|
| 🚨 CRITICAL | 0 |
| ⚠️ WARNING | 12 |
| 📋 INFO | 0 |
| **总计** | **12** |

## 按类别统计

| 类别 | CRITICAL | WARNING | INFO |
|------|----------|---------|------|
| agent_structure | 0 | 3 | 0 |
| deterministic_tools | 0 | 1 | 0 |
| orchestrator | 0 | 5 | 0 |
| state_layer | 0 | 3 | 0 |

## 违规详情

### ⚠️ [ARCH-007] 终止条件(v2)

- **严重级别**: WARNING
- **类别**: orchestrator
- **违规描述**: Orchestrator 缺少终止条件: pipeline-tile背离
- **详情**: v2 架构要求终止条件全部实现
- **修复建议**: 在 orchestrator.py 中实现 pipeline-tile背离 终止条件

### ⚠️ [ARCH-007] 终止条件(v2)

- **严重级别**: WARNING
- **类别**: orchestrator
- **违规描述**: Orchestrator 缺少终止条件: LLM失败率halt
- **详情**: v2 架构要求终止条件全部实现
- **修复建议**: 在 orchestrator.py 中实现 LLM失败率halt 终止条件

### ⚠️ [ARCH-024] 双阶段评估

- **严重级别**: WARNING
- **类别**: orchestrator
- **违规描述**: 未检测到 pipeline-mode 评估
- **详情**: v2: 每轮迭代必须跑 tile-mode + pipeline-mode 两套评估
- **修复建议**: 在 eval_dispatcher 或 orchestrator 中添加 pipeline 评估步骤

### ⚠️ [ARCH-025] 双写legacy文件

- **严重级别**: WARNING
- **类别**: state_layer
- **违规描述**: db.py 未检测到对 results.tsv 的双写
- **详情**: v2: SQLite 为主存储，同时写 legacy 文件维持兼容
- **修复建议**: 在 db.py 中添加对 results.tsv 的双写逻辑

### ⚠️ [ARCH-025] 双写legacy文件

- **严重级别**: WARNING
- **类别**: state_layer
- **违规描述**: db.py 未检测到对 last_metrics.json 的双写
- **详情**: v2: SQLite 为主存储，同时写 legacy 文件维持兼容
- **修复建议**: 在 db.py 中添加对 last_metrics.json 的双写逻辑

### ⚠️ [ARCH-025] 双写legacy文件

- **严重级别**: WARNING
- **类别**: state_layer
- **违规描述**: db.py 未检测到对 status.md 的双写
- **详情**: v2: SQLite 为主存储，同时写 legacy 文件维持兼容
- **修复建议**: 在 db.py 中添加对 status.md 的双写逻辑

### ⚠️ [ARCH-026] 纯Ollama后端

- **严重级别**: WARNING
- **类别**: agent_structure
- **违规描述**: agents/researcher.py 导入了 anthropic SDK
- **详情**: v2: LLM 全部本地 Ollama，不应依赖 anthropic
- **修复建议**: 移除 anthropic 导入，改用 Ollama HTTP 客户端

### ⚠️ [ARCH-026] 纯Ollama后端

- **严重级别**: WARNING
- **类别**: agent_structure
- **违规描述**: agents/curator.py 导入了 anthropic SDK
- **详情**: v2: LLM 全部本地 Ollama，不应依赖 anthropic
- **修复建议**: 移除 anthropic 导入，改用 Ollama HTTP 客户端

### ⚠️ [ARCH-026] 纯Ollama后端

- **严重级别**: WARNING
- **类别**: agent_structure
- **违规描述**: agents/triage.py 导入了 anthropic SDK
- **详情**: v2: LLM 全部本地 Ollama，不应依赖 anthropic
- **修复建议**: 移除 anthropic 导入，改用 Ollama HTTP 客户端

### ⚠️ [ARCH-029] pipeline-tile背离检测

- **严重级别**: WARNING
- **类别**: orchestrator
- **违规描述**: 未检测到 pipeline-tile 背离检测逻辑
- **详情**: v2: pipeline_metrics 与 tile_metrics 严重背离(>10%差距3轮)时触发 HITL
- **修复建议**: 添加 pipeline_metrics vs tile_metrics 背离检测

### ⚠️ [ARCH-030] LLM失败率halt

- **严重级别**: WARNING
- **类别**: orchestrator
- **违规描述**: 未检测到 LLM 失败率 halt 逻辑
- **详情**: v2: LLM 调用失败率 > 30% 时必须 halt
- **修复建议**: 添加 LLM 调用失败率统计和 halt 机制

### ⚠️ [ARCH-031] evaluate.py FP/FN导出

- **严重级别**: WARNING
- **类别**: deterministic_tools
- **违规描述**: evaluate.py 未支持 --export-fpfn 参数
- **详情**: v2: evaluate.py 必须支持 FP/FN 导出给 Curator 使用
- **修复建议**: 在 evaluate.py 中添加 --export-fpfn DIR 参数

## 架构规则检查清单 (v2)

| 规则ID | 规则名称 | 状态 |
|--------|----------|------|
| ARCH-001 | 三智能体架构 | ✅ PASS |
| ARCH-002 | 智能体模型分配(v2) | ✅ PASS |
| ARCH-003 | 合约优先 | ✅ PASS |
| ARCH-004 | 禁止NAS | ✅ PASS |
| ARCH-005 | 预算双重守门 | ✅ PASS |
| ARCH-006 | HITL门触发 | ✅ PASS |
| ARCH-007 | 终止条件(v2) | ⚠️ WARNING |
| ARCH-008 | HPO用Optuna串行TPE | ✅ PASS |
| ARCH-009 | Triage置信度门 | ✅ PASS |
| ARCH-010 | Curator图片限制 | ✅ PASS |
| ARCH-011 | 状态层6张表(v2) | ✅ PASS |
| ARCH-012 | Orchestrator单线程 | ✅ PASS |
| ARCH-013 | 智能体不直接调train.py | ✅ PASS |
| ARCH-014 | 指标回归检测用规则 | ✅ PASS |
| ARCH-015 | Curator输出落data_issues | ✅ PASS |
| ARCH-016 | 文件结构合规(v2) | ✅ PASS |
| ARCH-017 | 数据质量校验必跑 | ✅ PASS |
| ARCH-018 | 决策审计日志不可变 | ✅ PASS |
| ARCH-019 | 平台期检测 | ✅ PASS |
| ARCH-020 | 连续失败上限 | ✅ PASS |
| ARCH-021 | Researcher config_patch 白名单 | ✅ PASS |
| ARCH-022 | MODEL路径强约束 | ✅ PASS |
| ARCH-023 | BATCH上限约束 | ✅ PASS |
| ARCH-024 | 双阶段评估 | ⚠️ WARNING |
| ARCH-025 | 双写legacy文件 | ⚠️ WARNING |
| ARCH-026 | 纯Ollama后端 | ⚠️ WARNING |
| ARCH-027 | Curator风险预案 | ✅ PASS |
| ARCH-028 | Optuna串行约束 | ✅ PASS |
| ARCH-029 | pipeline-tile背离检测 | ⚠️ WARNING |
| ARCH-030 | LLM失败率halt | ⚠️ WARNING |
| ARCH-031 | evaluate.py FP/FN导出 | ⚠️ WARNING |
| ARCH-032 | ollama_runner并存 | ✅ PASS |
