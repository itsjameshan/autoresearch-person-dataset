# 体育场密集人群识别系统 — Autoresearch 自主实验框架

基于 [Andrej Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch) 理念，构建的 **YOLOv12 密集人群检测自主优化系统**。AI Agent 全自主运行实验循环：修改超参 → 训练 → 评估 → 保留或丢弃 → 继续，无需人工干预。

## 项目背景

体育场密集人群识别是一个高难度的目标检测场景：
- **密集遮挡**：人群互相遮挡严重，IoU 阈值需要极低（0.20）才能避免框合并
- **小目标多**：远处观众仅占图像面积 < 0.5%，极易漏检
- **大图拼接**：原始图像 5120×3840，需裁切为 1280×1280 tiles → 检测 → 坐标还原 → 全局 NMS 去重
- **硬性指标**：精确率 ≥ 90%、召回率 ≥ 85%（来自项目文档《全流程优化解决方案》）

**当前瓶颈**：mAP50-95 仅 ~0.25，精确率 ~0.71，距离目标差距大。

## 核心方法：Autoresearch 自主实验循环

借鉴 Karpathy 的理念："让 AI 自己做研究"。

```
┌──────────────────────────────────────────────────┐
│                自主实验循环                         │
│                                                    │
│  1. AI 提出假设（修改 train.py 超参）                │
│  2. git commit                                     │
│  3. 训练模型 → run.log                              │
│  4. 评估 CDS 指标                                   │
│  5. CDS 提升 > 0.005 → 保留 (git push)              │
│     CDS 没提升 → 丢弃 (git reset)                   │
│  6. 更新 status.md 进度面板                          │
│  7. 检查停止条件                                     │
│  8. 冷却 → 继续下一轮                                │
└──────────────────────────────────────────────────┘
```

**关键设计**：
- **一个文件**：Agent 只修改 `train.py`，评估脚本 `evaluate.py` 只读不可改
- **一个指标**：CDS（Crowd Detection Score），综合 5 个维度的加权分数
- **Git 即状态机**：每次实验 commit，好的 push，差的 reset，完整可追溯
- **三层安全控制**：硬件保护 + 智能停止 + 自主运行

## CDS 指标（Crowd Detection Score）

单一标量指标，用于 keep/discard 决策：

```
CDS = 0.30 × mAP50 + 0.30 × mAP50-95 + 0.10 × F1 + 0.15 × counting_acc + 0.15 × small_obj_recall
```

| 组件 | 权重 | 衡量内容 |
|------|------|----------|
| mAP50 | 30% | 标准检测指标 |
| mAP50-95 | 30% | 定位精度（当前最弱项） |
| F1_optimal | 10% | 精确率-召回率平衡 |
| counting_accuracy | 15% | 每张图数对了多少人 |
| small_object_recall | 15% | 小目标（远处观众）召回率 |

**Quality Gate**（硬性目标）：precision ≥ 0.90 且 recall ≥ 0.85 → 标记 `[TARGET_MET]`

## 三层安全控制

### 第一层：硬件保护
- 每轮训练前检查内存/温度
- 每轮之间强制冷却等待
- 内存紧张时自动降低 batch size

### 第二层：智能停止
| 条件 | 动作 |
|------|------|
| 连续 3 次 keep 达标 `[TARGET_MET]` | 目标达成 → 停止，合并回 main |
| 连续 5 次 discard | 收益递减 → 停止，写分析报告 |
| 累计 20 轮实验 | 会话上限 → 停止，输出总结 |
| 单次训练超 60 分钟 | 超时 → kill，视为 crash |

### 第三层：自主运行
> Agent 全自主运行，不问用户、不等确认。用户可能在睡觉。遇到停止条件自动停下，push 结果，输出总结。

## 进度监控

Agent 每轮实验后更新 `status.md`，用户随时可查看：

```bash
# Mac
cat status.md            # 看总览
tail -3 results.tsv      # 看最近实验
tail -20 run.log         # 看训练进度

# Windows (PowerShell)
type status.md
powershell -c "Get-Content run.log -Tail 20"
```

## 双平台并行

本项目在两台机器上同时运行实验，互不干扰：

### MacBook Air M1（8GB）— 分支 `autoresearch/crowd-v1`

| 配置 | 值 |
|------|---|
| GPU | MPS → CPU fallback（8GB 内存不够 MPS 训练） |
| batch | 8, imgsz=320 (CPU) |
| Agent | Claude Code (Opus 4.6) |
| 每轮耗时 | ~40 分钟 |
| 冷却时间 | 60 秒（无风扇） |

**遇到的硬件问题**：
- MPS imgsz=1280 batch=8 → OOM
- MPS imgsz=640 batch=4 → swap 占满磁盘（1GB/5min 速度吞噬空间）
- 最终方案：CPU imgsz=320 batch=8，稳定但慢

### Windows 实验室电脑（RTX 5070）— 分支 `autoresearch/crowd-win`

| 配置 | 值 |
|------|---|
| GPU | RTX 5070 12GB CUDA |
| RAM | 64GB DDR4 |
| CPU | i5-14600KF 14 核 |
| batch | 16, imgsz=1280 |
| cache | "ram"（64GB 可以全部缓存） |
| Agent | Ollama + qwen2.5-coder:14b（本地，无需云端） |
| 每轮耗时 | ~5-10 分钟 |
| 冷却时间 | 30 秒（主动散热） |

## 仓库结构

```
autoresearch-person-dataset/
├── train.py              # Agent 修改的唯一文件（训练配置 + 超参）
├── evaluate.py           # 只读评估脚本（CDS 计算、quality gate）
├── program.md            # Agent 指令（自主循环规则）
├── ollama_runner.py      # [Windows] Ollama 驱动的自主循环脚本
├── setup_windows.bat     # [Windows] 一键环境配置
├── requirements.txt      # [Windows] Python 依赖
├── data_quality.py       # 标注质量检查工具（独立于循环）
├── build_pipeline_val.py # 大图拼接工具（从 tiles 重建原始大图）
├── person.yaml           # 数据集配置
├── results.tsv           # 实验日志（每次 keep/discard 记录）
├── status.md             # 实时进度面板（Agent 每轮更新）
├── suggestions.md        # 顾问建议文件（Codex/Ollama 写入）
├── pipeline_val/         # [Windows 分支] 8 张拼接大图 + GT 标注
└── .gitignore            # 排除 images/labels/weights（2.3GB+）
```

## 分支策略

```
main                          ← 稳定版本
├── autoresearch/crowd-v1     ← Mac 实验分支（Claude Code）
└── autoresearch/crowd-win    ← Windows 实验分支（Ollama）
```

- `main`：只在重大里程碑时合并
- 实验分支：各自独立推进，通过 `results.tsv` 追踪进度
- 当 quality gate 全部达标，合并最优方案回 main

## 快速开始

### Mac（Claude Code）

```bash
cd person_dataset/
git checkout autoresearch/crowd-v1
claude "read program.md and suggestions.md (if exists). Run baseline benchmark first, record to results.tsv, then begin the experiment loop. Do not stop or ask me anything. Keep running experiments autonomously until a stop condition is met."
```

### Windows（Ollama）

```powershell
git clone git@github.com:itsjameshan/autoresearch-person-dataset.git
cd autoresearch-person-dataset
git checkout autoresearch/crowd-win
setup_windows.bat
# 另一个终端: ollama serve && ollama pull qwen2.5-coder:14b
python ollama_runner.py
```

## 数据集

- **来源**：体育场俯拍密集人群照片
- **规模**：~2991 张训练图 + ~859 张验证图（1280×1280 tiles）
- **标注**：YOLO 格式，单类别（person）
- **原始大图**：5120×3840（3×4 网格裁切而来）
- **Pipeline 验证集**：8 张拼接大图，共 1560 个 GT 标注框

> 注意：images/ 和 labels/ 目录（~1.8GB）不在 git 中。Windows 机器需要数据集在 `D:\PythonProject\person_dataset\`。

## 灵感来源

- [Andrej Karpathy - autoresearch](https://github.com/karpathy/autoresearch) — 原始框架（LLM 自主 NLP 实验）
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) — Windows/RTX 适配参考
- 《体育场密集人群识别系统——全流程优化解决方案》 — 项目技术文档，定义了质量门槛和部署流程

## 技术栈

| 组件 | 版本/工具 |
|------|----------|
| 检测模型 | YOLOv8s / YOLOv12n/s/l (Ultralytics) |
| 训练框架 | Ultralytics YOLO |
| Mac Agent | Claude Code (Opus 4.6) |
| Windows Agent | Ollama + qwen2.5-coder:14b |
| 版本管理 | Git（实验状态机） |
| 评估指标 | CDS（自定义复合指标） |
