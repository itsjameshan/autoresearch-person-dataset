# Karpathy 式自主研究循环 - 完整使用指南

## 📦 系统概览

这是一个完整的 **Andrej Karpathy 式 autoresearch 系统**，让 LLM 扮演 PhD 学生，自主运行研究循环。

```
┌─────────────────────────────────────────────────────────────┐
│                     自主研究循环                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 🧠 LLM 提出假设（基于 program.md + 历史结果）             │
│  2. ✏️ LLM 修改 train.py                                      │
│  3. 📝 Git commit（记录假设）                                 │
│  4. 🏋️ 训练模型（限时 2 小时）                                 │
│  5. 🔍 evaluate.py 计算 CDS 指标                               │
│  6. ✅/❌ 决策：有提升 keep，无提升 discard（git reset）         │
│  7. 📊 更新 status.md + results.tsv                          │
│  8. 🛡️ 硬件安全检查 + 冷却等待                                 │
│  9. 🔄 检查停止条件 → 继续或停止                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始（Windows GPU）

### 前置条件

1. **Ollama** 已安装
   - 下载: https://ollama.com/
   
2. **Python 环境** 已配置
   - ultralytics, torch, httpx, psutil 等
   
3. **Git** 已配置

---

### 启动步骤

#### 方法1: 一键启动（推荐）

```batch
双击运行: start_autoresearch.bat
```

#### 方法2: 手动启动

```batch
# 终端1: 启动 Ollama（CPU推理，让GPU专注YOLO）
$env:OLLAMA_GPU_LAYERS = 0
ollama serve

# 终端2: 启动研究循环
cd D:\PythonProject\person_dataset
python autoresearch_loop.py
```

---

## 📊 实时监控

研究循环运行时，你可以随时监控：

```batch
# 状态概览
type status.md

# 完整历史记录
type results.tsv

# 实时训练日志
type run.log

# 查看最佳模型
dir best_model
```

---

## 📁 系统文件说明

| 文件 | 用途 |
|------|------|
| `autoresearch_loop.py` | 主循环程序 |
| `start_autoresearch.bat` | Windows 一键启动脚本 |
| `train.py` | 训练配置（LLM 唯一可修改的文件） |
| `evaluate.py` | 评估脚本（只读，计算 CDS） |
| `program.md` | LLM 研究规则手册 |
| `status.md` | 实时状态仪表盘 |
| `results.tsv` | 完整实验历史记录 |
| `best_model/` | 最佳模型存储目录 |

---

## ⚙️ 核心配置（在 autoresearch_loop.py 中）

```python
# 停止条件
MAX_EXPERIMENTS     = 30        # 最多实验次数
MAX_NO_IMPROVE      = 8         # 连续无提升停止
TARGET_CDS          = 0.85      # 目标 CDS
COOLDOWN_SEC        = 30        # 冷却时间

# LLM 配置
OLLAMA_MODEL        = "gemma3:4b"
OLLAMA_URL          = "http://localhost:11434"

# 硬件安全
MAX_VRAM_USAGE      = 0.90      # VRAM 上限
MAX_RAM_USAGE       = 0.85      # RAM 上限
TRAIN_TIMEOUT       = 7200      # 训练超时（秒）
```

---

## 🎯 指标说明

### CDS（Crowd Detection Score）

```python
CDS = 0.25 * mAP50 
    + 0.25 * mAP50-95 
    + 0.10 * F1 
    + 0.15 * counting_mae 
    + 0.15 * small_obj_recall 
    + 0.10 * latency_score
```

### Quality Gates

- ✅ Precision >= 0.90
- ✅ Recall >= 0.85

---

## 🔧 故障排除

### Ollama 连接失败

```
检查:
1. Ollama 正在运行？
2. 端口 11434 可访问？
3. 模型已拉取？ollama pull gemma3:4b
```

### 训练超时

- 检查 `TRAIN_TIMEOUT` 配置
- 检查 GPU 是否被占用
- 减少 `EPOCHS` 或 `IMGSZ`

### Git 操作失败

- 检查 Git 配置
- 检查仓库权限
- 检查分支是否正确

---

## 📖 研究规则（program.md）

LLM 会阅读 `program.md` 来理解研究规则。主要规则:

1. **只修改 `train.py`** - 其他文件只读
2. **一次一个假设** - 不要同时改多个参数
3. **VRAM 预算** - yolo12s + 1280 + batch8 = 安全
4. **简洁优先** - 同等 CDS 选简单配置
5. **Git 工作流** - keep 就 push，discard 就 reset

---

## 🎉 研究完成后

研究循环停止时，会:

1. 保存最佳模型到 `best_model/best.pt`
2. 完整历史在 `results.tsv`
3. 最终状态在 `status.md`

---

## 🔄 两种运行模式对比

| 模式 | 文件 | 说明 |
|------|------|------|
| **LLM 自主研究** | `autoresearch_loop.py` | ✅ Karpathy 式，可解释，有创意 |
| **贝叶斯优化** | `ollama_runner.py` | 快速超参搜索，但黑箱 |

---

## 💡 提示

- 可以随时按 **Ctrl+C** 停止研究循环
- 停止后可以随时重新启动，历史会保留
- `best_model` 会自动保留最佳模型
- 建议先运行 `ollama_runner.py` 热身，再启动 `autoresearch_loop.py`

---

祝研究顺利！🎯
