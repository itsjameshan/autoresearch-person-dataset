# 🚀 混合自主研究循环 - 完整使用指南

## 核心思想

**结合两种方法的优点：**

| 方法 | 优点 | 缺点 |
|------|------|------|
| 贝叶斯优化 | 快速超参搜索，数学严谨 | 黑箱，缺乏可解释性，无法做结构创新 |
| Karpathy LLM | 有假设，可解释，可创新 | 可能"乱试"，需要方向引导 |

---

## 🎯 混合系统设计

```
┌─────────────────────────────────────────────────────────────┐
│                  HYBRID AUTORESEARCH LOOP                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 1: 📊 BAYESIAN WARMUP (12 次实验)                       │
│     ├─ 快速搜索超参空间                                      │
│     ├─ 建立初始性能基线                                      │
│     └─ 更新贝叶斯先验                                        │
│                                                              │
│  PHASE 2: 🧠 LLM DEEP RESEARCH (剩余实验)                      │
│     ├─ 基于贝叶斯结果做有假设的研究                            │
│     ├─ 贝叶斯提供参数范围提示                                 │
│     └─ 探索数据增强、损失权重、训练策略等                      │
│                                                              │
│  两个阶段的贝叶斯模型会持续更新，互相反馈！                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始（Windows GPU）

### 前置条件

1. Ollama 已安装
2. Python 环境已配置
3. Git 已配置

### 启动步骤

#### 方法1: 一键启动（推荐）

```batch
双击运行: start_hybrid.bat
```

#### 方法2: 手动启动

```batch
# 终端1: Ollama
$env:OLLAMA_GPU_LAYERS = 0
ollama serve

# 终端2: 混合循环
cd D:\PythonProject\person_dataset
python hybrid_loop.py
```

---

## 📊 三种运行模式对比

你现在有三种选择！

| 模式 | 文件 | 说明 | 推荐场景 |
|------|------|------|----------|
| 📊 **贝叶斯优化** | `ollama_runner.py` | 纯贝叶斯，快速超参搜索 | 只需要调参时 |
| 🧠 **Karpathy LLM** | `autoresearch_loop.py` | 纯 LLM，有创意 | 需要深度研究时 |
| 🚀 **混合模式** | `hybrid_loop.py` | ✅ **推荐**，两者结合 | 最佳效果 |

---

## ⚙️ 混合系统配置

在 `hybrid_loop.py` 中调整：

```python
# 阶段划分
MAX_EXPERIMENTS     = 30        # 总实验次数
BAYESIAN_PHASE_EXPS = 12        # Phase1: 贝叶斯阶段次数

# 停止条件
MAX_NO_IMPROVE      = 8         # 连续无提升停止
TARGET_CDS          = 0.85      # 目标 CDS

# LLM 配置
OLLAMA_MODEL        = "gemma3:4b"
```

---

## 📁 系统文件

| 文件 | 用途 |
|------|------|
| `hybrid_loop.py` | 混合循环主程序 ✨ |
| `start_hybrid.bat` | Windows 一键启动 |
| `autoresearch_loop.py` | 纯 LLM 循环（备选） |
| `ollama_runner.py` | 纯贝叶斯（备选） |

---

## 📊 实时监控

```batch
# 状态概览（含当前阶段）
type status.md

# 完整历史（含 phase 列）
type results.tsv

# 训练日志
type run.log
```

---

## 🎯 results.tsv 格式说明

新增 `phase` 列：

| exp | phase | description | ... |
|-----|-------|-------------|-----|
| 1 | BAYESIAN | Bayesian opt: {...} | ... |
| 12 | BAYESIAN | Bayesian opt: {...} | ... |
| 13 | LLM | 基于贝叶斯结果，尝试... | ... |

---

## 🔧 自定义阶段划分

如果你想调整阶段划分：

```python
# 修改这两个参数
BAYESIAN_PHASE_EXPS = 15   # 更多贝叶斯
# 或者
BAYESIAN_PHASE_EXPS = 8    # 更快进入 LLM
```

---

## 💡 为什么混合更好？

1. **更快收敛** - 贝叶斯快速找到好的起点
2. **避免乱试** - LLM 有贝叶斯提示做方向引导
3. **可解释 + 高效** - 两者优点结合
4. **持续学习** - LLM 阶段也会更新贝叶斯模型

---

## 🎉 研究完成后

- 最佳模型: `best_model/best.pt`
- 完整历史: `results.tsv`
- 最终状态: `status.md`

---

## 🔄 三种模式，按需选择！

- 急着调参？→ `ollama_runner.py`
- 想做深度研究？→ `autoresearch_loop.py`
- 想要最佳效果？→ `hybrid_loop.py` ✨
