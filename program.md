# autoresearch — Stadium Dense Crowd Detection (Windows / RTX 5070)

## Platform

- **GPU**: NVIDIA RTX 5070 12GB VRAM (CUDA)
- **RAM**: 64GB DDR4-3200
- **CPU**: Intel i5-14600KF (14 cores, 20 threads)
- **Storage**: 1TB NVMe SSD
- **Agent**: Ollama（本地）；默认 **qwen2.5-coder:14b**（代码生成专用，推荐）

## Quick Start

```powershell
# 1. Clone the repo
git clone git@github.com:itsjameshan/autoresearch-person-dataset.git
cd autoresearch-person-dataset
git checkout autoresearch/crowd-win

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. 唯一数据根目录：仓库内 person_dataset/（勿再维护单独的盘符路径副本）。
#    person_dataset/images/{train,val,test}/ 与 person_dataset/labels/...
#    预训练权重示例: person_dataset/yolo12s.pt、yolo12n.pt、yolo12l.pt、yolov8n.pt
#    大文件已被 .gitignore 排除，勿 commit/push。

# 4. Start Ollama (CPU mode — keep GPU free for YOLO training)
$env:OLLAMA_GPU_LAYERS = 0
ollama serve
# (in another terminal)
ollama pull qwen2.5-coder:14b

# 5. Preflight check (verify everything works, ~2 min)
python ollama_runner.py --preflight

# 6. Run the autonomous experiment loop
python ollama_runner.py
```

## How It Works

`ollama_runner.py` is the autonomous experiment loop driver:

1. Connects to Ollama (localhost:11434)
2. Reads this file, results.tsv, suggestions.md for context
3. Runs baseline (no changes to train.py)
4. Asks qwen2.5-coder:14b (via Ollama CPU inference) to propose train.py modifications
5. Applies changes, git commits, trains, evaluates
6. Keep (push) or discard (reset) based on CDS improvement
7. Repeats until stop condition met

## The Goal

**Maximize CDS (Crowd Detection Score)** — composite metric (see `evaluate.py` for exact constants):

```
CDS = 0.25×mAP50 + 0.25×mAP50-95 + 0.10×F1 + 0.15×counting_acc + 0.15×small_obj_recall + 0.10×latency_score
```

Higher is better. Quality gates: precision ≥ 0.90, recall ≥ 0.85, and mean tile `inference_ms` ≤ latency cap (default / env `AUTORESEARCH_LATENCY_GATE_MS` in `evaluate.py`).

## 预训练权重（本地，禁止触发下载）

- **目录（相对仓库根 `autoresearch_new/`）**：`person_dataset/*.pt`
- **`train.py` 里 `MODEL` 必须** 写成带前缀的路径，例如 `person_dataset/yolo12s.pt`。  
  **禁止** 写成裸文件名（如 `yolo12s.pt`）：Ultralytics 会在**当前工作目录**找不到时从 GitHub **重新下载**。
- **训练前会检查文件是否存在**；不存在则立即报错退出，避免默默下载。
- 允许切换的本地文件（按你机器实际存在的为准）：`person_dataset/yolov8n.pt`、`person_dataset/yolo12n.pt`、`person_dataset/yolo12s.pt`、`person_dataset/yolo12l.pt` 等。

## What Gets Modified

**Only `train.py`** — the EXPERIMENT CONFIG section:
- **Model (`MODEL`)**：仅允许 `person_dataset/<文件名>.pt` 形式（见上一节）
- Hyperparameters: LR, batch size, epochs, patience
- Data augmentation: mosaic, mixup, copy_paste, erasing, etc.
- Loss weights: box, cls

## What Does NOT Change

- `evaluate.py` — read-only CDS computation
- `ollama_runner.py` — the loop driver
- Deployment thresholds: conf=0.25, iou_nms=0.35
- CDS weights, quality gate thresholds, and latency mapping (`evaluate.py`)

## Hardware-Optimized Defaults

| Parameter | Value | Reason |
|-----------|-------|--------|
| DEVICE | 0 (CUDA) | RTX 5070 |
| BATCH | 16 | yolov8s+640 在 12GB 内安全 |
| CACHE | "ram" | 64GB RAM 足够缓存数据集 |
| WORKERS | 8 | i5-14600KF 14 核 |
| IMGSZ | 640 | 安全起步；稳定后可升 1280（见 VRAM 预算表） |
| AMP | True | RTX 5070 has fast FP16 |

## VRAM 预算表（RTX 5070 12GB 硬限制）

| 模型 | imgsz | batch=8 VRAM | batch=16 VRAM | 安全？ |
|------|-------|-------------|--------------|--------|
| yolov8n | 640 | ~2GB | ~4GB | ✓ |
| yolov8s | 640 | ~3GB | ~6GB | ✓ |
| **yolov8s** | **1280** | **~8GB** | ~16GB(溢出) | batch≤8 |
| yolo12n | 640 | ~2.5GB | ~5GB | ✓ |
| yolo12s | 640 | ~5GB | ~10GB | batch≤8 |
| yolo12s | 1280 | **~25GB** | 溢出 | ✗ 禁用 |
| yolo12l | any | >10GB | 溢出 | ✗ 禁用 |

**注意**：copy_paste>0.2 或 mixup>0.2 会使 VRAM 增加约 50%。
`ollama_runner.py` 内置 VRAM 安全检查，超限配置会被自动降级。

## Stop Conditions

| Condition | Action |
|-----------|--------|
| 3 consecutive keeps with [TARGET_MET] | Target achieved → stop, push |
| 5 consecutive discards | Plateau → stop, write analysis |
| 20 experiments | Session cap → stop, summarize |
| Training > 4 hours | Kill, treat as crash |

## Monitoring Progress

While the loop runs, open another terminal:

```powershell
type status.md              # Dashboard: best CDS, gate progress
type results.tsv            # Full experiment history
powershell -c "Get-Content run.log -Tail 20"  # Current training progress
```

## Git Workflow

- Branch: `autoresearch/crowd-win`
- Each experiment: modify train.py → commit → train → evaluate
- Keep → push to remote
- Discard → git reset --hard HEAD~1
- Never force push, never modify evaluate.py

## 探索顺序（建议按序尝试）

Ollama 每轮只改 **一个** 明确假设；优先从上到下（先大方向、再细调），避免同一轮堆多项大改。

1. **模型体量**：先用 `person_dataset/yolov8s.pt`+640 建立 baseline → 升 1280(batch≤8) → 试 `yolo12s.pt`+640(batch≤8)。**禁止** yolo12s+1280 或 yolo12l（VRAM 溢出）。
2. **Batch 与显存**：参考 VRAM 预算表。yolov8s+640 可用 batch=16；yolov8s+1280 限 batch≤8；yolo12s+640 限 batch≤8。
3. **学习率与 schedule**：`LR0` / `LRF` / `cos_lr` 小幅调整；一次只改一类。
4. **增强（单项）**：`mosaic`、`mixup`、`copy_paste`、`erasing` 等 **每次只动一两个参数**，便于归因。
5. **Loss**：`box` / `cls` 权重微调。
6. **Epochs / patience**：在单次训练时长可接受前提下再拉长；避免仅靠「训更久」刷 CDS。

（英文备忘：Model → batch → LR → aug → loss → epochs，one hypothesis per experiment.）

## 简洁性原则

- **同等 CDS 下更简单更好**：更少增强开关、更少矛盾组合；能删复杂项不掉点则倾向删。
- **拒绝为 +0.005 CDS 引入难维护配置**（例如极端增强、仅适配单张卡的魔法数）。
- **可解释**：每轮 `DESCRIPTION` 应能说清「改什么、为什么」。

## 依赖边界

- **禁止**在实验中引入 `requirements.txt` 未列出的新 pip 包；需要新库须由人改依赖并提交，不由循环擅自假设。
- **只改 `train.py` 的 EXPERIMENT CONFIG 区**；不修改 `evaluate.py`、`ollama_runner.py` 或未列文件来「绕开」指标。

## Pipeline Evaluation

Every 5 keeps, evaluate on large stitched images in `pipeline_val/`:
```
Large image → 1280×1280 tiles (200px overlap) → detect → coordinate restore → global NMS (IoU=0.35)
```
