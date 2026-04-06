# autoresearch — Stadium Dense Crowd Detection (Windows / RTX 5070)

## Platform

- **GPU**: NVIDIA RTX 5070 12GB VRAM (CUDA)
- **RAM**: 64GB DDR4-3200
- **CPU**: Intel i5-14600KF (14 cores, 20 threads)
- **Storage**: 1TB NVMe SSD
- **Agent**: Ollama + qwen2.5-coder:14b (local, no cloud required)

## Quick Start

```powershell
# 1. Clone the repo
git clone git@github.com:itsjameshan/autoresearch-person-dataset.git
cd autoresearch-person-dataset
git checkout autoresearch/crowd-win

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Ensure dataset is at D:\PythonProject\person_dataset\
#    (images/, labels/ directories with YOLO format annotations)

# 4. Start Ollama
ollama serve
# (in another terminal)
ollama pull qwen2.5-coder:14b

# 5. Run the autonomous experiment loop
python ollama_runner.py
```

## How It Works

`ollama_runner.py` is the autonomous experiment loop driver:

1. Connects to Ollama (localhost:11434)
2. Reads this file, results.tsv, suggestions.md for context
3. Runs baseline (no changes to train.py)
4. Asks qwen2.5-coder:14b to propose train.py modifications
5. Applies changes, git commits, trains, evaluates
6. Keep (push) or discard (reset) based on CDS improvement
7. Repeats until stop condition met

## The Goal

**Maximize CDS (Crowd Detection Score)** — composite metric:

```
CDS = 0.30×mAP50 + 0.30×mAP50-95 + 0.10×F1 + 0.15×counting_acc + 0.15×small_obj_recall
```

Higher is better. Quality gates: precision >= 0.90, recall >= 0.85.

## What Gets Modified

**Only `train.py`** — the EXPERIMENT CONFIG section:
- Model: yolov8s.pt, yolo12n.pt, yolo12s.pt, yolo12l.pt
- Hyperparameters: LR, batch size, epochs, patience
- Data augmentation: mosaic, mixup, copy_paste, erasing, etc.
- Loss weights: box, cls

## What Does NOT Change

- `evaluate.py` — read-only CDS computation
- `ollama_runner.py` — the loop driver
- Deployment thresholds: conf=0.25, iou_nms=0.35
- CDS weights and quality gate thresholds

## Hardware-Optimized Defaults

| Parameter | Value | Reason |
|-----------|-------|--------|
| DEVICE | 0 (CUDA) | RTX 5070 |
| BATCH | 16 | 12GB VRAM handles batch=16 at imgsz=1280 |
| CACHE | "ram" | 64GB RAM — entire dataset fits in memory |
| WORKERS | 8 | 14 cores, 8 workers is sweet spot |
| IMGSZ | 1280 | Full resolution, no compromise needed |
| AMP | True | RTX 5070 has fast FP16 |

## Stop Conditions

| Condition | Action |
|-----------|--------|
| 3 consecutive keeps with [TARGET_MET] | Target achieved → stop, push |
| 5 consecutive discards | Plateau → stop, write analysis |
| 20 experiments | Session cap → stop, summarize |
| Training > 60 minutes | Kill, treat as crash |

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

## Exploration Priorities

1. **Model upgrade**: yolo12s (12GB VRAM can handle it easily)
2. **Bigger batch**: try batch=24 or batch=32 if VRAM allows
3. **Data augmentation**: copy_paste 0.3, mosaic tuning
4. **Loss weights**: box loss, DFL tuning
5. **LR schedule**: warmup, cosine decay
6. **yolo12l**: 12GB VRAM might support it with batch=8

## Pipeline Evaluation

Every 5 keeps, evaluate on large stitched images in `pipeline_val/`:
```
Large image → 1280×1280 tiles (200px overlap) → detect → coordinate restore → global NMS (IoU=0.35)
```
