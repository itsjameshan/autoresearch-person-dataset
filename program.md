# autoresearch — Stadium Dense Crowd Detection

## Platform

- **GPU**: NVIDIA RTX 5070 12GB VRAM (CUDA)
- **RAM**: 64GB DDR4-3200
- **CPU**: Intel i5-14600KF (14 cores, 20 threads)
- **Agent**: Ollama + qwen2.5-coder:14b (local, no cloud)

## Quick Start

```powershell
# 1. Start Ollama (CPU inference — keep GPU 100% for YOLO)
$env:OLLAMA_GPU_LAYERS = 0; ollama serve

# 2. (another terminal) Verify model
ollama list  # should show qwen2.5-coder:14b

# 3. Preflight
python ollama_runner.py --preflight

# 4. Run
python ollama_runner.py
# Ctrl+C to stop. Results persist in results.tsv and status.md.
```

## How It Works

**Fixed 5-minute time budget per experiment.** ~12 experiments/hour, ~100 overnight.

```
LOOP FOREVER:
  1. LLM proposes ONE change to train.py (~30s)
  2. git commit
  3. Train for 5 minutes (time= parameter, not epochs)
  4. Evaluate (CDS metric)
  5. If CDS improved: KEEP (branch advances)
     If not: DISCARD (git reset --hard HEAD~1)
  6. Log to results.tsv
```

## The Goal

**Maximize CDS (Crowd Detection Score):**

```
CDS = 0.25×mAP50 + 0.25×mAP50-95 + 0.10×F1 + 0.15×counting_acc + 0.15×small_obj_recall + 0.10×latency_score
```

Higher is better. Quality gates: precision >= 0.90, recall >= 0.85.

## What You CAN Modify

**Only `train.py`** — the EXPERIMENT CONFIG section:
- Model: person_dataset/yolov8s.pt, yolov8n.pt, yolo12n.pt, yolo12s.pt
- IMGSZ, BATCH, LR0, LRF, COS_LR
- Augmentation: mosaic, mixup, copy_paste, erasing, etc.
- Loss weights: BOX, CLS
- TIME_MINUTES (but keep <=10)

Model paths MUST use `person_dataset/` prefix (e.g. `person_dataset/yolov8s.pt`).
Bare filenames trigger Ultralytics download.

## What You CANNOT Modify

- `evaluate.py` — read-only CDS computation
- `ollama_runner.py` — the loop driver
- Deployment thresholds, CDS weights, quality gates

## VRAM Budget (12GB hard limit)

| Model | imgsz | batch=8 | batch=16 | Safe? |
|-------|-------|---------|----------|-------|
| yolov8n | 640 | ~2GB | ~4GB | YES |
| yolov8s | 640 | ~3GB | ~6GB | YES |
| yolov8s | 1280 | ~8GB | overflow | batch<=8 |
| yolo12s | 640 | ~5GB | ~10GB | batch<=8 |
| yolo12s | 1280 | ~25GB | overflow | BANNED |
| yolo12l | any | >10GB | overflow | BANNED |

copy_paste>0.2 or mixup>0.2 adds ~50% VRAM. Auto-guard will reject unsafe configs.

## Exploration Priorities

One hypothesis per experiment. Order:

1. **Model**: yolov8s+640 → yolov8s+1280(batch8) → yolo12s+640(batch8)
2. **LR**: LR0, LRF, cos_lr
3. **Augmentation**: mosaic, copy_paste, erasing, mixup (one at a time)
4. **Loss weights**: BOX, CLS
5. **IMGSZ + batch**: trade resolution vs batch size within VRAM budget

## Simplicity

- Equal CDS → prefer simpler config
- Don't add complexity for +0.001 CDS
- Every DESCRIPTION must explain what and why

## Git Workflow

- Branch: `autoresearch/crowd-win`
- Keep → commit stays, push to remote
- Discard → `git reset --hard HEAD~1`
- Crash → revert train.py, log the crash
- Never force push, never modify evaluate.py

## Monitoring

```powershell
type status.md                                    # dashboard
type results.tsv                                  # full history
powershell -c "Get-Content run.log -Tail 20"      # live training
```
