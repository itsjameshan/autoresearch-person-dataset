# Autoresearch Status

## Session Info
- Started: 2026-04-06 16:52
- Experiment branch: autoresearch/crowd-v1
- Total experiments: 0 (baseline training epoch 1/10 IN PROGRESS)
- Python: x86_64 under Rosetta (functional but ~4-8x slower than native ARM)

## BASELINE TRAINING IN PROGRESS
- Model: yolov8s.pt
- imgsz: 320 (CPU: 4x faster than 640; MPS fills disk via swap on 8GB M1)
- batch: 8
- device: cpu (MPS causes swap-driven disk fill — hit 87% swap, disk drained at 1GB/5min)
- PID: 12449
- Speed: ~3.4 s/it (stable)
- Est. time/epoch: ~42 min
- Est. total: ~7h (10 epochs)
- Progress: epoch 1/10

## Hardware Lessons Learned
- MPS at imgsz=640 batch=4: swap hits 87% (8.9GB/10.2GB), disk fills at ~1GB/5min → KILLS disk
- MPS at imgsz=1280 batch=4: Metal shader deadlock (UN state 25min)
- MPS at imgsz=1280 batch=8: OOM (7.4GB / 8GB allocated)
- CPU at imgsz=640 batch=8: STABLE but 15s/it → 45h for 15 epochs (too slow)
- CPU at imgsz=320 batch=8: STABLE, 3.4s/it → 42min/epoch → 7h baseline ← CURRENT

## Current Best
- CDS: (pending baseline)

## Quality Gate Progress
- Precision: ?/0.90 target
- Recall: ?/0.85 target
- TARGET_MET streak: 0/3 needed to stop

## Stop Condition Status
- Consecutive discards: 0/5
- Session experiments: 0/20
- Hardware: stable (48% disk, 5GB swap)

## Last 3 Experiments
(none completed yet)

## What's Working
- CPU imgsz=320 batch=8: stable, no memory pressure, ~3.4s/it
- Dataset: 5980 train / 1714 val images loaded cleanly

## Next Experiments (after baseline)
1. yolo12s model with same config (suggestions.md priority 1)
2. copy_paste 0.1→0.3 (suggestions.md priority 2)
3. Resume MPS only if disk has 30GB+ free and swap < 3GB
