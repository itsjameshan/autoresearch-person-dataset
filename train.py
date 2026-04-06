"""
train.py — THE ONE FILE the agent modifies.

Each experiment: change hyperparams/model/augmentation here, then run:
  python train.py > run.log 2>&1

After training, this script calls evaluate_model() and prints structured
metrics for the autoresearch loop to parse via grep.

DO NOT modify: evaluate.py, deployment thresholds, CDS weights, quality gates.
"""

import os
import sys
import torch
import ultralytics

# ── Fix torch.load for newer PyTorch ──
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from ultralytics import YOLO
from evaluate import evaluate_model, print_metrics

# ══════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG — Agent modifies this section
# ══════════════════════════════════════════════════════════════

# Model
MODEL = "yolov8s.pt"

# Dataset
DATA_YAML = "person.yaml"
IMGSZ = 640  # reduced from 1280 — MPS OOM at 1280 on 8GB M1 (7.4GB used, only 600MB headroom)

# Training
EPOCHS = 100
BATCH = 4
PATIENCE = 30
DEVICE = "mps"  # Apple Silicon GPU

# Learning rate
LR0 = 0.01
LRF = 0.01
COS_LR = True

# Data augmentation
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 5.0
TRANSLATE = 0.1
SCALE = 0.5
FLIPUD = 0.5
FLIPLR = 0.5
MOSAIC = 1.0
MIXUP = 0.1
COPY_PASTE = 0.1
ERASING = 0.4
CLOSE_MOSAIC = 20

# Loss weights
BOX = 7.5
CLS = 0.5

# Other
AMP = True
CACHE = False  # "ram" uses too much on 8GB M1
WORKERS = 0
SINGLE_CLS = True

# ══════════════════════════════════════════════════════════════
# TRAINING — do not modify below this line
# ══════════════════════════════════════════════════════════════

PROJECT = "autoresearch_runs"
NAME = "current"


def train():
    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,

        single_cls=SINGLE_CLS,
        conf=0.001,
        iou=0.6,

        lr0=LR0,
        lrf=LRF,
        cos_lr=COS_LR,

        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASING,
        close_mosaic=CLOSE_MOSAIC,

        box=BOX,
        cls=CLS,

        amp=AMP,
        cache=CACHE,
        workers=WORKERS,
        val=True,
        plots=True,
        save=True,
        verbose=True,
    )

    # Find best model — use actual save_dir from results (handles runs/detect/ prefix)
    try:
        save_dir = str(results.save_dir)
    except Exception:
        save_dir = os.path.join(PROJECT, NAME)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(save_dir, "weights", "last.pt")

    # Get epochs completed from results
    try:
        epochs_completed = results.epoch
    except Exception:
        epochs_completed = EPOCHS

    return best_pt, epochs_completed


if __name__ == "__main__":
    print(f"=== Autoresearch Experiment ===")
    print(f"Model: {MODEL}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH}, ImgSz: {IMGSZ}")
    print(f"LR: {LR0} → {LRF}, CosLR: {COS_LR}")
    print(f"Augmentation: mosaic={MOSAIC} mixup={MIXUP} copy_paste={COPY_PASTE}")
    print(f"Loss: box={BOX} cls={CLS}")
    print()

    best_pt, epochs_completed = train()

    print()
    print(f"=== Evaluation (model: {best_pt}) ===")
    data_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_YAML)
    metrics = evaluate_model(best_pt, data_yaml_path, IMGSZ)
    print_metrics(metrics, epochs_completed)
