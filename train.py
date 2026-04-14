"""
train.py — THE ONE FILE the agent modifies.

Each experiment: change hyperparams/model/augmentation here, then run:
  python train.py > run.log 2>&1

Training runs for a FIXED TIME BUDGET of 5 minutes (wall clock).
After training, this script calls evaluate_model() and prints structured
metrics for the autoresearch loop to parse via grep.

DO NOT modify: evaluate.py, deployment thresholds, CDS weights, quality gates.
"""

import os
import sys
import torch

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

# Model — use repo-relative path with person_dataset/ prefix
# Available: person_dataset/yolov8n.pt, yolov8s.pt, yolo12n.pt, yolo12s.pt
# VRAM budget (12GB limit): yolov8s+640→~3GB | yolov8s+1280→~8GB | yolo12s+640→~5GB
MODEL = "person_dataset/yolo12s.pt"

# Dataset
DATA_YAML = "person_dataset/person.yaml"
IMGSZ = 640

# Training — FIXED 5-MINUTE TIME BUDGET (do not increase beyond 10)
TIME_MINUTES = 5    # Ultralytics time= parameter (in hours internally)
BATCH = 16
DEVICE = 0          # CUDA GPU 0

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
CLOSE_MOSAIC = 10

# Loss weights
BOX = 7.5
CLS = 0.5

# Other
AMP = True
CACHE = "disk"      # Prefer deterministic and lower RAM pressure on Windows
WORKERS = 2
SINGLE_CLS = True

# ══════════════════════════════════════════════════════════════
# TRAINING — do not modify below this line
# ══════════════════════════════════════════════════════════════

PROJECT = "autoresearch_runs"
NAME = "current"

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _abs(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(_REPO_ROOT, rel_or_abs))


def train():
    model_path = _abs(MODEL)
    if not os.path.isfile(model_path):
        print(f"[FATAL] model not found: {model_path}")
        sys.exit(1)

    data_yaml = _abs(DATA_YAML)
    if not os.path.isfile(data_yaml):
        print(f"[FATAL] data yaml not found: {data_yaml}")
        sys.exit(1)

    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        epochs=300,             # high ceiling — time= will stop training
        time=TIME_MINUTES / 60, # convert minutes to hours for Ultralytics
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=300,           # disable early stopping — let time budget decide
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

    # Find best model
    try:
        save_dir = str(results.save_dir)
    except Exception:
        save_dir = os.path.join(PROJECT, NAME)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(save_dir, "weights", "last.pt")

    try:
        epochs_completed = results.epoch
    except Exception:
        epochs_completed = 0

    return best_pt, epochs_completed


if __name__ == "__main__":
    print(f"=== Autoresearch Experiment ===")
    print(f"Model: {MODEL}")
    print(f"Time budget: {TIME_MINUTES} min, Batch: {BATCH}, ImgSz: {IMGSZ}")
    print(f"LR: {LR0} → {LRF}, CosLR: {COS_LR}")
    print(f"Augmentation: mosaic={MOSAIC} mixup={MIXUP} copy_paste={COPY_PASTE}")
    print(f"Loss: box={BOX} cls={CLS}")
    print()

    best_pt, epochs_completed = train()

    print()
    print(f"=== Evaluation (model: {best_pt}) ===")
    data_yaml_path = _abs(DATA_YAML)
    metrics = evaluate_model(best_pt, data_yaml_path, IMGSZ)
    print_metrics(metrics, epochs_completed)
