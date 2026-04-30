import os
import sys
import torch

_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from ultralytics import YOLO
from evaluate import evaluate_model, print_metrics

# ══════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG
# ══════════════════════════════════════════════════════════════

MODEL = "yolo12m.pt"
DATA_YAML = "D:\\PythonProject\\person_dataset\\person.yaml"
IMGSZ = 1280
EPOCHS = 100

BATCH = 16
DEVICE = 0
AMP = True
CACHE = "ram"
WORKERS = 8
SINGLE_CLS = True
COS_LR = True
PATIENCE = 30

LR0 = 0.01
LRF = 0.01

HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 10.0
TRANSLATE = 0.3
SCALE = 0.5
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 0.95  # Increased mosaic
MIXUP = 0.10  # Reduced mixup
COPY_PASTE = 0.15
ERASING = 0.4
CLOSE_MOSAIC = 15

BOX = 10.0
CLS = 0.5

CONF = 0.001
IOU = 0.5

PROJECT = "autoresearch_runs"
NAME = "exp_mosaic_boost_v3"
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def _abs(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(_REPO_ROOT, rel_or_abs))

def train():
    model_path = _abs(MODEL)
    data_yaml = _abs(DATA_YAML)
    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        single_cls=SINGLE_CLS,
        conf=CONF,
        iou=IOU,
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
    best_pt, epochs_completed = train()
    data_yaml_path = _abs(DATA_YAML)
    metrics = evaluate_model(best_pt, data_yaml_path, IMGSZ)
    print_metrics(metrics, epochs_completed)

    try:
        torch.cuda.empty_cache()
    except:
        pass