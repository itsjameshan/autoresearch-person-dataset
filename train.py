"""
train.py — THE ONE FILE the agent modifies.
Each experiment: change hyperparams/model/augmentation here, then run:
  python train.py > run.log 2>&1
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

MODEL = "yolo12s.pt"
DATA_YAML = r"D:\PythonProject\person_dataset\person.yaml"
IMGSZ = 1280
epoch = 50

# ===================== 【关键】贝叶斯控制的参数 =====================
TIME_MINUTES = 0
BATCH = 3
DEVICE = 0
AMP = True
CACHE = None
WORKERS = 0
SINGLE_CLS = True
COS_LR = True

# 学习率
LR0 = 0.0035
LRF = 0.0004969409379021998

# 增强
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 5.0
TRANSLATE = 0.3
SCALE = 0.7
FLIPUD = 0.5
FLIPLR = 0.5
MOSAIC = 0.40342537519900845
MIXUP = 0.0
COPY_PASTE = 0.00
ERASING = 0.4
CLOSE_MOSAIC = 10

# 损失
BOX = 19.445283794975794
CLS = 0.7085783830096928

# 【贝叶斯可调】置信度与NMS阈值
CONF = 0.002439817175999175
IOU = 0.4836474394931204

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
    data_yaml = _abs(DATA_YAML)
    model = YOLO(model_path)

    # 只在这里加了 epochs=50，其他所有参数完全不变！
    results = model.train(
        data=data_yaml,
        epochs=50,
        time=TIME_MINUTES / 60,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=0,
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