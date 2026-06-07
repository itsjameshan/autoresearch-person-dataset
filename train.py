import os
import sys

from ultralytics import YOLO
from evaluate import evaluate_model, print_metrics

# ══════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG
# ══════════════════════════════════════════════════════════════

MODEL = "yolo12n.pt"
DATA_YAML = "D:\\PythonProject\\person_dataset\\person.yaml"
IMGSZ = 640
EPOCHS = 2

BATCH = 15
import torch

# Auto-detect compatible device (handles RTX 50-series sm_120 compatibility)
def _get_compatible_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training requires a GPU. "
            "Please install a CUDA-capable PyTorch version."
        )
    cc = torch.cuda.get_device_capability()
    major, minor = cc
    compute_capability = major * 10 + minor  # e.g. (8,6) -> 86
    # Check PyTorch CUDA version to determine max supported CC
    pt_cuda_ver = float(torch.version.cuda) if torch.version.cuda else 0.0
    max_supported_cc = 90 if pt_cuda_ver < 13.0 else 120
    if compute_capability > max_supported_cc:
        raise RuntimeError(
            f"GPU CC {major}.{minor} (sm_{compute_capability}) exceeds PyTorch max supported CC {max_supported_cc}. "
            f"Training stopped. To use GPU, install PyTorch with CUDA 13.x or newer."
        )
    return 0

DEVICE = _get_compatible_device()
AMP = False
CACHE = "disk"
WORKERS = 4
SINGLE_CLS = True
COS_LR = True
PATIENCE = 30

LR0 = 0.0028265938763074048
LRF = 0.37261932455399976
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 10.0
TRANSLATE = 0.3
SCALE = 0.5
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 0.6786607357228835
MIXUP = 0.1
COPY_PASTE = 0.1
ERASING = 0.4
CLOSE_MOSAIC = 15

BOX = 13.484196360221048
CLS = 1.0

CONF = 0.001
IOU = 0.5

PROJECT = "autoresearch_runs"
NAME = "exp_mosaic_boost_v6"
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
        imgsz = 320,
        batch = 6,
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
LLM_BACKEND = "ollama"
