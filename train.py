"""
train.py — THE ONE FILE the agent modifies.

Each experiment: change hyperparams/model/augmentation here, then run:
  python train.py > run.log 2>&1

After training, this script calls evaluate_model() and prints structured
metrics for the autoresearch loop to parse via grep.

DO NOT modify: evaluate.py (defines CDS including latency), deployment thresholds, quality gates.
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

# Model — 必须使用仓库内路径（相对 train.py 所在目录），勿写裸文件名如 yolo12s.pt（否则会联网下载）
MODEL = "person_dataset/yolo12s.pt"

# Dataset
DATA_YAML = "person_dataset/person.yaml"
IMGSZ = 1280

# Training — RTX 5070 12GB VRAM, 64GB RAM
EPOCHS = 100
BATCH = 8
PATIENCE = 30
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
MIXUP = 0.3
COPY_PASTE = 0.5
ERASING = 0.6
CLOSE_MOSAIC = 20

# Loss weights
BOX = 7.5
CLS = 0.5

# Other
AMP = True
# Windows + ollama_runner 子进程里 ram cache + 多 workers 易 MemoryError；稳定后可改回 "ram" / 提高 workers
CACHE = False
WORKERS = 2
SINGLE_CLS = True

# ══════════════════════════════════════════════════════════════
# TRAINING — do not modify below this line
# ══════════════════════════════════════════════════════════════

PROJECT = "autoresearch_runs"
NAME = "current"

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _abs_under_repo(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(_REPO_ROOT, rel_or_abs))


def train():
    model_path = _abs_under_repo(MODEL)
    if not os.path.isfile(model_path):
        print(
            f"[FATAL] 找不到权重: {model_path}\n"
            "  请将 .pt 放在 person_dataset/ 下，且 MODEL 写成 person_dataset/xxx.pt；"
            "勿使用裸文件名，否则会触发 Ultralytics 从 GitHub 下载。"
        )
        sys.exit(1)

    data_yaml = _abs_under_repo(DATA_YAML)
    if not os.path.isfile(data_yaml):
        print(f"[FATAL] 找不到数据配置: {data_yaml}")
        sys.exit(1)

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

    # Find best model — use actual save_dir from results
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

    # ── Optional: mark experiment 'started' in v2 state.db (P1) ──
    import time as _time
    _state_db_path = os.environ.get("AUTORESEARCH_STATE_DB", "").strip()
    _train_started_at = _time.time()
    _train_sha = None
    if _state_db_path:
        try:
            from person_dataset.autoresearch_v2 import db as _state_db
            _train_sha = _state_db.get_current_git_sha()
            if _train_sha:
                _state_db.init_db(_state_db_path)
                _state_db.upsert_experiment(
                    _state_db_path, _train_sha,
                    status="started",
                    started_at=_train_started_at,
                )
        except Exception as _e:
            print(f"train: state.db 'started' write failed: {_e}", file=sys.stderr)

    best_pt, epochs_completed = train()

    if _state_db_path and _train_sha:
        try:
            _state_db.upsert_experiment(
                _state_db_path, _train_sha,
                status="trained",
                gpu_minutes=round((_time.time() - _train_started_at) / 60.0, 2),
                epochs_completed=epochs_completed,
            )
        except Exception as _e:
            print(f"train: state.db 'trained' write failed: {_e}", file=sys.stderr)

    print()
    print(f"=== Evaluation (model: {best_pt}) ===")
    data_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_YAML)
    metrics = evaluate_model(best_pt, data_yaml_path, IMGSZ)
    print_metrics(metrics, epochs_completed)

    # ── P2: Auto-run pipeline (large-image) eval if pipeline_val/ available ──
    from evaluate import evaluate_pipeline, resolve_pipeline_dir
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _pipeline_dir = resolve_pipeline_dir(repo_root=_repo_root)
    pipeline_metrics = None
    if _pipeline_dir is not None:
        print()
        print(f"=== Pipeline Eval (large images: {_pipeline_dir}) ===")
        try:
            pipeline_metrics = evaluate_pipeline(best_pt, str(_pipeline_dir), IMGSZ)
        except Exception as _e:
            print(f"train: pipeline eval failed: {_e}", file=sys.stderr)
    else:
        print("train: pipeline eval skipped (set AUTORESEARCH_PIPELINE_DIR or "
              "place pipeline_val/ at repo root; set AUTORESEARCH_DISABLE_PIPELINE=1 to silence)")

    # ── P2: persist pipeline_metrics into state.db ──
    if _state_db_path and _train_sha and pipeline_metrics:
        try:
            _state_db.record_metrics(
                _state_db_path, _train_sha,
                pipeline_metrics=pipeline_metrics,
            )
        except Exception as _e:
            print(f"train: state.db pipeline_metrics write failed: {_e}", file=sys.stderr)
