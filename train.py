import os
import sys

from ultralytics import YOLO
from evaluate import evaluate_model, print_metrics
from train_utils import resolve_model, resolve_data_yaml

# ══════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG
# ══════════════════════════════════════════════════════════════

# 5070 (12GB) 实测预算: yolo12x@batch2 一轮 32 epochs 远超调度器 2h 看门狗
# → 每次都被 SIGKILL 记为 failed。改回 l + 提高 imgsz (small_obj_recall
# 是 CDS 最大的洞), TIME_LIMIT_H 保证在看门狗之前优雅收尾。
MODEL = "yolo12l.pt"
DATA_YAML = "person_dataset/person.yaml"
IMGSZ = 960
EPOCHS = 32
# 训练墙钟上限 (小时)。必须 < train_dispatcher.TRAIN_TIMEOUT(2h) 留出
# 模型加载/缓存/收尾评估的余量 — ultralytics 会在该时限内优雅停止并保存
# best.pt, 实验以"完成+真实指标"落库, 而不是被看门狗杀掉记为 failed。
TIME_LIMIT_H = 1.7

BATCH = 4
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

LR0 = 7.012711284048462e-05
LRF = 0.37261932455399976
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 10.0
TRANSLATE = 0.3
SCALE = 0.5
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 0.9653455441872767
MIXUP = 0.1
COPY_PASTE = 0.1
ERASING = 0.4
CLOSE_MOSAIC = 15

BOX = 19.596278648369893
CLS = 1.0

CONF = 0.001
IOU = 0.5

PROJECT = "autoresearch_runs"
NAME = "exp_mosaic_boost_v6"
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

LLM_BACKEND = "ollama"


def _abs(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(_REPO_ROOT, rel_or_abs))


def _resolve_model(model_name: str) -> str:
    """模型解析 (实现移到 train_utils.resolve_model: 显式容量优先级,
    修掉旧版漏 yolo12x + 按字母序选'最大'的 bug)。"""
    return resolve_model(model_name, _REPO_ROOT)


def _resolved_data_yaml() -> str:
    """把 person.yaml 的相对 `path:` 解析为绝对路径副本 — 相对路径由
    ultralytics 按 datasets_dir 设置解析, 跨机器漂移是历史上反复出现的
    '图片路径'失败根因。"""
    return resolve_data_yaml(_abs(DATA_YAML), _abs(PROJECT))


def train():
    model_path = _resolve_model(MODEL)
    data_yaml = _resolved_data_yaml()

    print(f"[INFO] Training configuration:")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_yaml}")
    print(f"  Image size: {IMGSZ}")
    print(f"  Batch size: {BATCH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Device: {DEVICE}")
    
    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        # 必须用 IMGSZ 变量 — 之前硬编码 640, Researcher/HPO 调 IMGSZ
        # 完全不生效, small_obj_recall 一直 ~0 (CDS 的 15% 直接归零)
        imgsz=IMGSZ,
        # 墙钟上限: 在调度器看门狗 SIGKILL 之前优雅收尾 (覆盖 epochs)
        time=TIME_LIMIT_H,
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
    data_yaml_path = _resolved_data_yaml()
    metrics = evaluate_model(best_pt, data_yaml_path, IMGSZ)
    print_metrics(metrics, epochs_completed)

    try:
        torch.cuda.empty_cache()
    except:
        passcurrent_epochs = 100
suggested_epochs = 100
current_lr0 = 0.01
suggested_lr0 = 0.005
reason = "Increase batch size to reduce memory pressure.  Retain current epochs and LR0. Increase image size to allow for more detail and potentially alleviate pressure.  This action directly addresses the crash reported by the Orchestrator."
batch_size_suggestion = 16
current_epochs = 100
current_imgsz = 640
suggested_imgsz = 1280
current_batch = 16
suggested_batch = 32
