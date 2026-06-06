"""
通用训练脚本 - 支持从配置文件读取
使用方式: python train_with_config.py <config_file.py>
"""

import os
import sys
import importlib.util
import torch

_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from ultralytics import YOLO
from evaluate import evaluate_model, print_metrics

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Auto-detect compatible device (handles RTX 50-series sm_120 compatibility)
def _get_compatible_device(preferred_device):
    if preferred_device == "cpu":
        return "cpu"
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
    return preferred_device

def train(config):
    _repo_root = os.path.dirname(os.path.abspath(__file__))

    def _abs(rel_or_abs):
        if os.path.isabs(rel_or_abs):
            return os.path.normpath(rel_or_abs)
        return os.path.normpath(os.path.join(_repo_root, rel_or_abs))

    model_path = _abs(config.MODEL)
    data_yaml = _abs(config.DATA_YAML)
    model = YOLO(model_path)

    device = _get_compatible_device(getattr(config, "DEVICE", 0))

    results = model.train(
        data=data_yaml,
        epochs=config.EPOCHS,
        imgsz=config.IMGSZ,
        batch=config.BATCH,
        device=device,
        patience=config.PATIENCE,
        project=config.PROJECT,
        name=config.NAME,
        exist_ok=True,
        single_cls=getattr(config, "SINGLE_CLS", True),
        conf=config.CONF,
        iou=config.IOU,
        lr0=config.LR0,
        lrf=config.LRF,
        cos_lr=config.COS_LR,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        degrees=config.DEGREES,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        flipud=config.FLIPUD,
        fliplr=config.FLIPLR,
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        copy_paste=config.COPY_PASTE,
        erasing=config.ERASING,
        close_mosaic=config.CLOSE_MOSAIC,
        box=config.BOX,
        cls=config.CLS,
        amp=getattr(config, "AMP", True),
        cache=getattr(config, "CACHE", "ram"),
        workers=getattr(config, "WORKERS", 8),
        val=True,
        plots=True,
        save=True,
        verbose=True,
    )
    
    try:
        save_dir = str(results.save_dir)
    except Exception:
        save_dir = os.path.join(config.PROJECT, config.NAME)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(save_dir, "weights", "last.pt")
    
    try:
        epochs_completed = results.epoch
    except Exception:
        epochs_completed = 0
    
    return best_pt, epochs_completed, save_dir

def main():
    if len(sys.argv) < 2:
        print("使用方式: python train_with_config.py <config_file.py>")
        print("例如: python train_with_config.py exp01_config.py")
        sys.exit(1)
    
    config_path = sys.argv[1]
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    print(f"实验名称: {config.NAME}")
    print(f"模型: {config.MODEL}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Image Size: {config.IMGSZ}")
    print("-" * 50)
    
    best_pt, epochs_completed, save_dir = train(config)
    
    print("-" * 50)
    print("训练完成!")
    print(f"最佳模型: {best_pt}")
    
    data_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.DATA_YAML)
    metrics = evaluate_model(best_pt, data_yaml_path, config.IMGSZ)
    print_metrics(metrics, epochs_completed)
    
    try:
        torch.cuda.empty_cache()
    except:
        pass

if __name__ == "__main__":
    main()