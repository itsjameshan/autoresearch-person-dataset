import torch
import os
from ultralytics import YOLO

# ==================== 安全修复 ====================
original_load = torch.load


def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)


torch.load = safe_load

# ==================== 配置 ====================
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person.yaml")

model = YOLO("yolo12n.pt")

if __name__ == "__main__":
    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=1280,
        batch=8,
        device=0,
        project="person_count",
        name="best_model",
        exist_ok=True,

        # ==================== 数据增强 ====================
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,

        # ==================== 训练参数 ====================
        cache=False,
        workers=0,
        val=True,
        plots=True,
        patience=15,
        save=True,
        save_period=-1,
        single_cls=True,
        conf=0.001,
        iou=0.6,
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
    )