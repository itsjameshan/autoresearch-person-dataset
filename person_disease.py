import torch
import os
from ultralytics import YOLO
import ultralytics

# ==================== 安全加载修复 ====================
original_load = torch.load


def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)


torch.load = safe_load
ultralytics.settings.set('run_dir', None)
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person_dataset", "person.yaml")
EPOCHS = 50
IMGSZ = 1280  # 图片尺寸
BATCH = 8  # 批次
DEVICE = 0  # GPU
PROJECT = "person_count"
NAME = "best_model"
model = YOLO("yolov8s.pt")

if __name__ == "__main__":
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        amp=False,
        cache="ram",
        verbose=True,
        workers=0,
        val=True,
        plots=True,
        patience=15,
        save=True,

        # 单类别专用
        single_cls=True,
        conf=0.001,
        iou=0.6,

        # 学习率
        lr0=0.005,
        lrf=0.1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )
