import os
import sys

import torch
from ultralytics import YOLO

original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

DEFAULT_MODEL = "person_count/weights/best.pt"
model_path = os.environ.get("EXPORT_MODEL_PATH", DEFAULT_MODEL)
if not os.path.isfile(model_path):
    print(
        f"[ERROR] 找不到模型文件: {model_path}\n"
        f"        请通过 EXPORT_MODEL_PATH 环境变量指定 .pt 路径，"
        f"或将模型放至默认位置 ({DEFAULT_MODEL})。",
        file=sys.stderr,
    )
    sys.exit(1)

model = YOLO(model_path)
model.export(
    format="onnx",
    imgsz=1280,
    simplify=True,
    device=0
)

print("✅ ONNX 导出完成！")
