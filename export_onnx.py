import torch
from ultralytics import YOLO

original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

model = YOLO("person_count/weights/best.pt")
model.export(
    format="onnx",
    imgsz=1280,
    simplify=True,
    device=0
)

print("✅ ONNX 导出完成！")