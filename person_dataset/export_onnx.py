from ultralytics import YOLO
# 加载官方yolo12l
model = YOLO("yolo12s.pt")
# 导出为ONNX（1280分辨率，和你代码一致）
model.export(format="onnx", imgsz=1280, simplify=True, opset=17)