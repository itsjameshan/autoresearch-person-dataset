import os
from ultralytics import YOLO
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# 配置（适合密集人群）
# --------------------------
MODEL = "yolo12l.pt"  # 大模型，适合密集人群
CONF = 0.25           # 置信度阈值（低一点，能检测更多人）
IOU = 0.20            # IoU阈值（非常低，避免重叠人群合并）
IMAGE_PATH = os.path.join(_BASE, "images", "train", "DSC01523_0_0.JPG")  # 按需改文件名

# 加载模型
model = YOLO(MODEL)

# 推理（只检测人）
results = model(
    IMAGE_PATH,
    conf=CONF,
    iou=IOU,
    classes=[0]  # 只保留人
)

# 取结果
res = results[0]
boxes = res.boxes

if boxes is not None:
    # 1. 总人数
    total = len(boxes)

    # 2. 置信度统计
    confs = boxes.conf.cpu().numpy()
    avg_conf = np.mean(confs)
    min_conf = np.min(confs)
    max_conf = np.max(confs)

    # 3. IoU重叠统计
    iou_list = []
    xyxy = boxes.xyxy.cpu().numpy()
    for i in range(len(xyxy)):
        for j in range(i + 1, len(xyxy)):
            x1 = max(xyxy[i][0], xyxy[j][0])
            y1 = max(xyxy[i][1], xyxy[j][1])
            x2 = min(xyxy[i][2], xyxy[j][2])
            y2 = min(xyxy[i][3], xyxy[j][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1])
            area2 = (xyxy[j][2] - xyxy[j][0]) * (xyxy[j][3] - xyxy[j][1])
            iou_val = inter / (area1 + area2 - inter + 1e-6)
            if iou_val > 0:
                iou_list.append(iou_val)
    avg_iou = np.mean(iou_list) if iou_list else 0

    # 输出统计
    print("=" * 50)
    print(f"照片：{IMAGE_PATH}")
    print(f"总人数：{total}")
    print(f"置信度：平均 {avg_conf:.2f} | 最低 {min_conf:.2f} | 最高 {max_conf:.2f}")
    print(f"人群平均重叠 IoU：{avg_iou:.2f}")
    print("=" * 50)

else:
    print("未检测到人")

# 显示带框照片
res.show()