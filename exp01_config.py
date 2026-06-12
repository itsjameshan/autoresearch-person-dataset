# 实验1: yolo12m 基线
# 目标: 测试更大模型的性能提升
# 预期: mAP50 提升 ~10-15%
MODEL = "yolo12m.pt"
DATA_YAML = "person.yaml"
IMGSZ = 1280
EPOCHS = 100
BATCH = 16
DEVICE = 0
CACHE = "ram"
WORKERS = 8

# 学习率
LR0 = 0.01
LRF = 0.01
COS_LR = True
PATIENCE = 30

# 数据增强
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 10.0
TRANSLATE = 0.3
SCALE = 0.5
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 1.0
MIXUP = 0.15
COPY_PASTE = 0.15
ERASING = 0.4
CLOSE_MOSAIC = 15

# 损失权重
BOX = 10.0
CLS = 0.5

# 评估阈值
CONF = 0.001
IOU = 0.5

PROJECT = "autoresearch_runs"
NAME = "exp01_yolo12m_baseline"