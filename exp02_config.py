# 实验2: Copy-Paste 增强 (密集人群优化)
# 目标: 提升小目标检测和密集人群表现
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

# 数据增强: 提升Copy-Paste和混合
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 10.0
TRANSLATE = 0.3
SCALE = 0.5
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 1.0
MIXUP = 0.2
COPY_PASTE = 0.3
ERASING = 0.4
CLOSE_MOSAIC = 15

# 损失权重
BOX = 10.0
CLS = 0.5

# 评估阈值
CONF = 0.001
IOU = 0.5

PROJECT = "autoresearch_runs"
NAME = "exp02_copypaste_heavy"