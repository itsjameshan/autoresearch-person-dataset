model = YOLO("yolov8s.pt")  # 已经用了s，保持不变（比n更好）

model.train(
    data=DATA_YAML,
    epochs=200,          # 放大epoch上限，让模型充分训练
    imgsz=1280,
    batch=8,
    device=0,
    patience=30,         # 加大耐心，当前15太小了

    single_cls=True,
    conf=0.001,
    iou=0.6,

    # 学习率 — 当前lr0=0.005偏低
    lr0=0.01,            # 恢复默认值
    lrf=0.01,            # 最终学习率衰减到1%（当前0.1太高）
    cos_lr=True,         # 使用余弦退火，比线性衰减更稳定

    # 数据增强 — 当前全关了，这是个大问题！
    hsv_h=0.015,         # 色调增强
    hsv_s=0.7,           # 饱和度增强
    hsv_v=0.4,           # 亮度增强（应对光照变化）
    degrees=5.0,         # 轻微旋转
    translate=0.1,
    scale=0.5,
    flipud=0.5,          # 垂直翻转（俯拍场景有用）
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,           # 轻微mixup，有助于抗模糊
    copy_paste=0.1,      # 复制粘贴增强，增加小目标样本
    erasing=0.4,

    # 损失权重
    box=7.5,
    cls=0.5,             # 单类别cls影响不大

    cache="ram",
    workers=0,
    amp=True,            # 5070支持amp，当前设为False浪费了
    close_mosaic=20,     # 最后20个epoch关闭mosaic精调
)
