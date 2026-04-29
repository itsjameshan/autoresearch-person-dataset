# 自动迭代优化实验设计
# 目标: mAP50>0.95, precision>0.90, recall>0.85

---

## 阶段 0: 基线测试 (Baseline)
**目标:** 建立当前性能基准
**配置:**
- model: yolo12s
- epochs: 100
- imgsz: 1280
- lr0: 0.01, lrf: 0.01
- mosaic: 1.0, mixup: 0.15, copy_paste: 0.15

**预期指标:** mAP50≈0.65

---

## 阶段 1: 模型升级 (Model Scale)
**目标:** 通过更大模型提升容量
**实验 A: yolo12m**
- model: yolo12m
- 其他参数同基线

**实验 B: yolo12l (如果GPU内存足够)**
- model: yolo12l
- batch: 8 (减半)

---

## 阶段 2: 数据增强优化 (Augmentation Tuning)
**目标:** 提升密集人群泛化能力
**实验 A: Copy-Paste增强**
- copy_paste: 0.3 (加倍)
- mosaic: 1.0, mixup: 0.2

**实验 B: 旋转增强**
- degrees: 15.0
- translate: 0.4

**实验 C: 颜色抖动增强**
- hsv_h: 0.02, hsv_s: 0.8, hsv_v: 0.5

---

## 阶段 3: 小目标专门优化 (Small Object Optimization)
**目标:** 提升small_obj_recall
**实验 A: 多尺度训练**
- imgsz: 640,1280 (多尺度)
- 或训练两个模型合并预测

**实验 B: 提升anchor匹配**
- cls: 0.8 (增加分类权重)
- box: 7.5 (降低定位权重)

**实验 C: 降低conf阈值**
- conf: 0.001 (降低置信度)
- iou: 0.5 (标准NMS)

---

## 阶段 4: 超参数搜索 (Hyperparameter Search)
**学习率扫描:**
- lr0: [0.005, 0.01, 0.02]
- lrf: [0.005, 0.01, 0.02]

**损失权重扫描:**
- box: [5.0, 7.5, 10.0, 12.5]
- cls: [0.3, 0.5, 0.7, 1.0]

---

## 阶段 5: 集成优化 (Ensemble)
**如果单一模型无法达标:**
- 训练多个不同配置的模型
- 集成预测结果

---

## 实验记录模板
```
实验名称:
模型:
epochs:
imgsz:
batch:
关键参数:
mAP50:
precision:
recall:
small_obj_recall:
counting_mae:
备注:
```

---

## 当前性能
| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| mAP50 | 0.6479 | >0.95 |
| precision | 0.6712 | >0.90 |
| recall | 0.6857 | >0.85 |
| small_obj_recall | 0.602 | >0.90 |
| counting_mae | 2.89 | <1.0 |