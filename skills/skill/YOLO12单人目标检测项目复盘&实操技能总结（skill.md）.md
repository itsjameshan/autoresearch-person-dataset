# YOLO12单人目标检测项目复盘\&amp;实操技能总结（skill\.md）

# 一、项目概述

本次项目核心是完成「单人目标检测」数据集的制作、清洗、校验及YOLO12模型训练全流程，解决数据合规性、训练参数异常等问题，最终实现稳定的50轮模型训练，为后续推理测试奠定基础。

# 二、核心问题与解决方案（重点复盘）

## 2\.1 脏数据检测与数据集合规校验

- 检测范围：图片与标签数量匹配、YOLO标签格式正确性、边界框坐标（\[0,1\]范围）、类别ID合法性、图片文件完整性（无损坏）

- 出现问题：检测提示「存在1个标签无对应图片」，经排查为`labels/train`文件夹中残留无关的`classes\.txt`（配置文件，非标注文件）

- 结论：原始数据集标注规范，脏数据总数为0，可正常用于训练；冗余`classes\.txt`不影响训练，可删除或移出labels文件夹消除警告

## 2\.2 数据集深度清洗（去重/无效框过滤）

使用专属清洗脚本，重点完成3件事，且默认保障数据安全：

- 去除标签中的重复标注框（通过保留4位小数的坐标去重，避免浮点精度误差）

- 过滤宽高≤0\.001的无效极小框（像素级无效标注，无实际意义）

- 清理空标签、损坏标签文件；默认仅删除标签文件，不删除对应图片，避免误丢数据

## 2\.3 训练轮数异常（核心踩坑）

- 问题：手动设置`epoch=50`无效，实际训练仅跑2轮

- 根因：原代码中训练轮数由命令行参数控制（贝叶斯runner传入），全局变量`epoch=50`未被实际调用

- 解决方案：仅在`model\.train\(\)`参数中写死`epochs=50`，不修改任何贝叶斯超参、增强、损失等参数，确保轮数生效

## 2\.4 路径报错问题

- 错误：代码试图读取`labels/train/classes\.txt`，提示文件不存在

- 原因：`classes\.txt`实际存放在数据集根目录，而非labels子文件夹

- 解决方案：将代码中`CLASSES\_FILE`路径修正为数据集根目录的`classes\.txt`绝对路径

# 三、标准化实操流程（下次直接套用）

## 3\.1 数据集准备（标准结构）

- 核心结构：`person\_dataset`根目录下，包含`images/train`（训练图片）、`labels/train`（训练标签）、`classes\.txt`（类别配置）

- 单类别配置：`classes\.txt`中仅保留一行`person`，对应类别ID=0

## 3\.2 数据自检（必做步骤）

1. 运行脏数据检测脚本，确认无坐标越界、格式错误、图片损坏等问题，确保脏数据总数为0

2. 运行标签清洗脚本，去除重复标注、无效框、空标签，完成后再次校验数据合规性

## 3\.3 数据集划分

- 按9:1比例划分训练集（train）和验证集（val），生成`train\.txt`和`val\.txt`索引文件，存放于`dataset\_split`文件夹

- 索引文件中需写入图片的绝对路径，确保模型可正常读取

## 3\.4 训练配置

1. 编写`person\.yaml`配置文件（存放于数据集根目录），明确数据集路径、训练/验证集索引、类别数及类别名称

2. 修改`train\.py`，仅固定`epochs=50`，保留所有贝叶斯超参、增强参数、损失参数不变

## 3\.5 模型训练（固定参数）

- 模型：yolo12s\.pt（预训练权重）

- 关键参数：输入尺寸`IMGSZ=1280`、批次`BATCH=2`、轮数`epochs=50`、设备`DEVICE=0`（GPU）、单类别`SINGLE\_CLS=True`

- 启动命令：`python train\.py \&gt; run\.log 2\&gt;\&amp;1`（将日志输出到run\.log，便于后续排查问题）

# 四、关键配置留存（直接复用）

## 4\.1 person\.yaml 配置模板（单类别）

```yaml
# 数据集配置
path: D:/PythonProject/person_dataset  # 数据集根目录（绝对路径）
train: dataset_split/train.txt        # 训练集索引文件路径
val: dataset_split/val.txt           # 验证集索引文件路径

# 类别配置
nc: 1  # 单类别（person）
names: ['person']  # 类别名称
```

## 4\.2 训练核心固定参数（贝叶斯参数不变）

- 基础配置：`MODEL=\&\#34;yolo12s\.pt\&\#34;`、`DATA\_YAML=r\&\#34;D:\\PythonProject\\person\_dataset\\person\.yaml\&\#34;`、`IMGSZ=1280`

- 训练控制：`epochs=50`、`BATCH=2`、`DEVICE=0`、`AMP=True`、`SINGLE\_CLS=True`

- 贝叶斯超参（原样保留）：LR0=0\.0035、LRF=0\.0004969409379021998、MOSAIC=0\.40342537519900845、BOX=19\.445283794975794、CLS=0\.7085783830096928、CONF=0\.002439817175999175、IOU=0\.4836474394931204

# 五、踩坑清单（避坑重点）

1. 不要将`classes\.txt`放入`labels`文件夹，会被识别为标签文件，产生「无对应图片」的冗余警告

2. 训练轮数必须写死在`model\.train\(epochs=50\)`中，仅设置全局变量`epoch=50`无效，会被命令行参数覆盖

3. 高分辨率（1280x1280）训练时，`BATCH`不宜过大（当前设为2），避免GPU显存溢出

4. 数据清洗时，优先只清理标签文件，原图需谨慎删除（建议先注释删除图片的代码，验证无误后再执行）

5. Windows系统中，文件路径需使用原始字符串`r\&\#34;\&\#34;`（如`r\&\#34;D:\\PythonProject\\person\_dataset\&\#34;`），防止转义字符导致路径报错

6. 训练日志建议输出到文件（`\&gt; run\.log 2\&gt;\&amp;1`），便于后续排查训练中断、loss异常等问题

# 六、下一步行动清单

1. 使用已修复的`train\.py`（固定50轮）启动模型训练，观察训练日志，确认轮数正常、无显存报错

2. 训练完成后，提取`autoresearch\_runs/current/weights/best\.pt`（最优权重），若不存在则使用`last\.pt`

3. 进行推理测试，验证单人目标检测效果（调整置信度`conf`，确保检测准确率和召回率）

4. 若训练过程中出现loss异常（如box\_loss居高不下、mAP50不上升），可微调学习率、马赛克增强强度等参数，无需改动核心配置

5. 复盘训练日志，记录最优参数组合，为后续模型优化、数据集扩充提供参考

# 七、核心脚本留存（直接复用）

## 7\.1 脏数据检测脚本

```python
import os
import cv2

def check_dirty_data(images_dir, labels_dir, classes_file):
    # 1. 读取类别文件
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)
    print(f"检测到类别数: {num_classes}，类别列表: {classes}")
    print("-" * 60)

    # 2. 获取所有图片和标签文件
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(img_exts)]
    all_labels = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]

    print(f"图片文件总数: {len(all_images)}")
    print(f"标签文件总数: {len(all_labels)}")
    print("-" * 60)

    # 3. 检查图片与标签是否一一对应
    img_names_no_ext = {os.path.splitext(f)[0] for f in all_images}
    label_names_no_ext = {os.path.splitext(f)[0] for f in all_labels}

    missing_labels = img_names_no_ext - label_names_no_ext
    missing_images = label_names_no_ext - img_names_no_ext

    if missing_labels:
        print(f"⚠️  存在图片无标签文件: {len(missing_labels)} 个")
        print("    示例:", list(missing_labels)[:5])
    if missing_images:
        print(f"⚠️  存在标签无对应图片: {len(missing_images)} 个")
        print("    示例:", list(missing_images)[:5])
    if not missing_labels and not missing_images:
        print("✅ 图片与标签文件一一对应")
    print("-" * 60)

    # 4. 逐个检测标签格式和边界框合法性
    dirty_count = 0
    for img_file in all_images:
        img_name_no_ext = os.path.splitext(img_file)[0]
        label_file = f"{img_name_no_ext}.txt"
        label_path = os.path.join(labels_dir, label_file)
        img_path = os.path.join(images_dir, img_file)

        # 跳过没有标签的情况（前面已经统计过）
        if not os.path.exists(label_path):
            continue

        # 先检查图片是否能正常读取
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ 图片损坏/无法读取: {img_file}")
                dirty_count += 1
                continue
        except Exception as e:
            print(f"❌ 图片读取异常 {img_file}: {e}")
            dirty_count += 1
            continue

        # 读取标签
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line_idx, line in enumerate(lines):
            parts = line.split()
            # 格式必须是: class_id x_center y_center width height
            if len(parts) != 5:
                print(f"❌ 标签格式错误 [{label_file}:{line_idx+1}]: {line}")
                dirty_count += 1
                continue

            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                print(f"❌ 标签数值无法解析 [{label_file}:{line_idx+1}]: {line}")
                dirty_count += 1
                continue

            # 检查类别号是否合法（单类别class_id只能是0）
            if class_id < 0 or class_id >= num_classes:
                print(f"❌ 类别号越界 [{label_file}:{line_idx+1}]: class_id={class_id} (有效范围: 0~{num_classes-1})")
                dirty_count += 1

            # 检查边界框坐标是否在[0,1]范围内
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                print(f"❌ 边界框坐标越界 [{label_file}:{line_idx+1}]: xc={xc:.4f}, yc={yc:.4f}, w={w:.4f}, h={h:.4f}")
                dirty_count += 1

    print("=" * 60)
    print(f"检测完成！脏数据总数: {dirty_count}")
    if dirty_count == 0:
        print("✅ 未检测到脏数据，数据集格式规范！")

if __name__ == "__main__":
    # 配置实际路径
    IMAGES_DIR = r"D:\PythonProject\person_dataset\images\train"
    LABELS_DIR = r"D:\PythonProject\person_dataset\labels\train"
    CLASSES_FILE = r"D:\PythonProject\person_dataset\classes.txt"  # 根目录路径

    check_dirty_data(IMAGES_DIR, LABELS_DIR, CLASSES_FILE)
```

## 7\.2 标签清洗脚本

```python
import os
import numpy as np
from pathlib import Path

# 配置你的数据集路径
DATASET_ROOT = r"D:\PythonProject\person_dataset"
LABELS_DIR = os.path.join(DATASET_ROOT, "labels", "train")  # 训练集标签路径
IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "train")

def clean_dirty_labels():
    label_files = list(Path(LABELS_DIR).glob("*.txt"))
    deleted_count = 0
    corrupted_labels = []  # 记录损坏的标签文件

    for label_file in label_files:
        try:
            # 读取标签（格式：class x_center y_center width height）
            with open(label_file, "r") as f:
                lines = f.readlines()
            
            if not lines:
                # 空标签，标记为待删除
                corrupted_labels.append(str(label_file))
                continue

            # 解析标签并去重（转换为整数坐标后比较，避免浮点精度问题）
            unique_boxes = set()
            valid_lines = []
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 5:
                    corrupted_labels.append(str(label_file))
                    break
                cls, x, y, w, h = parts
                # 转换为整数坐标（保留4位小数），用于去重
                box_key = (cls, round(x,4), round(y,4), round(w,4), round(h,4))
                if box_key in unique_boxes:
                    # 重复标注，跳过
                    continue
                # 检查是否为无效框（宽/高≤0.001，即像素级无效）
                if w <= 0.001 or h <= 0.001:
                    continue
                unique_boxes.add(box_key)
                valid_lines.append(line)
            
            if len(valid_lines) == 0:
                # 无有效标签，标记为删除
                corrupted_labels.append(str(label_file))
            else:
                # 重写清理后的标签（保留去重后的有效框）
                with open(label_file, "w") as f:
                    f.writelines(valid_lines)
                deleted_count += len(lines) - len(valid_lines)  # 统计删除的无效框数量

        except Exception as e:
            corrupted_labels.append(str(label_file))

    # 删除空标签/损坏标签文件
    for label_path in corrupted_labels:
        os.remove(label_path)
        # 同步删除对应的图片（可选，建议先注释掉，先验证再删）
        img_path = os.path.join(IMAGES_DIR, os.path.basename(label_path).replace(".txt", ".jpg"))
        if os.path.exists(img_path):
            # os.remove(img_path)  # 取消注释才会删除图片
            pass

    print(f"清理完成：")
    print(f"1. 删除无效/重复标注共 {deleted_count} 个")
    print(f"2. 删除空标签/损坏标签文件 {len(corrupted_labels)} 个")
    print(f"3. 对应图片可手动删除（建议先验证，再批量删）")

if __name__ == "__main__":
    clean_dirty_labels()
```

## 7\.3 最终50轮训练脚本（train\.py）

```python
"""
train.py — THE ONE FILE the agent modifies.
Each experiment: change hyperparams/model/augmentation here, then run:
  python train.py > run.log 2>&1
"""

import os
import sys
import torch

# ── Fix torch.load for newer PyTorch ──
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from ultralytics import YOLO
from evaluate import evaluate_model, print_metrics

# ══════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG — Agent modifies this section
# ══════════════════════════════════════════════════════════════

MODEL = "yolo12s.pt"
DATA_YAML = r"D:\PythonProject\person_dataset\person.yaml"
IMGSZ = 1280
epoch = 50

# ===================== 【关键】贝叶斯控制的参数 =====================
TIME_MINUTES = 0
BATCH = 2
DEVICE = 0
AMP = True
CACHE = None
WORKERS = 0
SINGLE_CLS = True
COS_LR = True

# 学习率
LR0 = 0.0035
LRF = 0.0004969409379021998

# 增强
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 5.0
TRANSLATE = 0.3
SCALE = 0.7
FLIPUD = 0.5
FLIPLR = 0.5
MOSAIC = 0.40342537519900845
MIXUP = 0.0
COPY_PASTE = 0.00
ERASING = 0.4
CLOSE_MOSAIC = 10

# 损失
BOX = 19.445283794975794
CLS = 0.7085783830096928

# 【贝叶斯可调】置信度与NMS阈值
CONF = 0.002439817175999175
IOU = 0.4836474394931204

# ══════════════════════════════════════════════════════════════
# TRAINING — do not modify below this line
# ══════════════════════════════════════════════════════════════

PROJECT = "autoresearch_runs"
NAME = "current"
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def _abs(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(_REPO_ROOT, rel_or_abs))

def train():
    model_path = _abs(MODEL)
    data_yaml = _abs(DATA_YAML)
    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        epochs=50,  # 仅修改此处，固定50轮
        time=TIME_MINUTES / 60,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=0,
        project=PROJECT,
        name=NAME,
        exist_ok=True,

        single_cls=SINGLE_CLS,
        conf=CONF,       # 贝叶斯参数不变
        iou=IOU,         # 贝叶斯参数不变

        lr0=LR0,
        lrf=LRF,
        cos_lr=COS_LR,

        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASING,
        close_mosaic=CLOSE_MOSAIC,

        box=BOX,
        cls=CLS,

        amp=AMP,
        cache=CACHE,
        workers=WORKERS,
        val=True,
        plots=True,
        save=True,
        verbose=True,
    )

    try:
        save_dir = str(results.save_dir)
    except Exception:
        save_dir = os.path.join(PROJECT, NAME)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(save_dir, "weights", "last.pt")

    try:
        epochs_completed = results.epoch
    except Exception:
        epochs_completed = 0

    return best_pt, epochs_completed

if __name__ == "__main__":
    best_pt, epochs_completed = train()
    data_yaml_path = _abs(DATA_YAML)
    metrics = evaluate_model(best_pt, data_yaml_path, IMGSZ)
    print_metrics(metrics, epochs_completed)

    try:
        torch.cuda.empty_cache()
    except:
        pass
```

> （注：文档部分内容可能由 AI 生成）
