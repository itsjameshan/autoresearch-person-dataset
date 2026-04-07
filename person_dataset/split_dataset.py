import os
import shutil
from sklearn.model_selection import train_test_split

# ---------------------- 配置路径 ----------------------
# 原始数据路径
img_val_dir = "images/val"    # 原始验证集图片
label_val_dir = "labels/val"  # 原始验证集标签

# 目标保存路径（会自动创建）
train_img_dir = "image/train"
val_img_dir = "image/val"
train_label_dir = "label/train"
val_label_dir = "label/val"

# 划分比例（训练集 : 验证集 = 8 : 2）
test_size = 0.2
random_seed = 42  # 固定随机种子，保证可复现

# ---------------------- 创建文件夹 ----------------------
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# ---------------------- 获取所有图片文件名（不带后缀） ----------------------
img_files = [f for f in os.listdir(img_val_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
img_stems = [os.path.splitext(f)[0] for f in img_files]  # 提取文件名（不含后缀）

# ---------------------- 划分训练集和验证集 ----------------------
train_stems, val_stems = train_test_split(
    img_stems,
    test_size=test_size,
    random_state=random_seed
)

# ---------------------- 复制图片和标签 ----------------------
def copy_files(stems, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    for stem in stems:
        # 复制图片
        for ext in ['.jpg', '.png', '.jpeg']:
            img_src = os.path.join(src_img_dir, f"{stem}{ext}")
            if os.path.exists(img_src):
                shutil.copy(img_src, os.path.join(dst_img_dir, f"{stem}{ext}"))
                break
        # 复制标签
        label_src = os.path.join(src_label_dir, f"{stem}.txt")
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(dst_label_dir, f"{stem}.txt"))
        else:
            print(f"警告：未找到标签文件 {label_src}")

# 执行复制
copy_files(train_stems, img_val_dir, label_val_dir, train_img_dir, train_label_dir)
copy_files(val_stems, img_val_dir, label_val_dir, val_img_dir, val_label_dir)

print("✅ 数据集划分完成！")
print(f"训练集数量：{len(train_stems)}")
print(f"验证集数量：{len(val_stems)}")