
"""
Create proper train/val/test split from labeled data.
Split strategy: 70% train / 20% val / 10% test = 3184 / 910 / 455 images
"""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

# Paths
base_dir = Path(r"D:\PythonProject\person_dataset")
train_img_dir = base_dir / "images" / "train"
val_img_dir = base_dir / "images" / "val"
train_lbl_dir = base_dir / "labels" / "train"
val_lbl_dir = base_dir / "labels" / "val"

# Output paths
new_test_img_dir = base_dir / "images" / "test"
new_test_lbl_dir = base_dir / "labels" / "test"
new_train_img_dir = base_dir / "images" / "train_new"
new_train_lbl_dir = base_dir / "labels" / "train_new"
new_val_img_dir = base_dir / "images" / "val_new"
new_val_lbl_dir = base_dir / "labels" / "val_new"

# Create output directories
for dir_path in [new_test_img_dir, new_test_lbl_dir, new_train_img_dir, 
                 new_train_lbl_dir, new_val_img_dir, new_val_lbl_dir]:
    dir_path.mkdir(exist_ok=True)

# Get all labeled images from train+val
all_images = []

# From train
for img_path in train_img_dir.glob("*.JPG"):
    stem = img_path.stem
    lbl_path = train_lbl_dir / f"{stem}.txt"
    if lbl_path.exists():
        all_images.append((img_path, lbl_path))

for img_path in train_img_dir.glob("*.jpg"):
    stem = img_path.stem
    lbl_path = train_lbl_dir / f"{stem}.txt"
    if lbl_path.exists():
        all_images.append((img_path, lbl_path))

# From val
for img_path in val_img_dir.glob("*.JPG"):
    stem = img_path.stem
    lbl_path = val_lbl_dir / f"{stem}.txt"
    if lbl_path.exists():
        all_images.append((img_path, lbl_path))

for img_path in val_img_dir.glob("*.jpg"):
    stem = img_path.stem
    lbl_path = val_lbl_dir / f"{stem}.txt"
    if lbl_path.exists():
        all_images.append((img_path, lbl_path))

print(f"Total labeled images: {len(all_images)}")

# Shuffle and split
random.shuffle(all_images)

total = len(all_images)
train_count = int(total * 0.7)
val_count = int(total * 0.2)
test_count = total - train_count - val_count

print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

train_set = all_images[:train_count]
val_set = all_images[train_count:train_count+val_count]
test_set = all_images[train_count+val_count:]

# Copy files
def copy_files(file_list, dst_img_dir, dst_lbl_dir, split_name):
    print(f"\nCopying {split_name}...")
    for img_path, lbl_path in file_list:
        # Copy image
        dst_img = dst_img_dir / img_path.name
        shutil.copy(str(img_path), str(dst_img))
        # Copy label
        dst_lbl = dst_lbl_dir / lbl_path.name
        shutil.copy(str(lbl_path), str(dst_lbl))
    print(f"  Copied {len(file_list)} files")

copy_files(train_set, new_train_img_dir, new_train_lbl_dir, "train")
copy_files(val_set, new_val_img_dir, new_val_lbl_dir, "val")
copy_files(test_set, new_test_img_dir, new_test_lbl_dir, "test")

print("\n✅ Done! New dataset split created.")
print("Now update person.yaml to use:")
print("  train: images/train_new")
print("  val: images/val_new")
print("  test: images/test")
