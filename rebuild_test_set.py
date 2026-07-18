
"""Rebuild test set completely - clear and recreate from labeled data."""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

base_dir = Path(r"D:\PythonProject\person_dataset")

# Clear existing test directories
test_img_dir = base_dir / "images" / "test"
test_lbl_dir = base_dir / "labels" / "test"

# Remove all files in test directories
for f in test_img_dir.glob("*"):
    os.remove(str(f))
for f in test_lbl_dir.glob("*"):
    os.remove(str(f))

# Get all labeled images from original train+val
train_img_dir = base_dir / "images" / "train"
val_img_dir = base_dir / "images" / "val"
train_lbl_dir = base_dir / "labels" / "train"
val_lbl_dir = base_dir / "labels" / "val"

all_images = []

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

# Shuffle and take 10% for test
random.shuffle(all_images)
test_count = int(len(all_images) * 0.1)
test_set = all_images[:test_count]

print(f"Creating test set with {test_count} images...")

for img_path, lbl_path in test_set:
    shutil.copy(str(img_path), str(test_img_dir / img_path.name))
    shutil.copy(str(lbl_path), str(test_lbl_dir / lbl_path.name))

final_imgs = len(list(test_img_dir.glob("*.JPG"))) + len(list(test_img_dir.glob("*.jpg")))
final_lbls = len(list(test_lbl_dir.glob("*.txt")))
print(f"✅ Done! Test set: {final_imgs} images, {final_lbls} labels")
