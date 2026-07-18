
"""Rebuild entire dataset with proper non-overlapping train/val/test splits."""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

base_dir = Path(r"D:\PythonProject\person_dataset")

# Clear all new directories
for split in ["train_new", "val_new", "test"]:
    for type_dir in ["images", "labels"]:
        dir_path = base_dir / type_dir / split
        if dir_path.exists():
            for f in dir_path.glob("*"):
                os.remove(str(f))

# Get all unique labeled images from original train+val
train_img_dir = base_dir / "images" / "train"
val_img_dir = base_dir / "images" / "val"
train_lbl_dir = base_dir / "labels" / "train"
val_lbl_dir = base_dir / "labels" / "val"

unique_images = {}  # stem_lower -> (img_path, lbl_path)

# Process train
for img_path in train_img_dir.glob("*.JPG"):
    stem_lower = img_path.stem.lower()
    if stem_lower not in unique_images:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            unique_images[stem_lower] = (img_path, lbl_path)

for img_path in train_img_dir.glob("*.jpg"):
    stem_lower = img_path.stem.lower()
    if stem_lower not in unique_images:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            unique_images[stem_lower] = (img_path, lbl_path)

# Process val
for img_path in val_img_dir.glob("*.JPG"):
    stem_lower = img_path.stem.lower()
    if stem_lower not in unique_images:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            unique_images[stem_lower] = (img_path, lbl_path)

for img_path in val_img_dir.glob("*.jpg"):
    stem_lower = img_path.stem.lower()
    if stem_lower not in unique_images:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            unique_images[stem_lower] = (img_path, lbl_path)

print(f"Total unique labeled images: {len(unique_images)}")

# Shuffle and split: 70% train / 20% val / 10% test
items = list(unique_images.values())
random.shuffle(items)

total = len(items)
train_count = int(total * 0.7)
val_count = int(total * 0.2)
test_count = total - train_count - val_count

print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

train_set = items[:train_count]
val_set = items[train_count:train_count+val_count]
test_set = items[train_count+val_count:]

# Copy files
def copy_files(file_list, img_dir, lbl_dir, split_name):
    print(f"\nCopying {split_name}...")
    for img_path, lbl_path in file_list:
        shutil.copy(str(img_path), str(img_dir / img_path.name))
        shutil.copy(str(lbl_path), str(lbl_dir / lbl_path.name))
    print(f"  Copied {len(file_list)} files")

copy_files(train_set, base_dir / "images" / "train_new", base_dir / "labels" / "train_new", "train")
copy_files(val_set, base_dir / "images" / "val_new", base_dir / "labels" / "val_new", "val")
copy_files(test_set, base_dir / "images" / "test", base_dir / "labels" / "test", "test")

# Verify
print("\n✅ Done! Rebuilt full dataset with proper splits.")
