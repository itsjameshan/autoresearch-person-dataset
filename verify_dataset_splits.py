
"""Verify dataset splits - no overlap, all test images have labels."""

import os
from pathlib import Path

base_dir = Path(r"D:\PythonProject\person_dataset")

# Get all image stems for each split
def get_image_stems(img_dir):
    stems = set()
    for ext in [".jpg", ".JPG"]:
        for img_path in img_dir.glob(f"*{ext}"):
            stems.add(img_path.stem)
    return stems

train_stems = get_image_stems(base_dir / "images" / "train_new")
val_stems = get_image_stems(base_dir / "images" / "val_new")
test_stems = get_image_stems(base_dir / "images" / "test")

# Check for overlaps
print("=== Checking for dataset overlap ===")
print(f"Train images: {len(train_stems)}")
print(f"Val images: {len(val_stems)}")
print(f"Test images: {len(test_stems)}")

train_val_overlap = train_stems & val_stems
train_test_overlap = train_stems & test_stems
val_test_overlap = val_stems & test_stems

print(f"\nTrain <-> Val overlap: {len(train_val_overlap)}")
print(f"Train <-> Test overlap: {len(train_test_overlap)}")
print(f"Val <-> Test overlap: {len(val_test_overlap)}")

# Check test labels
print("\n=== Checking test labels ===")
test_lbl_dir = base_dir / "labels" / "test"
test_lbl_stems = set()
for lbl_path in test_lbl_dir.glob("*.txt"):
    test_lbl_stems.add(lbl_path.stem)

print(f"Test labels found: {len(test_lbl_stems)}")

missing_labels = test_stems - test_lbl_stems
extra_labels = test_lbl_stems - test_stems

print(f"Test images missing labels: {len(missing_labels)}")
print(f"Extra test labels (no image): {len(extra_labels)}")

if len(missing_labels) > 0:
    print("\nFirst 10 missing labels:")
    for stem in list(missing_labels)[:10]:
        print(f"  {stem}")

# Summary
print("\n=== Summary ===")
all_ok = len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0 and len(missing_labels) == 0
if all_ok:
    print("✅ All checks passed!")
else:
    print("❌ Issues found!")
