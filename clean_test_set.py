
"""Clean test set to only include images with corresponding labels."""

import os
from pathlib import Path

base_dir = Path(r"D:\PythonProject\person_dataset")
test_img_dir = base_dir / "images" / "test"
test_lbl_dir = base_dir / "labels" / "test"

# Get all labels
labels = set()
for lbl_path in test_lbl_dir.glob("*.txt"):
    labels.add(lbl_path.stem)

# Remove images without labels
count_removed = 0
for img_path in list(test_img_dir.glob("*.JPG")) + list(test_img_dir.glob("*.jpg")):
    stem = img_path.stem
    if stem not in labels:
        os.remove(str(img_path))
        count_removed += 1

print(f"Removed {count_removed} images without labels")
print(f"Final test set: {len(labels)} images")
