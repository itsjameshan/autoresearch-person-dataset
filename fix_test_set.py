
"""Fix test set - remove orphaned images and verify all have labels."""

import os
import shutil
from pathlib import Path

base_dir = Path(r"D:\PythonProject\person_dataset")
test_img_dir = base_dir / "images" / "test"
test_lbl_dir = base_dir / "labels" / "test"

# Get all label stems
label_stems = set()
for lbl_path in test_lbl_dir.glob("*.txt"):
    label_stems.add(lbl_path.stem)

print(f"Found {len(label_stems)} label files")

# Remove all images that don't have labels
removed = 0
for img_path in list(test_img_dir.glob("*.JPG")) + list(test_img_dir.glob("*.jpg")):
    stem = img_path.stem
    if stem not in label_stems:
        try:
            os.remove(str(img_path))
            removed += 1
        except:
            pass

print(f"Removed {removed} orphaned images")

# Final count
final_imgs = len(list(test_img_dir.glob("*.JPG"))) + len(list(test_img_dir.glob("*.jpg")))
final_lbls = len(list(test_lbl_dir.glob("*.txt")))
print(f"Final test set: {final_imgs} images, {final_lbls} labels")
