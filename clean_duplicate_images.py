
"""Remove duplicate images with different case extensions."""

import os
from pathlib import Path

base_dir = Path(r"D:\PythonProject\person_dataset")
test_img_dir = base_dir / "images" / "test"

# Track which stems we've seen
seen = set()
removed = 0

for img_path in test_img_dir.glob("*.jpg"):
    stem_lower = img_path.stem.lower()
    if stem_lower in seen:
        os.remove(str(img_path))
        removed += 1
    else:
        seen.add(stem_lower)

for img_path in test_img_dir.glob("*.JPG"):
    stem_lower = img_path.stem.lower()
    if stem_lower in seen:
        os.remove(str(img_path))
        removed += 1
    else:
        seen.add(stem_lower)

print(f"Removed {removed} duplicate images")
final_imgs = len(list(test_img_dir.glob("*.JPG"))) + len(list(test_img_dir.glob("*.jpg")))
final_lbls = len(list((base_dir / "labels" / "test").glob("*.txt")))
print(f"Final test set: {final_imgs} images, {final_lbls} labels")
