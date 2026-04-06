"""
Build pipeline validation set by stitching tiles back into large images.

Selects complete 3x4 grid images, stitches tiles, and merges GT labels
into full-image coordinates for pipeline evaluation.
"""

import os
import re
import glob
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image

TILE_SIZE = 1280
OUTPUT_DIR = "pipeline_val"
NUM_IMAGES = 8  # how many large images to build


def find_complete_grids(image_dirs, label_dirs):
    """Find base images where all tiles exist (across train+val) with labels."""
    tile_map = defaultdict(dict)  # base -> {(row,col): (img_path, label_path)}

    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        for img_path in (glob.glob(os.path.join(img_dir, "*.JPG")) +
                         glob.glob(os.path.join(img_dir, "*.jpg"))):
            name = Path(img_path).stem
            # Parse: BASE_row_col
            parts = name.rsplit("_", 2)
            if len(parts) < 3:
                continue
            try:
                row = int(parts[-2])
                col = int(parts[-1])
            except ValueError:
                continue
            base = "_".join(parts[:-2])
            lbl_path = os.path.join(lbl_dir, f"{name}.txt")
            if os.path.exists(lbl_path):
                tile_map[base][(row, col)] = (img_path, lbl_path)

    # Find complete grids (at least 3x4 = 12 tiles)
    complete = {}
    for base, tiles in tile_map.items():
        if not tiles:
            continue
        max_row = max(r for r, c in tiles.keys())
        max_col = max(c for r, c in tiles.keys())
        expected = (max_row + 1) * (max_col + 1)
        if len(tiles) == expected and expected >= 6:
            total_labels = sum(
                len(open(lp).readlines()) for _, lp in tiles.values()
            )
            complete[base] = {
                "tiles": tiles,
                "rows": max_row + 1,
                "cols": max_col + 1,
                "total_labels": total_labels,
            }

    return complete


def stitch_image(tiles_info):
    """Stitch tiles into a large image. Returns (PIL Image, label lines in absolute coords)."""
    rows = tiles_info["rows"]
    cols = tiles_info["cols"]
    tiles = tiles_info["tiles"]

    # Create large canvas
    W = cols * TILE_SIZE
    H = rows * TILE_SIZE
    canvas = Image.new("RGB", (W, H))

    all_labels = []

    for (r, c), (img_path, lbl_path) in tiles.items():
        tile = Image.open(img_path)
        x_offset = c * TILE_SIZE
        y_offset = r * TILE_SIZE
        canvas.paste(tile, (x_offset, y_offset))

        # Convert YOLO labels to absolute coords on large image
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                cx = float(parts[1]) * TILE_SIZE + x_offset
                cy = float(parts[2]) * TILE_SIZE + y_offset
                w = float(parts[3]) * TILE_SIZE
                h = float(parts[4]) * TILE_SIZE

                # Convert to normalized coords for large image
                cx_norm = cx / W
                cy_norm = cy / H
                w_norm = w / W
                h_norm = h / H

                all_labels.append(f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    return canvas, all_labels, W, H


def main():
    base_dir = Path(__file__).parent
    train_img = base_dir / "images" / "train"
    val_img = base_dir / "images" / "val"
    train_lbl = base_dir / "labels" / "train"
    val_lbl = base_dir / "labels" / "val"

    image_dirs = [str(train_img), str(val_img)]
    label_dirs = [str(train_lbl), str(val_lbl)]

    print("Scanning for complete tile grids...")
    complete = find_complete_grids(image_dirs, label_dirs)
    print(f"Found {len(complete)} complete grid images")

    if not complete:
        print("ERROR: No complete grids found!")
        return

    # Sort by most labels (= most people = best for evaluation)
    sorted_bases = sorted(complete.keys(),
                          key=lambda b: complete[b]["total_labels"],
                          reverse=True)

    # Take top N with most people
    selected = sorted_bases[:NUM_IMAGES]

    print(f"\nSelected {len(selected)} images for pipeline_val:")
    for base in selected:
        info = complete[base]
        print(f"  {base}: {info['rows']}x{info['cols']} grid, {info['total_labels']} GT boxes")

    # Create output directory
    out_dir = base_dir / OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    for base in selected:
        info = complete[base]
        print(f"\nStitching {base}...")
        canvas, labels, W, H = stitch_image(info)

        # Save image
        img_out = out_dir / f"{base}.JPG"
        canvas.save(str(img_out), quality=95)
        print(f"  Saved {img_out} ({W}x{H})")

        # Save labels (YOLO format, normalized to large image)
        lbl_out = out_dir / f"{base}.txt"
        with open(lbl_out, "w") as f:
            f.write("\n".join(labels) + "\n" if labels else "")
        print(f"  Saved {lbl_out} ({len(labels)} boxes)")

    print(f"\nDone! Pipeline validation set: {out_dir}/")
    print(f"Total: {len(selected)} large images with GT annotations")


if __name__ == "__main__":
    main()
