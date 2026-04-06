"""
data_quality.py — Annotation quality checker.
Independent from the autoresearch loop. Run manually after dataset updates.

Usage:
  python data_quality.py --data person.yaml --model autoresearch_runs/current/weights/best.pt
"""

import os
import glob
import random
import numpy as np
from pathlib import Path

import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _safe_load


def load_labels(label_dir):
    """Load all YOLO labels from a directory. Returns dict: filename -> list of (cls, cx, cy, w, h)."""
    labels = {}
    for f in glob.glob(os.path.join(label_dir, "*.txt")):
        name = Path(f).stem
        boxes = []
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append(tuple(float(x) for x in parts[:5]))
        labels[name] = boxes
    return labels


def check_missing_labels(image_dir, label_dir):
    """Find images without corresponding label files."""
    images = set(Path(f).stem for f in
                 glob.glob(os.path.join(image_dir, "*.JPG")) +
                 glob.glob(os.path.join(image_dir, "*.jpg")) +
                 glob.glob(os.path.join(image_dir, "*.png")))
    labels = set(Path(f).stem for f in glob.glob(os.path.join(label_dir, "*.txt")))
    missing = images - labels
    extra = labels - images
    return missing, extra


def check_box_tightness(labels):
    """Detect outlier box sizes (too large or too small)."""
    all_areas = []
    per_file = {}
    for name, boxes in labels.items():
        areas = [w * h for _, _, _, w, h in boxes]
        all_areas.extend(areas)
        per_file[name] = areas

    if not all_areas:
        return [], 0, 0

    mean_area = np.mean(all_areas)
    std_area = np.std(all_areas)
    outlier_files = []
    for name, areas in per_file.items():
        for a in areas:
            if abs(a - mean_area) > 3 * std_area:
                outlier_files.append((name, a, mean_area, std_area))
                break
    return outlier_files, mean_area, std_area


def check_model_assisted_missing(model_path, image_dir, label_dir, conf_thresh=0.8, sample_pct=0.3):
    """Use a trained model to find potential missing annotations.
    High-confidence predictions with no GT match suggest missed labels.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    images = sorted(
        glob.glob(os.path.join(image_dir, "*.JPG")) +
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png")))

    sample_size = max(1, int(len(images) * sample_pct))
    sampled = random.sample(images, sample_size)

    suspicious = []
    for img_path in sampled:
        name = Path(img_path).stem
        label_path = os.path.join(label_dir, f"{name}.txt")

        # Load GT
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        gt_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        # Run inference
        results = model(img_path, conf=conf_thresh, iou=0.35, device=0, verbose=False)
        if results[0].boxes is None:
            continue

        pred_boxes = results[0].boxes.xyxyn.cpu().numpy()  # normalized
        pred_confs = results[0].boxes.conf.cpu().numpy()

        for pi in range(len(pred_boxes)):
            pb = pred_boxes[pi].tolist()
            matched = False
            for gb in gt_boxes:
                # Simple center distance check
                pcx = (pb[0] + pb[2]) / 2
                pcy = (pb[1] + pb[3]) / 2
                gcx = (gb[0] + gb[2]) / 2
                gcy = (gb[1] + gb[3]) / 2
                dist = ((pcx - gcx)**2 + (pcy - gcy)**2) ** 0.5
                if dist < 0.05:  # within 5% of image
                    matched = True
                    break
            if not matched:
                suspicious.append({
                    "image": name,
                    "pred_conf": float(pred_confs[pi]),
                    "pred_box": pb,
                })

    return suspicious


def run_quality_check(data_yaml, model_path=None):
    """Run full quality check and print report."""
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    base = Path(data_yaml).parent
    train_img = base / cfg.get("train", "images/train")
    train_lbl = Path(str(train_img).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))
    val_img = base / cfg.get("val", "images/val")
    val_lbl = Path(str(val_img).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    # Missing labels
    for split, img_d, lbl_d in [("train", train_img, train_lbl), ("val", val_img, val_lbl)]:
        missing, extra = check_missing_labels(str(img_d), str(lbl_d))
        print(f"\n[{split}] Images: {len(list(img_d.glob('*')))} | Labels: {len(list(lbl_d.glob('*.txt')))}")
        if missing:
            print(f"  WARNING: {len(missing)} images without labels: {list(missing)[:5]}...")
        if extra:
            print(f"  WARNING: {len(extra)} labels without images: {list(extra)[:5]}...")
        if not missing and not extra:
            print(f"  OK: all images have labels")

    # Box tightness
    print("\n--- Box Tightness ---")
    all_labels = load_labels(str(train_lbl))
    outliers, mean_a, std_a = check_box_tightness(all_labels)
    print(f"Mean box area: {mean_a:.6f} (normalized), std: {std_a:.6f}")
    if outliers:
        print(f"WARNING: {len(outliers)} files with outlier box sizes:")
        for name, area, _, _ in outliers[:10]:
            print(f"  {name}: area={area:.6f} (mean={mean_a:.6f})")
    else:
        print("OK: no extreme outlier boxes")

    # Model-assisted missing annotation check
    if model_path and os.path.exists(model_path):
        print("\n--- Model-Assisted Missing Label Check (30% sample) ---")
        suspicious = check_model_assisted_missing(model_path, str(train_img), str(train_lbl))
        if suspicious:
            print(f"WARNING: {len(suspicious)} high-confidence predictions without GT match:")
            for s in suspicious[:10]:
                print(f"  {s['image']}: conf={s['pred_conf']:.2f}")
        else:
            print("OK: no suspicious unmatched predictions")
    else:
        print("\n--- Model-Assisted Check: skipped (no model provided) ---")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="person.yaml")
    parser.add_argument("--model", default=None, help="Trained model for missing label detection")
    args = parser.parse_args()
    run_quality_check(args.data, args.model)
