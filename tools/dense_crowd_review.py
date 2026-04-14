import argparse
import csv
import glob
import os
from pathlib import Path

from ultralytics import YOLO


def count_labels(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def resolve_val_paths(data_yaml):
    import yaml

    with open(data_yaml, "r", encoding="utf-8", errors="replace") as f:
        cfg = yaml.safe_load(f)
    base = Path(data_yaml).parent
    val_rel = cfg.get("val", "images/val")
    val_img_dir = Path(val_rel) if os.path.isabs(val_rel) else base / val_rel
    val_label_dir = Path(str(val_img_dir).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))
    imgs = sorted(
        set(
            glob.glob(str(val_img_dir / "*.JPG"))
            + glob.glob(str(val_img_dir / "*.jpg"))
            + glob.glob(str(val_img_dir / "*.png"))
        )
    )
    return imgs, val_label_dir


def main():
    ap = argparse.ArgumentParser(description="Review top-k dense crowd validation images.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="person_dataset/person.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--save-dir", default="dense_review")
    ap.add_argument("--out", default="dense_review.tsv")
    args = ap.parse_args()

    imgs, label_dir = resolve_val_paths(args.data)
    rows = []
    for p in imgs:
        stem = Path(p).stem
        gt = count_labels(str(label_dir / f"{stem}.txt"))
        rows.append({"image": p, "gt_count": gt})
    rows.sort(key=lambda r: r["gt_count"], reverse=True)
    rows = rows[: args.topk]

    os.makedirs(args.save_dir, exist_ok=True)
    model = YOLO(args.model)
    report = []
    for i, r in enumerate(rows, start=1):
        p = r["image"]
        pred = model(p, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=0, verbose=False)[0]
        pred_count = int(len(pred.boxes)) if pred.boxes is not None else 0
        err = abs(pred_count - r["gt_count"])
        out_img = os.path.join(args.save_dir, f"{i:02d}_{Path(p).name}")
        pred.save(filename=out_img)
        report.append(
            {
                "rank": i,
                "image": Path(p).name,
                "gt_count": r["gt_count"],
                "pred_count": pred_count,
                "abs_error": err,
                "saved_vis": out_img,
            }
        )
        print(f"[{i}/{len(rows)}] {Path(p).name}: gt={r['gt_count']} pred={pred_count} abs_err={err}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rank", "image", "gt_count", "pred_count", "abs_error", "saved_vis"],
            delimiter="\t",
        )
        w.writeheader()
        for row in report:
            w.writerow(row)
    print(f"[DONE] wrote {args.out}, visuals in {args.save_dir}")


if __name__ == "__main__":
    main()
