import argparse
import csv
import os
import time

from ultralytics import YOLO


def parse_csv_list(raw, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def run_one(model_path, data_yaml, imgsz, batch, fraction, device):
    model = YOLO(model_path)
    t0 = time.time()
    res = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=0.001,
        iou=0.6,
        verbose=False,
        plots=False,
        fraction=fraction,
    )
    elapsed = time.time() - t0
    spd = res.speed or {}
    infer_ms = float(spd.get("inference", 0.0))
    return {
        "model": os.path.basename(model_path),
        "imgsz": imgsz,
        "batch": batch,
        "fraction": fraction,
        "elapsed_sec": round(elapsed, 1),
        "inference_ms": round(infer_ms, 3),
        "fps": round(1000.0 / infer_ms, 2) if infer_ms > 0 else 0.0,
        "mAP50": round(float(res.box.map50), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Runtime-first model/imgsz/batch sweep.")
    ap.add_argument("--models", default="person_dataset/yolo12n.pt,person_dataset/yolo12s.pt")
    ap.add_argument("--imgsz", default="640,768,896")
    ap.add_argument("--batches", default="8,12,16")
    ap.add_argument("--data", default="person_dataset/person.yaml")
    ap.add_argument("--fraction", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="runtime_sweep.tsv")
    args = ap.parse_args()

    models = parse_csv_list(args.models, str)
    imgsz_list = parse_csv_list(args.imgsz, int)
    batch_list = parse_csv_list(args.batches, int)

    rows = []
    total = len(models) * len(imgsz_list) * len(batch_list)
    idx = 0
    for m in models:
        for s in imgsz_list:
            for b in batch_list:
                idx += 1
                print(f"[{idx}/{total}] model={m} imgsz={s} batch={b}")
                try:
                    rows.append(run_one(m, args.data, s, b, args.fraction, args.device))
                except Exception as e:  # keep sweep robust
                    rows.append(
                        {
                            "model": os.path.basename(m),
                            "imgsz": s,
                            "batch": b,
                            "fraction": args.fraction,
                            "elapsed_sec": 0.0,
                            "inference_ms": 0.0,
                            "fps": 0.0,
                            "mAP50": 0.0,
                            "mAP50_95": 0.0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "error": str(e),
                        }
                    )

    rows.sort(key=lambda r: (r.get("inference_ms", 999999), -r.get("mAP50", 0)))
    fields = [
        "model",
        "imgsz",
        "batch",
        "fraction",
        "elapsed_sec",
        "inference_ms",
        "fps",
        "mAP50",
        "mAP50_95",
        "precision",
        "recall",
        "error",
    ]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] wrote {args.out} ({len(rows)} rows)")
    if rows:
        print("[TOP-3]")
        for r in rows[:3]:
            print(r)


if __name__ == "__main__":
    main()
