import argparse
import csv
import json
import os
import subprocess
import sys
import time


def parse_csv_list(raw, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def run_eval(model, data, imgsz, conf, iou):
    env = os.environ.copy()
    env["AUTORESEARCH_DEPLOY_CONF"] = str(conf)
    env["AUTORESEARCH_DEPLOY_IOU_NMS"] = str(iou)
    cmd = [sys.executable, "evaluate.py", "--model", model, "--data", data, "--imgsz", str(imgsz)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - t0

    metrics = {}
    if os.path.exists("last_metrics.json"):
        try:
            with open("last_metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    return {
        "conf": conf,
        "iou_nms": iou,
        "elapsed_sec": round(elapsed, 1),
        "returncode": p.returncode,
        "cds": metrics.get("cds", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "inference_ms": metrics.get("inference_ms", 0.0),
        "precision_gate": metrics.get("precision_gate", "N/A"),
        "recall_gate": metrics.get("recall_gate", "N/A"),
        "latency_gate": metrics.get("latency_gate", "N/A"),
        "target_met": bool(metrics.get("target_met", False)),
        "error": "" if p.returncode == 0 else (p.stderr[-300:] or p.stdout[-300:]),
    }


def main():
    ap = argparse.ArgumentParser(description="Sweep deployment conf/iou thresholds against evaluate contract.")
    ap.add_argument("--model", required=True, help="Path to .pt model")
    ap.add_argument("--data", default="person_dataset/person.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", default="0.15,0.20,0.25,0.30")
    ap.add_argument("--iou", default="0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--out", default="threshold_sweep.tsv")
    args = ap.parse_args()

    conf_list = parse_csv_list(args.conf, float)
    iou_list = parse_csv_list(args.iou, float)

    rows = []
    total = len(conf_list) * len(iou_list)
    idx = 0
    for conf in conf_list:
        for iou in iou_list:
            idx += 1
            print(f"[{idx}/{total}] conf={conf:.2f} iou={iou:.2f}")
            rows.append(run_eval(args.model, args.data, args.imgsz, conf, iou))

    rows.sort(
        key=lambda r: (
            0 if r.get("target_met") else 1,
            -float(r.get("cds", 0.0)),
            -float(r.get("recall", 0.0)),
        )
    )

    fields = [
        "conf",
        "iou_nms",
        "elapsed_sec",
        "returncode",
        "cds",
        "precision",
        "recall",
        "inference_ms",
        "precision_gate",
        "recall_gate",
        "latency_gate",
        "target_met",
        "error",
    ]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] wrote {args.out} ({len(rows)} rows)")
    if rows:
        print("[BEST-3]")
        for r in rows[:3]:
            print(r)


if __name__ == "__main__":
    main()
