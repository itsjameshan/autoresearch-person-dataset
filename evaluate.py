"""
evaluate.py — Evaluation contract for the autoresearch loop (agent edits train.py only).

Computes CDS (Crowd Detection Score) and all metrics for the autoresearch
keep/discard decision. CDS includes a latency term derived from inference_ms.
"""

import json
import os
import sys
import threading
import time
import glob
import numpy as np

# ── Fix torch.load for newer PyTorch versions ──
import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from ultralytics import YOLO
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# CONSTANTS — these define the evaluation contract, never change
# ══════════════════════════════════════════════════════════════

# CDS weights (must sum to 1.0)
W_MAP50 = 0.25
W_MAP50_95 = 0.25
W_F1 = 0.10
W_COUNTING = 0.15
W_SMALL_OBJ = 0.15
W_LATENCY = 0.10

# Latency score: linear map inference_ms -> [0, 1], higher = faster.
# Tune for your GPU; defaults suit ~1280 single-tile on mid/high NVIDIA.
LATENCY_MS_FULL_SCORE = 25.0   # at or below -> latency_score = 1.0
LATENCY_MS_ZERO_SCORE = 220.0  # at or above -> latency_score = 0.0

# Deployment thresholds (from project doc)
DEPLOY_CONF = 0.25
DEPLOY_IOU_NMS = 0.35

# Quality gates
PRECISION_GATE = 0.90
RECALL_GATE = 0.85

# 延迟硬门槛（与 inference_ms 同一测法：部署 conf/iou、warmup 后 val 子集均值）。
# 默认按 RTX 5070 级 + 1280 单 tile、中等体量模型（如 yolo12s）的常见区间略留余量。
_LATENCY_GATE_DEFAULT_MS = 115.0


def get_latency_gate_ms():
    """返回本次评估使用的延迟上限（ms）。优先读环境变量，便于甲方机器/笔记本验收不调代码。"""
    raw = os.environ.get("AUTORESEARCH_LATENCY_GATE_MS", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            print(
                "evaluate: invalid AUTORESEARCH_LATENCY_GATE_MS, using default",
                file=sys.stderr,
            )
    return _LATENCY_GATE_DEFAULT_MS


# Small object threshold: GT box area < 0.5% of image area
SMALL_OBJ_AREA_THRESH = 0.005

# Inference timing
WARMUP_IMAGES = 5
TIMING_IMAGES = 20

# Evaluation watchdog (cross-platform; signal.SIGALRM is unavailable on Windows)
_EVAL_TIMEOUT_DEFAULT_SEC = 3600.0  # 1 hour, generous upper bound
METRICS_JSON_PATH = "last_metrics.json"


def get_eval_timeout_sec():
    """Return the evaluation watchdog timeout in seconds.

    Override via env var AUTORESEARCH_EVAL_TIMEOUT_SEC. Invalid values fall
    back to the default.
    """
    raw = os.environ.get("AUTORESEARCH_EVAL_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            print(
                "evaluate: invalid AUTORESEARCH_EVAL_TIMEOUT_SEC, using default",
                file=sys.stderr,
            )
    return _EVAL_TIMEOUT_DEFAULT_SEC


def _start_eval_watchdog():
    """Start a daemon watchdog thread that hard-exits if eval hangs.

    Returns a sentinel dict; set sentinel["done"] = True before normal return
    so the watchdog skips the kill.
    """
    sentinel = {"done": False}
    timeout = get_eval_timeout_sec()

    def _killer():
        time.sleep(timeout)
        if not sentinel["done"]:
            print(
                f"evaluate: TIMEOUT after {timeout:.0f}s — forcing exit",
                file=sys.stderr,
            )
            sys.stderr.flush()
            os._exit(124)

    threading.Thread(target=_killer, daemon=True).start()
    return sentinel


def load_gt_boxes(label_path, img_w=1280, img_h=1280):
    """Load YOLO-format ground truth boxes from a label file.
    Returns list of [x1, y1, x2, y2] in pixel coords and normalized areas.
    """
    boxes = []
    areas = []
    if not os.path.exists(label_path):
        return boxes, areas
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            areas.append(w * h)  # normalized area (fraction of image)
    return boxes, areas


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Greedy matching of predictions to GT boxes.
    Returns number of matched GT boxes.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0
    matched_gt = set()
    # Sort predictions by confidence (highest first) if available
    for pb in pred_boxes:
        best_iou = 0
        best_gt = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_thresh and best_gt >= 0:
            matched_gt.add(best_gt)
    return len(matched_gt)


def _match_with_indices(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """Greedy IoU matching that returns full index breakdown for FP/FN analysis.

    Predictions are visited in descending confidence order so highest-confidence
    detections claim GT boxes first.

    Returns
    -------
    matches : list of dict {"pred": int, "gt": int, "iou": float}
    fp_indices : list of int
        Indices into pred_boxes that did not match any GT.
    fn_indices : list of int
        Indices into gt_boxes that no prediction matched.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    if n_pred == 0:
        return [], [], list(range(n_gt))
    if n_gt == 0:
        return [], list(range(n_pred)), []

    if pred_scores and len(pred_scores) == n_pred:
        order = sorted(range(n_pred), key=lambda i: pred_scores[i], reverse=True)
    else:
        order = list(range(n_pred))

    matched_gt = {}
    matches = []
    for pi in order:
        best_iou = 0.0
        best_gt = -1
        for gi in range(n_gt):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred_boxes[pi], gt_boxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_thresh and best_gt >= 0:
            matched_gt[best_gt] = pi
            matches.append({"pred": pi, "gt": best_gt, "iou": float(best_iou)})

    matched_preds = {m["pred"] for m in matches}
    fp_indices = [i for i in range(n_pred) if i not in matched_preds]
    fn_indices = [i for i in range(n_gt) if i not in matched_gt]
    return matches, fp_indices, fn_indices


def _padded_crop_box(box, img_w, img_h, pad_frac=0.2):
    """Inflate a [x1,y1,x2,y2] box by pad_frac on each side, clamped to image."""
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * pad_frac
    py = h * pad_frac
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(img_w, int(round(x2 + px)))
    cy2 = min(img_h, int(round(y2 + py)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def _export_fpfn_for_image(
    img_path,
    img_stem,
    gt_boxes,
    pred_boxes,
    pred_scores,
    matches,
    fp_indices,
    fn_indices,
    out_dir,
    img_w,
    img_h,
    iou_thresh,
    top_k,
):
    """Write per-image FP/FN JSON + top-K crops. Returns summary dict for index."""
    from PIL import Image

    crops_dir = out_dir / "crops"
    per_image_dir = out_dir / "per_image"
    crops_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)

    # Rank FPs by confidence (highest first); FNs by area (largest first — most visually obvious failures)
    if pred_scores and len(pred_scores) == len(pred_boxes):
        fp_ranked = sorted(fp_indices, key=lambda i: pred_scores[i], reverse=True)
    else:
        fp_ranked = list(fp_indices)

    def _gt_area(gi):
        gb = gt_boxes[gi]
        return max(0.0, (gb[2] - gb[0]) * (gb[3] - gb[1]))
    fn_ranked = sorted(fn_indices, key=_gt_area, reverse=True)

    fp_top = fp_ranked[:top_k]
    fn_top = fn_ranked[:top_k]

    exported_fp = []
    exported_fn = []
    img = None
    if fp_top or fn_top:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"evaluate: failed to open {img_path} for cropping: {e}", file=sys.stderr)
            img = None

    if img is not None:
        for rank, pi in enumerate(fp_top):
            crop_box = _padded_crop_box(pred_boxes[pi], img_w, img_h)
            if crop_box is None:
                continue
            crop = img.crop(crop_box)
            fname = f"{img_stem}_fp_{rank:02d}_idx{pi}.jpg"
            try:
                crop.save(crops_dir / fname, format="JPEG", quality=85)
                exported_fp.append(f"crops/{fname}")
            except OSError as e:
                print(f"evaluate: failed to save crop {fname}: {e}", file=sys.stderr)

        for rank, gi in enumerate(fn_top):
            crop_box = _padded_crop_box(gt_boxes[gi], img_w, img_h)
            if crop_box is None:
                continue
            crop = img.crop(crop_box)
            fname = f"{img_stem}_fn_{rank:02d}_idx{gi}.jpg"
            try:
                crop.save(crops_dir / fname, format="JPEG", quality=85)
                exported_fn.append(f"crops/{fname}")
            except OSError as e:
                print(f"evaluate: failed to save crop {fname}: {e}", file=sys.stderr)

    per_image = {
        "image_path": str(img_path),
        "image_size": [img_w, img_h],
        "iou_threshold": iou_thresh,
        "gt_boxes": [[round(v, 2) for v in b] for b in gt_boxes],
        "pred_boxes": [[round(v, 2) for v in b] for b in pred_boxes],
        "pred_scores": [round(float(s), 4) for s in pred_scores] if pred_scores else [],
        "matches": matches,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "fp_top_indices": fp_top,
        "fn_top_indices": fn_top,
        "exported_fp_crops": exported_fp,
        "exported_fn_crops": exported_fn,
    }
    try:
        with open(per_image_dir / f"{img_stem}.json", "w", encoding="utf-8") as f:
            json.dump(per_image, f, indent=2)
    except OSError as e:
        print(f"evaluate: failed to write per-image json for {img_stem}: {e}", file=sys.stderr)

    return {
        "image_path": str(img_path),
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "matched_count": len(matches),
        "fp_count": len(fp_indices),
        "fn_count": len(fn_indices),
        "fp_top_indices": fp_top,
        "fn_top_indices": fn_top,
        "exported_fp_crops": len(exported_fp),
        "exported_fn_crops": len(exported_fn),
        "per_image_json": f"per_image/{img_stem}.json",
    }


def _resolve_fpfn_dir(cli_value):
    """Resolve FP/FN export directory from CLI arg or env var."""
    if cli_value:
        return Path(cli_value)
    raw = os.environ.get("AUTORESEARCH_FPFN_DIR", "").strip()
    if raw:
        return Path(raw)
    return None


def _resolve_fpfn_topk(cli_value, default=10):
    if cli_value is not None:
        return max(0, int(cli_value))
    raw = os.environ.get("AUTORESEARCH_FPFN_TOPK", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            print("evaluate: invalid AUTORESEARCH_FPFN_TOPK, using default", file=sys.stderr)
    return default


def evaluate_model(
    model_path,
    data_yaml,
    imgsz=1280,
    fpfn_dir=None,
    fpfn_topk=10,
    fpfn_iou=0.5,
):
    """Main evaluation function. Returns dict of all metrics.

    This is the single source of truth for the autoresearch loop.

    fpfn_dir : Path | None
        If set, write per-image FP/FN JSON and top-K failure crops here. Used by
        the v2 Data Curator agent for visual failure analysis. Default behavior
        (None) is unchanged from the legacy contract.
    """
    # ── Watchdog: hard-kill if anything below hangs ──
    _watchdog = _start_eval_watchdog()

    fpfn_enabled = fpfn_dir is not None
    fpfn_index = {}
    if fpfn_enabled:
        fpfn_dir = Path(fpfn_dir)
        fpfn_dir.mkdir(parents=True, exist_ok=True)
        print(f"evaluate: FP/FN export enabled -> {fpfn_dir} (top_k={fpfn_topk}, iou={fpfn_iou})")

    # ── Load model ──
    model = YOLO(model_path)

    # ── Run official validation (mAP, precision, recall) ──
    val_results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=8,
        conf=0.001,
        iou=0.6,
        device=0,
        verbose=False,
        plots=False,
    )

    mAP50 = float(val_results.box.map50)
    mAP50_95 = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)

    # F1 optimal
    f1_optimal = 2 * precision * recall / (precision + recall + 1e-6)

    # ── Counting accuracy & small object recall ──
    # Need to iterate val images with deployment thresholds
    data_dir = Path(data_yaml).parent
    # Try to resolve paths from yaml
    import yaml
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Resolve val image/label dirs
    val_img_rel = data_cfg.get("val", "images/val")
    if os.path.isabs(val_img_rel):
        val_img_dir = Path(val_img_rel)
    else:
        val_img_dir = data_dir / val_img_rel

    # Derive label dir from image dir
    val_label_dir = Path(str(val_img_dir).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))

    val_images = sorted(glob.glob(str(val_img_dir / "*.JPG")) +
                        glob.glob(str(val_img_dir / "*.jpg")) +
                        glob.glob(str(val_img_dir / "*.png")))

    per_image_accs = []
    counting_errors = []
    small_gt_total = 0
    small_gt_matched = 0

    print(f"evaluate: running deployment-threshold inference on {len(val_images)} val images...")
    for i, img_path in enumerate(val_images):
        img_name = Path(img_path).stem
        label_path = val_label_dir / f"{img_name}.txt"

        gt_boxes, gt_areas = load_gt_boxes(str(label_path), imgsz, imgsz)
        gt_count = len(gt_boxes)

        # Run inference with deployment thresholds (single pass per image)
        results = model(
            img_path,
            conf=DEPLOY_CONF,
            iou=DEPLOY_IOU_NMS,
            imgsz=imgsz,
            device=0,
            verbose=False,
        )
        res = results[0]

        if res.boxes is not None and len(res.boxes) > 0:
            pred_boxes = res.boxes.xyxy.cpu().numpy().tolist()
            pred_scores = res.boxes.conf.cpu().numpy().tolist()
            pred_count = len(pred_boxes)
        else:
            pred_boxes = []
            pred_scores = []
            pred_count = 0

        # Counting accuracy per image
        if gt_count > 0:
            acc = max(0, 1 - abs(pred_count - gt_count) / gt_count)
        else:
            acc = 1.0 if pred_count == 0 else 0.0
        per_image_accs.append(acc)
        counting_errors.append(abs(pred_count - gt_count))

        # Small object recall
        for gi, area in enumerate(gt_areas):
            if area < SMALL_OBJ_AREA_THRESH:
                small_gt_total += 1
                best_iou = 0
                for pb in pred_boxes:
                    iou = compute_iou(pb, gt_boxes[gi])
                    best_iou = max(best_iou, iou)
                if best_iou >= 0.5:
                    small_gt_matched += 1

        # Per-image FP/FN export (only when explicitly enabled)
        if fpfn_enabled:
            matches, fp_idx, fn_idx = _match_with_indices(
                pred_boxes, pred_scores, gt_boxes, iou_thresh=fpfn_iou,
            )
            entry = _export_fpfn_for_image(
                img_path=img_path,
                img_stem=img_name,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                matches=matches,
                fp_indices=fp_idx,
                fn_indices=fn_idx,
                out_dir=fpfn_dir,
                img_w=imgsz,
                img_h=imgsz,
                iou_thresh=fpfn_iou,
                top_k=fpfn_topk,
            )
            fpfn_index[img_name] = entry

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(val_images)} images...")

    counting_accuracy = float(np.mean(per_image_accs)) if per_image_accs else 0.0
    counting_mae = float(np.mean(counting_errors)) if counting_errors else 0.0
    small_obj_recall = small_gt_matched / max(small_gt_total, 1)

    # ── Inference timing ──
    timing_imgs = val_images[:WARMUP_IMAGES + TIMING_IMAGES]
    # Warmup
    for img in timing_imgs[:WARMUP_IMAGES]:
        model(img, conf=DEPLOY_CONF, iou=DEPLOY_IOU_NMS, imgsz=imgsz,
              device=0, verbose=False)
    # Timed runs
    times = []
    for img in timing_imgs[WARMUP_IMAGES:]:
        t0 = time.time()
        model(img, conf=DEPLOY_CONF, iou=DEPLOY_IOU_NMS, imgsz=imgsz,
              device=0, verbose=False)
        times.append((time.time() - t0) * 1000)
    inference_ms = float(np.mean(times)) if times else 0.0

    span = LATENCY_MS_ZERO_SCORE - LATENCY_MS_FULL_SCORE
    if span <= 0:
        latency_score = 0.0
    else:
        latency_score = (LATENCY_MS_ZERO_SCORE - inference_ms) / span
        latency_score = float(np.clip(latency_score, 0.0, 1.0))

    # ── Mean confidence of detections ──
    all_confs = []
    for img_path in val_images[:50]:  # Sample 50 images for speed
        results = model(img_path, conf=DEPLOY_CONF, iou=DEPLOY_IOU_NMS,
                        imgsz=imgsz, device=0, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            all_confs.extend(results[0].boxes.conf.cpu().numpy().tolist())
    mean_confidence = float(np.mean(all_confs)) if all_confs else 0.0

    # ── Compute CDS ──
    cds = (W_MAP50 * mAP50 +
           W_MAP50_95 * mAP50_95 +
           W_F1 * f1_optimal +
           W_COUNTING * counting_accuracy +
           W_SMALL_OBJ * small_obj_recall +
           W_LATENCY * latency_score)

    # ── Quality gates ──
    gate_ms = get_latency_gate_ms()
    precision_gate = "PASS" if precision >= PRECISION_GATE else f"FAIL({precision:.2f}<{PRECISION_GATE})"
    recall_gate = "PASS" if recall >= RECALL_GATE else f"FAIL({recall:.2f}<{RECALL_GATE})"
    latency_gate = (
        "PASS"
        if inference_ms <= gate_ms
        else f"FAIL({inference_ms:.1f}ms>{gate_ms}ms)"
    )
    target_met = (
        precision >= PRECISION_GATE
        and recall >= RECALL_GATE
        and inference_ms <= gate_ms
    )

    # ── Peak memory ──
    peak_memory_mb = 0.0
    try:
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif sys.platform == "darwin":
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        elif sys.platform != "win32":
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass

    metrics = {
        "cds": round(cds, 4),
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_optimal": round(f1_optimal, 4),
        "small_obj_recall": round(small_obj_recall, 4),
        "counting_acc": round(counting_accuracy, 4),
        "counting_mae": round(counting_mae, 4),
        "mean_confidence": round(mean_confidence, 4),
        "inference_ms": round(inference_ms, 1),
        "latency_score": round(latency_score, 4),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "precision_gate": precision_gate,
        "recall_gate": recall_gate,
        "latency_gate": latency_gate,
        "latency_gate_ms": round(gate_ms, 1),
        "target_met": target_met,
    }

    # ── Persist metrics to JSON for robust downstream parsing ──
    try:
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except OSError as e:
        print(f"evaluate: failed to write {METRICS_JSON_PATH}: {e}", file=sys.stderr)

    # ── Optional: write metrics to v2 state.db (P1) ──
    state_db_path = os.environ.get("AUTORESEARCH_STATE_DB", "").strip()
    if state_db_path:
        try:
            from person_dataset.autoresearch_v2 import db as _state_db
            sha = _state_db.get_current_git_sha()
            if sha:
                _state_db.init_db(state_db_path)
                _state_db.record_metrics(state_db_path, sha, tile_metrics=metrics)
            else:
                print(
                    "evaluate: state.db write skipped (could not resolve git HEAD)",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"evaluate: state.db write failed: {e}", file=sys.stderr)

    # ── FP/FN index (only when --export-fpfn was used) ──
    if fpfn_enabled:
        total_fp = sum(e["fp_count"] for e in fpfn_index.values())
        total_fn = sum(e["fn_count"] for e in fpfn_index.values())
        index_payload = {
            "version": 1,
            "model_path": str(model_path),
            "data_yaml": str(data_yaml),
            "imgsz": imgsz,
            "iou_threshold": fpfn_iou,
            "topk_per_image": fpfn_topk,
            "deploy_conf": DEPLOY_CONF,
            "deploy_iou_nms": DEPLOY_IOU_NMS,
            "total_images": len(fpfn_index),
            "total_fp": total_fp,
            "total_fn": total_fn,
            "metrics_summary": {
                "cds": metrics["cds"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            },
            "images": fpfn_index,
        }
        try:
            with open(fpfn_dir / "index.json", "w", encoding="utf-8") as f:
                json.dump(index_payload, f, indent=2)
            print(
                f"evaluate: FP/FN index written ({len(fpfn_index)} images, "
                f"{total_fp} FP, {total_fn} FN)"
            )
        except OSError as e:
            print(f"evaluate: failed to write FP/FN index: {e}", file=sys.stderr)

    # Disarm watchdog before returning
    _watchdog["done"] = True
    return metrics


def print_metrics(metrics, epochs_completed=0):
    """Print metrics in grep-friendly structured format."""
    print("---")
    print(f"cds:              {metrics['cds']:.4f}")
    print(f"mAP50:            {metrics['mAP50']:.4f}")
    print(f"mAP50_95:         {metrics['mAP50_95']:.4f}")
    print(f"precision:        {metrics['precision']:.4f}")
    print(f"recall:           {metrics['recall']:.4f}")
    print(f"f1_optimal:       {metrics['f1_optimal']:.4f}")
    print(f"small_obj_recall: {metrics['small_obj_recall']:.4f}")
    print(f"counting_acc:     {metrics['counting_acc']:.4f}")
    print(f"counting_mae:     {metrics['counting_mae']:.4f}")
    print(f"mean_confidence:  {metrics['mean_confidence']:.4f}")
    print(f"inference_ms:     {metrics['inference_ms']}")
    print(f"latency_score:    {metrics['latency_score']:.4f}")
    print(f"peak_memory_mb:   {metrics['peak_memory_mb']}")
    print(f"epochs_completed: {epochs_completed}")
    print(f"precision_gate:   {metrics['precision_gate']}")
    print(f"recall_gate:      {metrics['recall_gate']}")
    print(f"latency_gate:     {metrics['latency_gate']}")
    print(f"latency_gate_ms:  {metrics['latency_gate_ms']}")


# Where pipeline metrics are persisted (parallel to last_metrics.json)
PIPELINE_METRICS_JSON_PATH = "last_pipeline_metrics.json"

# Default IoU threshold for pipeline-level prediction-to-GT matching
PIPELINE_MATCH_IOU_DEFAULT = 0.5


def evaluate_pipeline(
    model_path,
    large_images_dir,
    imgsz=1280,
    overlap=200,
    nms_iou=0.35,
    match_iou=PIPELINE_MATCH_IOU_DEFAULT,
):
    """Large-image pipeline evaluation.

    Simulates deployment: large image → tiles → per-tile detection →
    coordinate restoration → global NMS → full-image P/R + counting.

    Returns
    -------
    dict with keys:
        pipeline_precision, pipeline_recall, pipeline_f1
        pipeline_counting_acc, pipeline_counting_mae
        pipeline_fps, pipeline_total_sec
        pipeline_images, pipeline_images_with_gt
        match_iou, nms_iou
        details: list of per-image dicts with image, pred_count, gt_count,
                 tp, fp, fn, counting_accuracy
    """
    from torchvision.ops import nms as torch_nms

    model = YOLO(model_path)
    large_images = sorted(
        glob.glob(os.path.join(large_images_dir, "*.JPG")) +
        glob.glob(os.path.join(large_images_dir, "*.jpg")) +
        glob.glob(os.path.join(large_images_dir, "*.png"))
    )

    if not large_images:
        print("pipeline: no large images found in", large_images_dir)
        return {}

    results_all = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    counting_accs = []
    counting_errors = []
    t_start = time.time()

    for img_path in large_images:
        from PIL import Image
        img = Image.open(img_path)
        W, H = img.size

        # Tile the image
        all_boxes = []
        all_scores = []
        stride = imgsz - overlap

        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                x1 = min(x0 + imgsz, W)
                y1 = min(y0 + imgsz, H)
                # Crop tile
                tile = img.crop((x0, y0, x1, y1))

                # Detect
                res = model(tile, conf=DEPLOY_CONF, iou=DEPLOY_IOU_NMS,
                            imgsz=imgsz, device=0, verbose=False)

                if res[0].boxes is not None and len(res[0].boxes) > 0:
                    boxes = res[0].boxes.xyxy.cpu().numpy()
                    scores = res[0].boxes.conf.cpu().numpy()

                    # Restore to full-image coordinates
                    for bi in range(len(boxes)):
                        bx1 = boxes[bi][0] + x0
                        by1 = boxes[bi][1] + y0
                        bx2 = boxes[bi][2] + x0
                        by2 = boxes[bi][3] + y0
                        all_boxes.append([bx1, by1, bx2, by2])
                        all_scores.append(float(scores[bi]))

        # Global NMS — keep boxes + scores after dedup
        if all_boxes:
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
            scores_t = torch.tensor(all_scores, dtype=torch.float32)
            keep_idx = torch_nms(boxes_t, scores_t, nms_iou).cpu().numpy().tolist()
            kept_boxes = [all_boxes[i] for i in keep_idx]
            kept_scores = [all_scores[i] for i in keep_idx]
            final_count = len(kept_boxes)
        else:
            kept_boxes = []
            kept_scores = []
            final_count = 0

        # Load GT for this large image if available
        gt_label = os.path.splitext(img_path)[0] + ".txt"
        per_image = {
            "image": os.path.basename(img_path),
            "pred_count": final_count,
            "gt_count": -1,
            "tp": -1,
            "fp": -1,
            "fn": -1,
            "counting_accuracy": -1,
        }

        if os.path.exists(gt_label):
            gt_boxes, _ = load_gt_boxes(gt_label, W, H)
            gt_count = len(gt_boxes)
            # Pipeline-level FP/FN matching (uses helper from P0)
            matches, fp_idx, fn_idx = _match_with_indices(
                kept_boxes, kept_scores, gt_boxes, iou_thresh=match_iou,
            )
            tp = len(matches)
            fp = len(fp_idx)
            fn = len(fn_idx)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            if gt_count > 0:
                acc = max(0, 1 - abs(final_count - gt_count) / gt_count)
            else:
                acc = 1.0 if final_count == 0 else 0.0
            counting_accs.append(acc)
            counting_errors.append(abs(final_count - gt_count))

            per_image.update({
                "gt_count": gt_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "counting_accuracy": round(acc, 4),
            })

        results_all.append(per_image)

    t_total = time.time() - t_start
    n_images = len(large_images)
    n_with_gt = sum(1 for r in results_all if r["gt_count"] >= 0)

    # Aggregate metrics
    if total_tp + total_fp > 0:
        pipeline_precision = total_tp / (total_tp + total_fp)
    else:
        pipeline_precision = 0.0
    if total_tp + total_fn > 0:
        pipeline_recall = total_tp / (total_tp + total_fn)
    else:
        pipeline_recall = 0.0
    pipeline_f1 = (
        2 * pipeline_precision * pipeline_recall
        / (pipeline_precision + pipeline_recall + 1e-6)
    )
    pipeline_counting_acc = float(np.mean(counting_accs)) if counting_accs else -1.0
    pipeline_counting_mae = float(np.mean(counting_errors)) if counting_errors else -1.0
    pipeline_fps = n_images / max(t_total, 0.001)

    # Quality gate check at the pipeline level (industrial target)
    pipeline_precision_gate = (
        "PASS" if pipeline_precision >= PRECISION_GATE
        else f"FAIL({pipeline_precision:.2f}<{PRECISION_GATE})"
    )
    pipeline_recall_gate = (
        "PASS" if pipeline_recall >= RECALL_GATE
        else f"FAIL({pipeline_recall:.2f}<{RECALL_GATE})"
    )
    pipeline_target_met = (
        pipeline_precision >= PRECISION_GATE and pipeline_recall >= RECALL_GATE
    )

    metrics = {
        "pipeline_precision": round(pipeline_precision, 4),
        "pipeline_recall": round(pipeline_recall, 4),
        "pipeline_f1": round(pipeline_f1, 4),
        "pipeline_counting_acc": round(pipeline_counting_acc, 4) if pipeline_counting_acc >= 0 else -1,
        "pipeline_counting_mae": round(pipeline_counting_mae, 2) if pipeline_counting_mae >= 0 else -1,
        "pipeline_fps": round(pipeline_fps, 2),
        "pipeline_total_sec": round(t_total, 1),
        "pipeline_images": n_images,
        "pipeline_images_with_gt": n_with_gt,
        "pipeline_total_tp": total_tp,
        "pipeline_total_fp": total_fp,
        "pipeline_total_fn": total_fn,
        "pipeline_precision_gate": pipeline_precision_gate,
        "pipeline_recall_gate": pipeline_recall_gate,
        "pipeline_target_met": pipeline_target_met,
        "match_iou": match_iou,
        "nms_iou": nms_iou,
        "details": results_all,
    }

    print_pipeline_metrics(metrics)

    # Persist to JSON for orchestrator parsing
    try:
        with open(PIPELINE_METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except OSError as e:
        print(
            f"evaluate: failed to write {PIPELINE_METRICS_JSON_PATH}: {e}",
            file=sys.stderr,
        )

    return metrics


def print_pipeline_metrics(metrics):
    """Grep-friendly pipeline metrics output (parallel to print_metrics)."""
    print("---pipeline---")
    print(f"pipeline_images:           {metrics['pipeline_images']}")
    print(f"pipeline_images_with_gt:   {metrics['pipeline_images_with_gt']}")
    print(f"pipeline_precision:        {metrics['pipeline_precision']:.4f}")
    print(f"pipeline_recall:           {metrics['pipeline_recall']:.4f}")
    print(f"pipeline_f1:               {metrics['pipeline_f1']:.4f}")
    if metrics['pipeline_counting_acc'] >= 0:
        print(f"pipeline_counting_acc:     {metrics['pipeline_counting_acc']:.4f}")
        print(f"pipeline_counting_mae:     {metrics['pipeline_counting_mae']:.2f}")
    else:
        print("pipeline_counting_acc:     N/A (no GT)")
    print(f"pipeline_fps:              {metrics['pipeline_fps']:.2f}")
    print(f"pipeline_total_sec:        {metrics['pipeline_total_sec']:.1f}")
    print(f"pipeline_total_tp:         {metrics['pipeline_total_tp']}")
    print(f"pipeline_total_fp:         {metrics['pipeline_total_fp']}")
    print(f"pipeline_total_fn:         {metrics['pipeline_total_fn']}")
    print(f"pipeline_precision_gate:   {metrics['pipeline_precision_gate']}")
    print(f"pipeline_recall_gate:      {metrics['pipeline_recall_gate']}")
    print(f"pipeline_target_met:       {metrics['pipeline_target_met']}")
    for r in metrics.get("details", []):
        if r["gt_count"] >= 0:
            print(
                f"  {r['image']}: pred={r['pred_count']} gt={r['gt_count']} "
                f"tp={r['tp']} fp={r['fp']} fn={r['fn']} acc={r['counting_accuracy']:.2f}"
            )
        else:
            print(f"  {r['image']}: pred={r['pred_count']} (no GT)")


def resolve_pipeline_dir(cli_value=None, repo_root=None):
    """Decide which pipeline_val/ directory to use, or None to skip.

    Precedence: CLI arg > AUTORESEARCH_PIPELINE_DIR env var > pipeline_val/ at
    repo root if it exists. AUTORESEARCH_DISABLE_PIPELINE=1 disables completely.
    """
    if os.environ.get("AUTORESEARCH_DISABLE_PIPELINE", "").strip() == "1":
        return None
    if cli_value:
        p = Path(cli_value)
        return p if p.is_dir() else None
    raw = os.environ.get("AUTORESEARCH_PIPELINE_DIR", "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_dir() else None
    # Auto-detect at repo root
    base = Path(repo_root) if repo_root else Path(__file__).resolve().parent
    candidate = base / "pipeline_val"
    return candidate if candidate.is_dir() else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--data", default="person_dataset/person.yaml", help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--pipeline-dir",
        default=None,
        help="Large images dir for pipeline eval. Falls back to "
             "AUTORESEARCH_PIPELINE_DIR env var, then auto-detects pipeline_val/ at repo root. "
             "Set AUTORESEARCH_DISABLE_PIPELINE=1 to skip.",
    )
    parser.add_argument(
        "--pipeline-match-iou",
        type=float,
        default=PIPELINE_MATCH_IOU_DEFAULT,
        help="IoU threshold for pipeline-level prediction-to-GT matching (default 0.5).",
    )
    parser.add_argument(
        "--export-fpfn",
        default=None,
        metavar="DIR",
        help="Write per-image FP/FN JSON + top-K crops to DIR (for v2 Data Curator agent). "
             "Falls back to AUTORESEARCH_FPFN_DIR env var. Default: disabled.",
    )
    parser.add_argument(
        "--fpfn-topk",
        type=int,
        default=None,
        help="Max crops per image per category (FP and FN). Falls back to "
             "AUTORESEARCH_FPFN_TOPK env var. Default: 10.",
    )
    parser.add_argument(
        "--fpfn-iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to GT (default 0.5).",
    )
    args = parser.parse_args()

    fpfn_dir = _resolve_fpfn_dir(args.export_fpfn)
    fpfn_topk = _resolve_fpfn_topk(args.fpfn_topk)

    metrics = evaluate_model(
        args.model,
        args.data,
        args.imgsz,
        fpfn_dir=fpfn_dir,
        fpfn_topk=fpfn_topk,
        fpfn_iou=args.fpfn_iou,
    )
    print_metrics(metrics)

    pipeline_dir = resolve_pipeline_dir(args.pipeline_dir)
    if pipeline_dir is not None:
        evaluate_pipeline(
            args.model,
            str(pipeline_dir),
            args.imgsz,
            match_iou=args.pipeline_match_iou,
        )
    else:
        print("evaluate: pipeline eval skipped (no pipeline_val/ found and not configured)")
