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
# Default deployment thresholds frozen from threshold_sweep.tsv (GPU 5070 baseline).
DEPLOY_CONF = 0.20
DEPLOY_IOU_NMS = 0.30

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


def get_deploy_conf():
    """Return deployment confidence threshold (supports env override)."""
    raw = os.environ.get("AUTORESEARCH_DEPLOY_CONF", "").strip()
    if raw:
        try:
            v = float(raw)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            print("evaluate: invalid AUTORESEARCH_DEPLOY_CONF, using default", file=sys.stderr)
    return DEPLOY_CONF


def get_deploy_iou_nms():
    """Return deployment NMS IoU threshold (supports env override)."""
    raw = os.environ.get("AUTORESEARCH_DEPLOY_IOU_NMS", "").strip()
    if raw:
        try:
            v = float(raw)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            print("evaluate: invalid AUTORESEARCH_DEPLOY_IOU_NMS, using default", file=sys.stderr)
    return DEPLOY_IOU_NMS


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


def _evaluate_on_split(model, data_dir, split_name, data_cfg, imgsz, deploy_conf, deploy_iou_nms):
    """Evaluate model on a specific dataset split (train/val/test)."""
    img_rel = data_cfg.get(split_name, f"images/{split_name}")
    if os.path.isabs(img_rel):
        img_dir = Path(img_rel)
    else:
        img_dir = data_dir / img_rel

    label_dir = Path(str(img_dir).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))

    images = sorted(glob.glob(str(img_dir / "*.JPG")) +
                    glob.glob(str(img_dir / "*.jpg")) +
                    glob.glob(str(img_dir / "*.png")))

    if not images:
        print(f"evaluate: no {split_name} images found in {img_dir}")
        return None

    # 无标签的 split (如 test 缺全部标签) 必须跳过: 否则每张图 GT=0,
    # 模型有检出 → acc=0/recall=0 的垃圾指标被当真 → val_test_gap 虚高
    # → 每轮都误报"过拟合", 还白白多跑几百张图的推理。
    n_label_files = sum(
        1 for img_path in images
        if (label_dir / f"{Path(img_path).stem}.txt").exists()
    )
    if n_label_files == 0:
        print(f"evaluate: {split_name} split has {len(images)} images but "
              f"0 label files in {label_dir} — skipping unlabeled split")
        return None

    per_image_accs = []
    counting_errors = []
    small_gt_total = 0
    small_gt_matched = 0
    pred_boxes_all = []
    gt_boxes_all = []

    print(f"evaluate: running {split_name} evaluation on {len(images)} images...")
    for i, img_path in enumerate(images):
        img_name = Path(img_path).stem
        label_path = label_dir / f"{img_name}.txt"

        gt_boxes, gt_areas = load_gt_boxes(str(label_path), imgsz, imgsz)
        gt_count = len(gt_boxes)

        results = model(
            img_path,
            conf=deploy_conf,
            iou=deploy_iou_nms,
            imgsz=imgsz,
            device=0,
            verbose=False,
        )
        res = results[0]

        if res.boxes is not None and len(res.boxes) > 0:
            pred_boxes = res.boxes.xyxy.cpu().numpy().tolist()
            pred_count = len(pred_boxes)
        else:
            pred_boxes = []
            pred_count = 0

        if gt_count > 0:
            acc = max(0, 1 - abs(pred_count - gt_count) / gt_count)
        else:
            acc = 1.0 if pred_count == 0 else 0.0
        per_image_accs.append(acc)
        counting_errors.append(abs(pred_count - gt_count))

        for gi, area in enumerate(gt_areas):
            if area < SMALL_OBJ_AREA_THRESH:
                small_gt_total += 1
                best_iou = 0
                for pb in pred_boxes:
                    iou = compute_iou(pb, gt_boxes[gi])
                    best_iou = max(best_iou, iou)
                if best_iou >= 0.5:
                    small_gt_matched += 1

        pred_boxes_all.extend(pred_boxes)
        gt_boxes_all.extend(gt_boxes)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(images)} {split_name} images...")

    counting_accuracy = float(np.mean(per_image_accs)) if per_image_accs else 0.0
    counting_mae = float(np.mean(counting_errors)) if counting_errors else 0.0
    small_obj_recall = small_gt_matched / max(small_gt_total, 1)

    matched = match_predictions_to_gt(pred_boxes_all, gt_boxes_all)
    precision_split = matched / max(len(pred_boxes_all), 1)
    recall_split = matched / max(len(gt_boxes_all), 1)

    return {
        "counting_acc": counting_accuracy,
        "counting_mae": counting_mae,
        "small_obj_recall": small_obj_recall,
        "precision": precision_split,
        "recall": recall_split,
        "n_images": len(images),
        "n_gt_boxes": len(gt_boxes_all),
        "n_pred_boxes": len(pred_boxes_all),
    }


def evaluate_model(model_path, data_yaml, imgsz=1280):
    """Main evaluation function. Returns dict of all metrics.

    This is the single source of truth for the autoresearch loop.
    Now includes test set evaluation to prevent overfitting.
    """
    # ── Watchdog: hard-kill if anything below hangs ──
    _watchdog = _start_eval_watchdog()

    # ── Resolve deployment thresholds for this run ──
    deploy_conf = get_deploy_conf()
    deploy_iou_nms = get_deploy_iou_nms()

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

    # ── Load data config ──
    data_dir = Path(data_yaml).parent
    import yaml
    with open(data_yaml, "r", encoding="utf-8", errors="replace") as f:
        data_cfg = yaml.safe_load(f)

    # ── Evaluate on val set (for training loop) ──
    val_metrics = _evaluate_on_split(model, data_dir, "val", data_cfg, imgsz, deploy_conf, deploy_iou_nms)

    # ── Evaluate on test set (for overfitting detection) ──
    test_metrics = _evaluate_on_split(model, data_dir, "test", data_cfg, imgsz, deploy_conf, deploy_iou_nms)

    # Use val metrics for primary evaluation
    counting_accuracy = val_metrics["counting_acc"] if val_metrics else 0.0
    counting_mae = val_metrics["counting_mae"] if val_metrics else 0.0
    small_obj_recall = val_metrics["small_obj_recall"] if val_metrics else 0.0

    # ── Inference timing ──
    val_img_rel = data_cfg.get("val", "images/val")
    val_img_dir = Path(val_img_rel) if os.path.isabs(val_img_rel) else data_dir / val_img_rel
    val_images = sorted(glob.glob(str(val_img_dir / "*.JPG")) +
                        glob.glob(str(val_img_dir / "*.jpg")) +
                        glob.glob(str(val_img_dir / "*.png")))

    timing_imgs = val_images[:WARMUP_IMAGES + TIMING_IMAGES]
    for img in timing_imgs[:WARMUP_IMAGES]:
        model(img, conf=deploy_conf, iou=deploy_iou_nms, imgsz=imgsz,
              device=0, verbose=False)
    times = []
    for img in timing_imgs[WARMUP_IMAGES:]:
        t0 = time.time()
        model(img, conf=deploy_conf, iou=deploy_iou_nms, imgsz=imgsz,
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
    for img_path in val_images[:50]:
        results = model(img_path, conf=deploy_conf, iou=deploy_iou_nms,
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

    # ── Overfitting detection ──
    overfitting_detected = False
    val_test_gap = 0.0
    if val_metrics and test_metrics:
        val_cds = (W_MAP50 * mAP50 +
                   W_MAP50_95 * mAP50_95 +
                   W_F1 * (2 * val_metrics["precision"] * val_metrics["recall"] / 
                           (val_metrics["precision"] + val_metrics["recall"] + 1e-6)) +
                   W_COUNTING * val_metrics["counting_acc"] +
                   W_SMALL_OBJ * val_metrics["small_obj_recall"])
        
        test_cds = (W_MAP50 * mAP50 +
                    W_MAP50_95 * mAP50_95 +
                    W_F1 * (2 * test_metrics["precision"] * test_metrics["recall"] / 
                            (test_metrics["precision"] + test_metrics["recall"] + 1e-6)) +
                    W_COUNTING * test_metrics["counting_acc"] +
                    W_SMALL_OBJ * test_metrics["small_obj_recall"])
        
        val_test_gap = val_cds - test_cds
        
        if val_test_gap > 0.05:
            overfitting_detected = True
            print(f"evaluate: WARNING - Overfitting detected! Val CDS - Test CDS = {val_test_gap:.4f}")

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
        "deploy_conf": round(deploy_conf, 4),
        "deploy_iou_nms": round(deploy_iou_nms, 4),
        "precision_gate": precision_gate,
        "recall_gate": recall_gate,
        "latency_gate": latency_gate,
        "latency_gate_ms": round(gate_ms, 1),
        "target_met": target_met,
    }

    # Test set metrics — 只在 test split 真有标签时输出。无标签时整组键
    # 省略 (不是置 None): orchestrator 按 `"test_counting_acc" in metrics`
    # 判断, None 值会让它格式化时崩溃; val_test_gap/overfitting_detected
    # 缺省时下游按 0.0/False 处理。
    if test_metrics:
        metrics.update({
            "test_counting_acc": round(test_metrics["counting_acc"], 4),
            "test_counting_mae": round(test_metrics["counting_mae"], 4),
            "test_small_obj_recall": round(test_metrics["small_obj_recall"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "test_n_images": test_metrics["n_images"],
            "val_test_gap": round(val_test_gap, 4),
            "overfitting_detected": overfitting_detected,
        })

    # ── Persist metrics to JSON for robust downstream parsing ──
    try:
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except OSError as e:
        print(f"evaluate: failed to write {METRICS_JSON_PATH}: {e}", file=sys.stderr)

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
    print(f"deploy_conf:      {metrics.get('deploy_conf', DEPLOY_CONF)}")
    print(f"deploy_iou_nms:   {metrics.get('deploy_iou_nms', DEPLOY_IOU_NMS)}")
    print(f"epochs_completed: {epochs_completed}")
    print(f"precision_gate:   {metrics['precision_gate']}")
    print(f"recall_gate:      {metrics['recall_gate']}")
    print(f"latency_gate:     {metrics['latency_gate']}")
    print(f"latency_gate_ms:  {metrics['latency_gate_ms']}")
    # Test set metrics
    if metrics.get("test_counting_acc") is not None:
        print("---test---")
        print(f"test_counting_acc:      {metrics['test_counting_acc']:.4f}")
        print(f"test_counting_mae:      {metrics['test_counting_mae']:.4f}")
        print(f"test_small_obj_recall:  {metrics['test_small_obj_recall']:.4f}")
        print(f"test_precision:         {metrics['test_precision']:.4f}")
        print(f"test_recall:            {metrics['test_recall']:.4f}")
        print(f"test_n_images:          {metrics['test_n_images']}")
        print(f"val_test_gap:           {metrics['val_test_gap']:.4f}")
        print(f"overfitting_detected:   {metrics['overfitting_detected']}")


def evaluate_pipeline(model_path, large_images_dir, imgsz=1280, overlap=200, nms_iou=0.35):
    """Large-image pipeline evaluation.

    Simulates deployment: large image → tiles → per-tile detection →
    coordinate restoration → global NMS → full-image count.

    Run every 5 keeps.
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

        # Global NMS
        if all_boxes:
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
            scores_t = torch.tensor(all_scores, dtype=torch.float32)
            keep_idx = torch_nms(boxes_t, scores_t, nms_iou)
            final_count = len(keep_idx)
        else:
            final_count = 0

        # Load GT for this large image if available
        gt_label = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(gt_label):
            gt_boxes, _ = load_gt_boxes(gt_label, W, H)
            gt_count = len(gt_boxes)
            acc = max(0, 1 - abs(final_count - gt_count) / max(gt_count, 1))
        else:
            gt_count = -1
            acc = -1

        results_all.append({
            "image": os.path.basename(img_path),
            "pred_count": final_count,
            "gt_count": gt_count,
            "accuracy": acc,
        })

    t_total = time.time() - t_start

    # Aggregate
    valid = [r for r in results_all if r["gt_count"] > 0]
    global_counting_acc = float(np.mean([r["accuracy"] for r in valid])) if valid else -1
    pipeline_fps = len(large_images) / max(t_total, 0.001)

    print("---pipeline---")
    print(f"pipeline_images:       {len(large_images)}")
    print(f"global_counting_acc:   {global_counting_acc:.4f}" if global_counting_acc >= 0 else "global_counting_acc:   N/A (no GT)")
    print(f"pipeline_fps:          {pipeline_fps:.2f}")
    print(f"pipeline_total_sec:    {t_total:.1f}")
    for r in results_all:
        print(f"  {r['image']}: pred={r['pred_count']} gt={r['gt_count']} acc={r['accuracy']:.2f}" if r['gt_count'] > 0 else f"  {r['image']}: pred={r['pred_count']} (no GT)")

    return {
        "global_counting_acc": global_counting_acc,
        "pipeline_fps": pipeline_fps,
        "details": results_all,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--data", default="person_dataset/person.yaml", help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--pipeline-dir", default=None, help="Large images dir for pipeline eval")
    args = parser.parse_args()

    metrics = evaluate_model(args.model, args.data, args.imgsz)
    print_metrics(metrics)

    if args.pipeline_dir:
        evaluate_pipeline(args.model, args.pipeline_dir, args.imgsz)
