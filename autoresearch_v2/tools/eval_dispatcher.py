"""
eval_dispatcher.py — 包装 evaluate.py + FP/FN 提取

职责:
  - 运行 evaluate_model 评估模型
  - 提取 False Positive / False Negative 图片裁剪
  - 将指标写入状态库
  - 为 Curator 准备 FP/FN 分析数据
"""

import json
import os
import sys
import glob
import shutil
from pathlib import Path
from typing import Optional

EVAL_SCRIPT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."
))
METRICS_FILE = os.path.join(EVAL_SCRIPT_DIR, "last_metrics.json")
FP_FN_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "state", "fp_fn_crops"
))


def run_evaluate(model_path: str, data_yaml: str, imgsz: int = 1280) -> Optional[dict]:
    sys.path.insert(0, EVAL_SCRIPT_DIR)
    try:
        from evaluate import evaluate_model
        metrics = evaluate_model(model_path, data_yaml, imgsz)
        return metrics
    except Exception as e:
        print(f"eval_dispatcher: evaluation failed: {e}", file=sys.stderr)
        return None
    finally:
        while EVAL_SCRIPT_DIR in sys.path:
            sys.path.remove(EVAL_SCRIPT_DIR)


def _padded_crop_box(box, img_w, img_h, pad_frac=0.2, min_pad_px=12):
    """Inflate a [x1,y1,x2,y2] box by pad_frac (or min_pad_px, whichever is
    larger) on each side, clamped to image bounds. Returns None for
    degenerate boxes."""
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = max(min_pad_px, w * pad_frac)
    py = max(min_pad_px, h * pad_frac)
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(img_w, int(round(x2 + px)))
    cy2 = min(img_h, int(round(y2 + py)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def extract_fp_fn_crops(model_path: str, data_yaml: str, imgsz: int = 1280,
                        output_dir: str = FP_FN_DIR, top_k: int = 20,
                        max_per_image: int = 2,
                        events_logger=None,
                        run_id: str = "",
                        keep_history: bool = True,
                        progress_every: int = 100) -> dict:
    """Run inference on val images, extract top-K FP and FN, save padded crops.

    Improvements vs the original:
      - Captures pred_conf so FPs are ranked by confidence (the "most
        damaging" wrong predictions) instead of by image-level gt_count.
      - FP crops have the same padding treatment as FN crops (Curator
        needs context to judge what was misclassified).
      - Per-image cap (max_per_image) prevents one bad image from
        hogging the top-K slot budget.
      - Per-run subdirectory when run_id is provided + keep_history=True
        so cross-run comparison is possible.
      - Emits eval_extract_start / eval_extract_progress (every
        progress_every images) / eval_extract_complete events so the
        operator's terminal isn't dark for the duration.
    """
    sys.path.insert(0, EVAL_SCRIPT_DIR)
    try:
        import torch
        import numpy as np
        from ultralytics import YOLO
        from evaluate import load_gt_boxes, compute_iou, get_deploy_conf, get_deploy_iou_nms
        import yaml
    except ImportError as e:
        print(f"eval_dispatcher: missing dependency: {e}", file=sys.stderr, flush=True)
        if events_logger:
            try:
                events_logger.emit("eval_extract_failed",
                                    run_id=run_id, error=str(e))
            except Exception:
                pass
        return {"fp_images": [], "fn_images": []}

    # Resolve output dir (per-run subdir for history-friendly layout).
    if run_id and keep_history:
        output_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Clean only the target dir's prior crops (keeps sibling per-run dirs).
    for f in glob.glob(os.path.join(output_dir, "fp_*.jpg")):
        try:
            os.remove(f)
        except OSError:
            pass
    for f in glob.glob(os.path.join(output_dir, "fn_*.jpg")):
        try:
            os.remove(f)
        except OSError:
            pass

    model = YOLO(model_path)
    deploy_conf = get_deploy_conf()
    deploy_iou = get_deploy_iou_nms()

    with open(data_yaml, "r", encoding="utf-8", errors="replace") as f:
        data_cfg = yaml.safe_load(f)

    data_dir = Path(data_yaml).parent
    val_img_rel = data_cfg.get("val", "images/val")
    if os.path.isabs(val_img_rel):
        val_img_dir = Path(val_img_rel)
    else:
        val_img_dir = data_dir / val_img_rel

    val_label_dir = Path(str(val_img_dir).replace(
        os.sep + "images" + os.sep,
        os.sep + "labels" + os.sep
    ))

    val_images = sorted(
        glob.glob(str(val_img_dir / "*.JPG")) +
        glob.glob(str(val_img_dir / "*.jpg")) +
        glob.glob(str(val_img_dir / "*.png"))
    )

    if events_logger:
        try:
            events_logger.emit(
                "eval_extract_start",
                run_id=run_id,
                n_val_images=len(val_images),
                top_k=top_k,
                max_per_image=max_per_image,
                output_dir=output_dir,
            )
        except Exception:
            pass

    fp_records = []
    fn_records = []

    for idx, img_path in enumerate(val_images):
        img_name = Path(img_path).stem
        label_path = val_label_dir / f"{img_name}.txt"

        gt_boxes, gt_areas = load_gt_boxes(str(label_path), imgsz, imgsz)
        gt_count = len(gt_boxes)

        results = model(img_path, conf=deploy_conf, iou=deploy_iou,
                        imgsz=imgsz, device=0, verbose=False)
        res = results[0]

        pred_boxes = []
        pred_scores: list = []
        if res.boxes is not None and len(res.boxes) > 0:
            pred_boxes = res.boxes.xyxy.cpu().numpy().tolist()
            pred_scores = res.boxes.conf.cpu().numpy().tolist()

        matched_gt = set()
        # Visit predictions in descending confidence order so the
        # highest-confidence prediction claims its best-IoU GT first.
        order = sorted(range(len(pred_boxes)),
                        key=lambda i: pred_scores[i] if i < len(pred_scores) else 0.0,
                        reverse=True)
        for pi in order:
            pb = pred_boxes[pi]
            pconf = float(pred_scores[pi]) if pi < len(pred_scores) else 0.0
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched_gt.add(best_gi)
            else:
                fp_records.append({
                    "image": img_name,
                    "image_path": img_path,
                    "pred_box": pb,
                    "pred_conf": pconf,
                    "gt_count": gt_count,
                    "pred_count": len(pred_boxes),
                })

        for gi, gb in enumerate(gt_boxes):
            if gi not in matched_gt:
                fn_records.append({
                    "image": img_name,
                    "image_path": img_path,
                    "gt_box": gb,
                    "gt_area": gt_areas[gi] if gi < len(gt_areas) else 0,
                    "gt_count": gt_count,
                    "pred_count": len(pred_boxes),
                })

        if events_logger and progress_every > 0 and (idx + 1) % progress_every == 0:
            try:
                events_logger.emit(
                    "eval_extract_progress",
                    run_id=run_id,
                    images_done=idx + 1,
                    images_total=len(val_images),
                    fp_so_far=len(fp_records),
                    fn_so_far=len(fn_records),
                )
            except Exception:
                pass

    # Sort: FPs by confidence DESC (most damaging first), FNs by area ASC
    # (smallest first — smallest objects are the hardest case in dense crowds).
    fp_records.sort(key=lambda x: x.get("pred_conf", 0.0), reverse=True)
    fn_records.sort(key=lambda x: x.get("gt_area", 0) or 1.0)

    # Per-image dedup so one bad image doesn't flood top-K.
    def _spread(records, max_per: int, k: int):
        per_image_count: dict = {}
        out = []
        for rec in records:
            img = rec.get("image", "")
            if per_image_count.get(img, 0) >= max_per:
                continue
            out.append(rec)
            per_image_count[img] = per_image_count.get(img, 0) + 1
            if len(out) >= k:
                break
        return out

    fp_records = _spread(fp_records, max_per_image, top_k)
    fn_records = _spread(fn_records, max_per_image, top_k)

    from PIL import Image

    fp_saved = 0
    fn_saved = 0

    for i, rec in enumerate(fp_records):
        try:
            img = Image.open(rec["image_path"])
            crop_box = _padded_crop_box(rec["pred_box"], img.width, img.height,
                                         pad_frac=0.2, min_pad_px=12)
            if crop_box is None:
                continue
            crop = img.crop(crop_box)
            crop_path = os.path.join(
                output_dir,
                f"fp_{i:03d}_{rec['image']}.jpg",
            )
            crop.save(crop_path, format="JPEG", quality=85)
            rec["crop_path"] = crop_path
            fp_saved += 1
        except Exception as e:
            print(f"eval_dispatcher: FP crop failed for {rec.get('image')}: {e}",
                  file=sys.stderr, flush=True)

    for i, rec in enumerate(fn_records):
        try:
            img = Image.open(rec["image_path"])
            crop_box = _padded_crop_box(rec["gt_box"], img.width, img.height,
                                         pad_frac=0.2, min_pad_px=20)
            if crop_box is None:
                continue
            crop = img.crop(crop_box)
            crop_path = os.path.join(
                output_dir,
                f"fn_{i:03d}_{rec['image']}.jpg",
            )
            crop.save(crop_path, format="JPEG", quality=85)
            rec["crop_path"] = crop_path
            fn_saved += 1
        except Exception as e:
            print(f"eval_dispatcher: FN crop failed for {rec.get('image')}: {e}",
                  file=sys.stderr, flush=True)

    if events_logger:
        try:
            events_logger.emit(
                "eval_extract_complete",
                run_id=run_id,
                n_fp=len(fp_records),
                n_fn=len(fn_records),
                fp_saved=fp_saved,
                fn_saved=fn_saved,
                output_dir=output_dir,
                top_fp_conf=fp_records[0]["pred_conf"] if fp_records else 0.0,
            )
        except Exception:
            pass

    return {
        "fp_images": fp_records,
        "fn_images": fn_records,
        "output_dir": output_dir,
    }


class EvalDispatcher:
    def __init__(self, state=None, events_logger=None, data_yaml: str = "",
                 imgsz: int = 1280):
        self.state = state
        self.events = events_logger
        self.data_yaml = data_yaml
        self.imgsz = imgsz

    def run(self, model_path: str, run_id: str = "",
            extract_fp_fn: bool = True) -> dict:
        if self.events:
            try:
                self.events.emit("eval_start", run_id=run_id, model=model_path)
            except Exception:
                pass

        metrics = run_evaluate(model_path, self.data_yaml, self.imgsz)

        if metrics is None:
            if self.state and run_id:
                self.state.update_experiment_status(run_id, "eval_failed")
            if self.events:
                try:
                    self.events.emit("eval_failed", run_id=run_id)
                except Exception:
                    pass
            return {}

        if self.state and run_id:
            self.state.update_experiment_metrics(run_id, metrics)

        fp_fn_data = {}
        if extract_fp_fn and model_path:
            fp_fn_data = extract_fp_fn_crops(
                model_path, self.data_yaml, self.imgsz,
                events_logger=self.events,
                run_id=run_id,
            )

        return {"metrics": metrics, "fp_fn_data": fp_fn_data}
