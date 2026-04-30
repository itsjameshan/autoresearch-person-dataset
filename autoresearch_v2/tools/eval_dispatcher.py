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
        if EVAL_SCRIPT_DIR in sys.path:
            sys.path.remove(EVAL_SCRIPT_DIR)


def extract_fp_fn_crops(model_path: str, data_yaml: str, imgsz: int = 1280,
                        output_dir: str = FP_FN_DIR, top_k: int = 20) -> dict:
    sys.path.insert(0, EVAL_SCRIPT_DIR)
    try:
        import torch
        import numpy as np
        from ultralytics import YOLO
        from evaluate import load_gt_boxes, compute_iou, get_deploy_conf, get_deploy_iou_nms
        import yaml
    except ImportError as e:
        print(f"eval_dispatcher: missing dependency: {e}", file=sys.stderr)
        return {"fp_images": [], "fn_images": []}

    os.makedirs(output_dir, exist_ok=True)

    for f in glob.glob(os.path.join(output_dir, "*.jpg")):
        os.remove(f)
    for f in glob.glob(os.path.join(output_dir, "*.JPG")):
        os.remove(f)

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

    fp_records = []
    fn_records = []

    for img_path in val_images:
        img_name = Path(img_path).stem
        label_path = val_label_dir / f"{img_name}.txt"

        gt_boxes, gt_areas = load_gt_boxes(str(label_path), imgsz, imgsz)
        gt_count = len(gt_boxes)

        results = model(img_path, conf=deploy_conf, iou=deploy_iou,
                        imgsz=imgsz, device=0, verbose=False)
        res = results[0]

        pred_boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            pred_boxes = res.boxes.xyxy.cpu().numpy().tolist()

        matched_gt = set()
        for pb in pred_boxes:
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

    fp_records.sort(key=lambda x: x["gt_count"], reverse=True)
    fn_records.sort(key=lambda x: x["gt_area"] if x["gt_area"] else 1)

    fp_records = fp_records[:top_k]
    fn_records = fn_records[:top_k]

    from PIL import Image

    for i, rec in enumerate(fp_records):
        try:
            img = Image.open(rec["image_path"])
            box = rec["pred_box"]
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2 = min(x2, img.width)
            y2 = min(y2, img.height)
            if x2 > x1 and y2 > y1:
                crop = img.crop((x1, y1, x2, y2))
                crop_path = os.path.join(output_dir, f"fp_{i:03d}_{rec['image']}.jpg")
                crop.save(crop_path)
        except Exception:
            pass

    for i, rec in enumerate(fn_records):
        try:
            img = Image.open(rec["image_path"])
            box = rec["gt_box"]
            margin = 30
            x1 = max(0, int(box[0]) - margin)
            y1 = max(0, int(box[1]) - margin)
            x2 = min(img.width, int(box[2]) + margin)
            y2 = min(img.height, int(box[3]) + margin)
            if x2 > x1 and y2 > y1:
                crop = img.crop((x1, y1, x2, y2))
                crop_path = os.path.join(output_dir, f"fn_{i:03d}_{rec['image']}.jpg")
                crop.save(crop_path)
        except Exception:
            pass

    return {"fp_images": fp_records, "fn_images": fn_records}


class EvalDispatcher:
    def __init__(self, state=None, data_yaml: str = "",
                 imgsz: int = 1280):
        self.state = state
        self.data_yaml = data_yaml
        self.imgsz = imgsz

    def run(self, model_path: str, run_id: str = "",
            extract_fp_fn: bool = True) -> dict:
        metrics = run_evaluate(model_path, self.data_yaml, self.imgsz)

        if metrics is None:
            if self.state and run_id:
                self.state.update_experiment_status(run_id, "eval_failed")
            return {}

        if self.state and run_id:
            self.state.update_experiment_metrics(run_id, metrics)

        fp_fn_data = {}
        if extract_fp_fn and model_path:
            fp_fn_data = extract_fp_fn_crops(
                model_path, self.data_yaml, self.imgsz
            )

        return {"metrics": metrics, "fp_fn_data": fp_fn_data}
