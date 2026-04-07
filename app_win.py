# -*- coding: utf-8 -*-
"""
Person Detection Flask Server - Windows Version
Usage:
    1. Put best.onnx and templates/ folder in the same directory as this file
    2. pip install flask pandas openpyxl onnxruntime opencv-python Pillow
    3. python app_win.py
    4. Open browser: http://localhost:5000
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import json
import numpy as np
import pandas as pd
import io
import base64
from PIL import Image
import onnxruntime as ort
from datetime import datetime
import os
import sys
import traceback
import warnings
import webbrowser
import threading

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 大图切块；可按机器再调

# ==================== Model Config ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
IMG_SIZE = 1280
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASSES = ["person"]
PORT = 5000
# 跨 tile 合并时的 NMS（与 evaluate.py 部署侧 0.35 对齐，可表单覆盖）
GLOBAL_MERGE_IOU = 0.35

# ==================== Load ONNX Model ====================
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] {MODEL_PATH} not found!")
    print(f"Please put best.onnx in: {BASE_DIR}")
    sys.exit(1)

# GPU first, auto fallback to CPU
available_providers = ort.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("[INFO] Using GPU (CUDA) for inference")
else:
    providers = ["CPUExecutionProvider"]
    print("[INFO] Using CPU for inference (install onnxruntime-gpu for GPU support)")

model_session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = model_session.get_inputs()[0].name
output_names = [o.name for o in model_session.get_outputs()]
print(f"[OK] ONNX model loaded, input shape: {model_session.get_inputs()[0].shape}")


# ==================== Letterbox Preprocessing ====================
def preprocess_image(image):
    h, w = image.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized_img = cv2.resize(image, (new_w, new_h))

    pad_w = (IMG_SIZE - new_w) / 2
    pad_h = (IMG_SIZE - new_h) / 2
    top, bottom = int(np.floor(pad_h)), int(np.ceil(pad_h))
    left, right = int(np.floor(pad_w)), int(np.ceil(pad_w))
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    input_img = np.transpose(padded_img, (2, 0, 1)) / 255.0
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
    return input_img, scale, pad_w, pad_h, w, h


# ==================== Utilities ====================
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        current = idxs[0]
        keep.append(current)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_current + area_rest - inter
        iou = inter / (union + 1e-6)
        idxs = idxs[1:][iou <= iou_threshold]
    return keep


# ==================== Postprocess ====================
def postprocess_output(outputs, scale, pad_w, pad_h, orig_w, orig_h,
                       conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """
    YOLOv8 ONNX output: [1, 5, 33600] -> transpose to [33600, 5]
    Each row: [x_center, y_center, width, height, score] (single class nc=1)
    """
    detections = []
    preds = outputs[0]

    if preds.ndim == 3:
        preds = preds[0]
        preds = np.transpose(preds)

    boxes_xywh = preds[:, :4]
    scores = preds[:, 4]

    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    if boxes_xywh.size == 0:
        return detections

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    keep_indices = nms(boxes_xyxy, scores, iou_threshold)

    for idx in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        detections.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'conf': float(scores[idx]),
            'class': 0
        })

    return detections


# ==================== Core inference (BGR array, tile-local coords) ====================
def run_inference_on_bgr(img_bgr):
    """Run ONNX on one BGR image (any size); returns list of person dicts in pixel coords of this image."""
    input_img, scale, pad_w, pad_h, orig_w, orig_h = preprocess_image(img_bgr)
    outputs = model_session.run(output_names, {input_name: input_img})
    return postprocess_output(outputs, scale, pad_w, pad_h, orig_w, orig_h)


def iter_sliding_tiles(img_bgr, tile_size, overlap):
    """Yield (x0, y0, tile_bgr) for sliding windows. stride = tile_size - overlap."""
    h, w = img_bgr.shape[:2]
    overlap = max(0, min(int(overlap), tile_size - 1))
    stride = max(1, tile_size - overlap)
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            if x1 <= x0 or y1 <= y0:
                continue
            yield x0, y0, img_bgr[y0:y1, x0:x1]


def merge_tiles_global_nms(payload, merge_iou):
    """
    payload: dict with orig_w, orig_h, tiles[]. Each tile has x0,y0 and persons[] in tile-local xyxy.
    Returns (list of (box_np(4,), score), orig_w, orig_h).
    """
    orig_w = int(payload["orig_w"])
    orig_h = int(payload["orig_h"])
    all_boxes = []
    all_scores = []
    for t in payload.get("tiles", []):
        x0, y0 = int(t["x0"]), int(t["y0"])
        for p in t.get("persons", []):
            all_boxes.append(
                [
                    float(p["x1"]) + x0,
                    float(p["y1"]) + y0,
                    float(p["x2"]) + x0,
                    float(p["y2"]) + y0,
                ]
            )
            all_scores.append(float(p["conf"]))
    if not all_boxes:
        return [], orig_w, orig_h
    boxes = np.array(all_boxes, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    keep = nms(boxes, scores, float(merge_iou))
    merged = [(boxes[i].copy(), float(scores[i])) for i in keep]
    return merged, orig_w, orig_h


def merged_boxes_to_yolo_txt(merged_pairs, orig_w, orig_h):
    """YOLO normalized lines (class 0) for full image size orig_w x orig_h."""
    lines = []
    ow, oh = float(orig_w), float(orig_h)
    for b, _s in merged_pairs:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        cx = ((x1 + x2) / 2) / ow
        cy = ((y1 + y2) / 2) / oh
        nw = (x2 - x1) / ow
        nh = (y2 - y1) / oh
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines)


# ==================== Image to Base64 ====================
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


# ==================== Single Image Detection ====================
def detect_single_image(image_bytes, draw_boxes=True):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Failed to read image"

        if model_session:
            boxes = run_inference_on_bgr(img)

            if draw_boxes:
                for box in boxes:
                    cv2.rectangle(img,
                                  (box['x1'], box['y1']),
                                  (box['x2'], box['y2']),
                                  (0, 255, 0), 2)

                    label = f"person: {box['conf']:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                    cv2.rectangle(img,
                                  (box['x1'], box['y1'] - label_size[1] - 5),
                                  (box['x1'] + label_size[0], box['y1']),
                                  (0, 255, 0), -1)

                    cv2.putText(img, label,
                                (box['x1'], box['y1'] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            img_base64 = image_to_base64(img)

            return {
                'count': len(boxes),
                'persons': boxes,
                'image_data': img_base64
            }, None
        else:
            return None, "Model not loaded"

    except Exception as e:
        print(f"Detection error:\n{traceback.format_exc()}")
        return None, f"Detection failed: {str(e)}"


# ==================== Routes ====================
@app.route('/templates/<path:filename>')
def serve_templates(filename):
    return send_from_directory('templates', filename)


@app.route('/templates/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('templates/css', filename)


@app.route('/templates/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('templates/js', filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect_single', methods=['POST'])
def detect_single():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        result, error = detect_single_image(image_bytes)

        if error:
            return jsonify({'error': error}), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_batch', methods=['POST'])
def detect_batch():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'Empty file list'}), 400

    results = []

    for file in files:
        if file.filename:
            try:
                image_bytes = file.read()
                result, error = detect_single_image(image_bytes, draw_boxes=False)

                if error:
                    results.append({
                        'name': file.filename,
                        'error': error,
                        'count': 0,
                        'image_data': None,
                        'persons': []
                    })
                else:
                    results.append({
                        'name': file.filename,
                        'count': result['count'],
                        'image_data': result['image_data'],
                        'persons': result['persons']
                    })
            except Exception as e:
                results.append({
                    'name': file.filename,
                    'error': str(e),
                    'count': 0,
                    'image_data': None,
                    'persons': []
                })

    excel_data = []
    for r in results:
        excel_data.append({
            'Filename': r['name'],
            'Person Count': r['count'],
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Status': 'OK' if 'error' not in r else f"Failed: {r.get('error', 'Unknown')}"
        })

    df = pd.DataFrame(excel_data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection Results')

    output.seek(0)
    excel_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
    excel_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return jsonify({
        'results': results,
        'excel_data': excel_base64,
        'excel_filename': excel_filename
    })


# ==================== Pipeline A / B (大图滑窗 → 合并 + 全局 NMS) ====================
@app.route('/a')
def page_a():
    return render_template('page_a.html')


@app.route('/b')
def page_b():
    return render_template('page_b.html')


@app.route('/api/pipeline_a', methods=['POST'])
def api_pipeline_a():
    """上传大图 → 1280 滑窗 → 每块 ONNX；返回 JSON 供 pipeline_b。"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    try:
        overlap = int(request.form.get('overlap', 200))
    except ValueError:
        overlap = 200
    overlap = max(0, min(overlap, IMG_SIZE - 1))

    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        H, W = img.shape[:2]
        stride = max(1, IMG_SIZE - overlap)
        tiles_out = []
        sum_raw = 0
        idx = 0

        for x0, y0, tile in iter_sliding_tiles(img, IMG_SIZE, overlap):
            persons = run_inference_on_bgr(tile)
            th, tw = tile.shape[:2]
            sum_raw += len(persons)
            tiles_out.append({
                'index': idx,
                'x0': x0,
                'y0': y0,
                'w': tw,
                'h': th,
                'persons': persons,
                'count': len(persons),
            })
            idx += 1

        return jsonify({
            'orig_w': W,
            'orig_h': H,
            'tile_size': IMG_SIZE,
            'overlap': overlap,
            'stride': stride,
            'tiles': tiles_out,
            'sum_raw': sum_raw,
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline_b', methods=['POST'])
def api_pipeline_b():
    """接收 pipeline_a 的 JSON → 坐标还原 → 全局 NMS → yolo_txt / 人数 / CSV；可选同一张大图画框。"""
    merge_iou = GLOBAL_MERGE_IOU
    image_bytes = None
    data = None

    if request.is_json:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Invalid or empty JSON'}), 400
        merge_iou = float(data.get('merge_iou', GLOBAL_MERGE_IOU))
        b64 = data.get('image_base64')
        if b64:
            s = b64.split(',', 1)[-1] if isinstance(b64, str) else ''
            try:
                image_bytes = base64.b64decode(s)
            except Exception:
                return jsonify({'error': 'Invalid image_base64'}), 400
    else:
        raw = request.form.get('payload')
        if not raw:
            return jsonify({'error': 'Missing form field payload (JSON string)'}), 400
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON in payload: {e}'}), 400
        try:
            merge_iou = float(request.form.get('merge_iou', GLOBAL_MERGE_IOU))
        except ValueError:
            merge_iou = GLOBAL_MERGE_IOU
        if 'image' in request.files and request.files['image'].filename:
            image_bytes = request.files['image'].read()

    if not isinstance(data, dict) or 'tiles' not in data:
        return jsonify({'error': 'Payload must be an object with key tiles'}), 400

    try:
        merged, ow, oh = merge_tiles_global_nms(data, merge_iou)
        yolo_txt = merged_boxes_to_yolo_txt(merged, ow, oh)
        count = len(merged)

        out = {
            'count': count,
            'merged_count': count,
            'sum_raw_hint': data.get('sum_raw'),
            'yolo_txt': yolo_txt,
            'merge_iou': merge_iou,
            'orig_w': ow,
            'orig_h': oh,
        }

        df = pd.DataFrame([{
            'total_persons': count,
            'orig_w': ow,
            'orig_h': oh,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }])
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        out['csv_base64'] = base64.b64encode(
            csv_buf.getvalue().encode('utf-8')
        ).decode('utf-8')

        if image_bytes:
            nparr = np.frombuffer(image_bytes, np.uint8)
            vis = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if vis is None:
                out['image_error'] = 'Could not decode image for visualization'
            else:
                ih, iw = vis.shape[:2]
                for b, sc in merged:
                    x1 = int(np.clip(b[0], 0, iw - 1))
                    y1 = int(np.clip(b[1], 0, ih - 1))
                    x2 = int(np.clip(b[2], 0, iw - 1))
                    y2 = int(np.clip(b[3], 0, ih - 1))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis, f'{sc:.2f}', (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                    )
                out['image_data'] = image_to_base64(vis)

        return jsonify(out)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ==================== Auto Open Browser ====================
def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")


# ==================== Main ====================
if __name__ == '__main__':
    print("=" * 50)
    print("  Person Detection Server (Windows)")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  URL:    http://localhost:{PORT}")
    print(f"  Page A: http://localhost:{PORT}/a")
    print(f"  Page B: http://localhost:{PORT}/b")
    print("=" * 50)

    # Auto open browser after 1.5s
    threading.Timer(1.5, open_browser).start()

    app.run(
        debug=False,
        host='0.0.0.0',
        port=PORT,
        threaded=True
    )
