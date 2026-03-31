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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ==================== Model Config ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
IMG_SIZE = 1280
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASSES = ["person"]
PORT = 5000

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

        input_img, scale, pad_w, pad_h, orig_w, orig_h = preprocess_image(img)

        if model_session:
            outputs = model_session.run(output_names, {input_name: input_img})
            boxes = postprocess_output(outputs, scale, pad_w, pad_h, orig_w, orig_h)

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


# ==================== Auto Open Browser ====================
def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")


# ==================== Main ====================
if __name__ == '__main__':
    print("=" * 50)
    print("  Person Detection Server (Windows)")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  URL:    http://localhost:{PORT}")
    print("=" * 50)

    # Auto open browser after 1.5s
    threading.Timer(1.5, open_browser).start()

    app.run(
        debug=False,
        host='0.0.0.0',
        port=PORT,
        threaded=True
    )
