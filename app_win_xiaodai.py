# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
import os
import warnings
import webbrowser
import threading
import time
import glob
import json
import base64
import pandas as pd

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
app.config['SECRET_KEY'] = 'person_detect'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT_DIR = os.path.join(BASE_DIR, "onnx_data")
SAVE_ROOT_DIR = r"D:\pythonProject\person_dataset\result"

os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

model_session = None
input_name = None
output_names = None
current_model_name = "未加载"
IMG_SIZE = 1280
PORT = 5000

model_lock = threading.Lock()


# ==================== 原版不变：只加载 onnx ====================
def get_available_models():
    models = []
    if not os.path.exists(MODEL_ROOT_DIR):
        return models
    for f in os.listdir(MODEL_ROOT_DIR):
        if f.lower().endswith(".onnx"):
            models.append({"name": f, "path": os.path.join(MODEL_ROOT_DIR, f)})
    return models


def load_model(model_path):
    global model_session, input_name, output_names, current_model_name
    try:
        with model_lock:
            if model_session:
                del model_session
            providers = ["CPUExecutionProvider"]
            model_session = ort.InferenceSession(model_path, providers=providers)
            input_name = model_session.get_inputs()[0].name
            output_names = [o.name for o in model_session.get_outputs()]
            current_model_name = os.path.basename(model_path)
        return True, f"✅ 模型加载成功：{current_model_name}"
    except Exception as e:
        return False, f"❌ 加载失败：{str(e)}"


# ==================== ✅ 自动优先加载 yolo12l.onnx ====================
models = get_available_models()
target_model = "yolo12l.onnx"
loaded = False

# 优先找 yolo12l.onnx
for m in models:
    if m["name"] == target_model:
        ok, msg = load_model(m["path"])
        print(msg)
        loaded = True
        break

# 找不到就加载第一个
if not loaded and models:
    load_model(models[0]["path"])

if not models:
    print("⚠️ 警告: 未找到ONNX模型文件，请将模型放入 onnx_data 目录")


def preprocess(image):
    h, w = image.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w = (IMG_SIZE - new_w) / 2
    pad_h = (IMG_SIZE - new_h) / 2
    top = int(np.floor(pad_h))
    bottom = int(np.ceil(pad_h))
    left = int(np.floor(pad_w))
    right = int(np.ceil(pad_w))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, pad_w, pad_h, w, h


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def postprocess(outputs, scale, pad_w, pad_h, w, h, conf_thres, iou_thres):
    pred = outputs[0][0].T
    conf = pred[:, 4]
    keep = conf >= conf_thres
    boxes = pred[keep, :4]
    scores = pred[keep, 4]
    if len(boxes) == 0:
        return []

    boxes = xywh2xyxy(boxes)

    if len(boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    if indices is None or len(indices) == 0:
        return []

    if isinstance(indices, tuple):
        indices = indices[0]
    elif len(indices.shape) > 1:
        indices = indices.flatten()

    dets = []
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        x1 = max(0, min(int(x1), w))
        y1 = max(0, min(int(y1), h))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))

        dets.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "conf": round(float(scores[i]), 3),
            "class_name": "person"
        })
    return dets


def to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


def detect_img_bytes(img_bytes, conf=0.5, iou=0.5):
    global model_session

    if model_session is None:
        return None, "模型未加载"

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "图片无效"

    t0 = time.time()

    with model_lock:
        if model_session is None:
            return None, "模型未加载"
        blob, scale, pw, ph, w, h = preprocess(img)
        outs = model_session.run(output_names, {input_name: blob})
        persons = postprocess(outs, scale, pw, ph, w, h, conf, iou)

    for p in persons:
        cv2.rectangle(img, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 165, 255), 2)
        cv2.putText(img, f"person {p['conf']:.2f}", (p["x1"], p["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    return {
        "count": len(persons),
        "persons": persons,
        "image_data": to_base64(img),
        "infer_time": round(time.time() - t0, 3)
    }, None


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/get_models')
def get_models():
    return jsonify({
        "models": get_available_models(),
        "current": current_model_name
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    data = request.get_json()
    if not data or 'model_name' not in data:
        return jsonify({"ok": False, "msg": "缺少模型名称"})

    name = data.get("model_name")
    path = os.path.join(MODEL_ROOT_DIR, name)

    if not os.path.exists(path):
        return jsonify({"ok": False, "msg": f"模型文件不存在: {name}"})

    ok, msg = load_model(path)
    return jsonify({"ok": ok, "msg": msg})

@app.route('/detect_single', methods=['POST'])
def detect_single():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "未上传图片"})

        f = request.files['image']
        if f.filename == '':
            return jsonify({"error": "未选择文件"})

        conf = float(request.form.get('conf', 0.5))
        iou = float(request.form.get('iou', 0.5))

        conf = max(0, min(1, conf))
        iou = max(0, min(1, iou))

        res, err = detect_img_bytes(f.read(), conf, iou)

        if err:
            return jsonify({"error": err})
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": f"检测异常: {str(e)}"})


@app.route('/detect_folder', methods=['POST'])
def detect_folder():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效请求数据", "results": [], "total": 0, "total_count": 0})

        folder = data.get('folder_path', '').strip()
        conf = float(data.get('conf', 0.5))
        iou = float(data.get('iou', 0.5))

        conf = max(0, min(1, conf))
        iou = max(0, min(1, iou))

        if not folder:
            return jsonify({"error": "未提供文件夹路径", "results": [], "total": 0, "total_count": 0})

        if not os.path.isdir(folder):
            return jsonify({"error": f"文件夹不存在: {folder}", "results": [], "total": 0, "total_count": 0})

        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        img_paths = []
        for ext in exts:
            img_paths += glob.glob(os.path.join(folder, ext))

        if not img_paths:
            return jsonify({"error": "文件夹中没有图片文件", "results": [], "total": 0, "total_count": 0})

        results = []
        total = 0
        for p in img_paths:
            try:
                with open(p, 'rb') as f:
                    res, err = detect_img_bytes(f.read(), conf, iou)
                    if err:
                        results.append({"name": os.path.basename(p), "count": 0, "error": err})
                    else:
                        cnt = res['count'] if res else 0
                        results.append({"name": os.path.basename(p), "count": cnt})
                        total += cnt
            except Exception as e:
                results.append({"name": os.path.basename(p), "count": 0, "error": str(e)})

        return jsonify({
            "results": results,
            "total": len(img_paths),
            "total_count": total,
            "error": None
        })
    except Exception as e:
        return jsonify({"error": f"文件夹检测异常: {str(e)}", "results": [], "total": 0, "total_count": 0})


@app.route('/save_results', methods=['POST'])
def save_results():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "msg": "无效请求数据"})

        persons = data.get('persons', [])
        img_b64 = data.get('image_base64', '')

        if not persons:
            return jsonify({"success": False, "msg": "没有检测结果可保存"})

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(SAVE_ROOT_DIR, f"result_{ts}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(persons, f, ensure_ascii=False, indent=2)

        csv_rows = [{
            "id": i + 1,
            "class": p.get("class_name", "person"),
            "conf": p.get("conf", 0),
            "x1": p.get("x1", 0),
            "y1": p.get("y1", 0),
            "x2": p.get("x2", 0),
            "y2": p.get("y2", 0)
        } for i, p in enumerate(persons)]

        csv_path = os.path.join(SAVE_ROOT_DIR, f"result_{ts}.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

        img_path = None
        if img_b64 and 'base64,' in img_b64:
            try:
                img_data = base64.b64decode(img_b64.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img_path = os.path.join(SAVE_ROOT_DIR, f"result_{ts}.jpg")
                    cv2.imwrite(img_path, img)
            except Exception as e:
                print(f"图片保存失败: {e}")

        return jsonify({"success": True, "msg": f"保存成功，已保存到 {SAVE_ROOT_DIR}"})
    except Exception as e:
        return jsonify({"success": False, "msg": f"保存失败: {str(e)}"})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_session is not None,
        "current_model": current_model_name,
        "models_available": len(get_available_models())
    })


def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == '__main__':
    print("=" * 60)
    print("  体育场人群检测系统 ✅ ")
    print(f"  模型目录：{MODEL_ROOT_DIR}")
    print(f"  保存目录：{SAVE_ROOT_DIR}")
    print(f"  当前模型：{current_model_name}")
    print("=" * 60)

    if model_session is None:
        print("⚠️  警告: 未加载任何模型")
        print("   请将ONNX模型文件放入以下目录:")
        print(f"   {MODEL_ROOT_DIR}")

    threading.Timer(1.2, open_browser).start()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)