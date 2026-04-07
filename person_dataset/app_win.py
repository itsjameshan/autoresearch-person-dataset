# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
import os
import sys
import warnings
import webbrowser
import traceback
import io
import threading
import time
import glob
import json
import base64
import pandas as pd

warnings.filterwarnings('ignore')

# Windows 控制台默认 GBK，print(emoji) 会 UnicodeEncodeError
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 页面与静态资源在 templates/；勿依赖 cwd，否则从别的目录启动会 404
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, static_folder=None)
CORS(app)
app.config['SECRET_KEY'] = 'person_detect'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
MODEL_ROOT_DIR = os.path.join(BASE_DIR, "onnx_data")
SAVE_ROOT_DIR = os.path.join(BASE_DIR, "result")

os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

model_session = None
input_name = None
output_names = None
current_model_name = "未加载"
IMG_SIZE = 1280
PORT = int(os.environ.get("APP_WIN_PORT", "5050"))
GLOBAL_MERGE_IOU = 0.35

model_lock = threading.Lock()


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


# 自动优先加载 yolo12l.onnx
models = get_available_models()
target_model = "yolo12l.onnx"
loaded = False

for m in models:
    if m["name"] == target_model:
        ok, msg = load_model(m["path"])
        print(msg)
        loaded = True
        break

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


def nms_xyxy(boxes, scores, iou_threshold):
    """boxes: Nx4 xyxy, scores: N. Returns list of kept indices."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        current = int(idxs[0])
        keep.append(current)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        bc = boxes[current]
        xx1 = np.maximum(bc[0], boxes[rest, 0])
        yy1 = np.maximum(bc[1], boxes[rest, 1])
        xx2 = np.minimum(bc[2], boxes[rest, 2])
        yy2 = np.minimum(bc[3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_c = (bc[2] - bc[0]) * (bc[3] - bc[1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_c + area_r - inter
        iou = inter / (union + 1e-6)
        idxs = rest[iou <= iou_threshold]
    return keep


def run_inference_tile(img_bgr, conf, iou):
    """单块 BGR 推理，返回 persons 列表（与 postprocess 一致）。"""
    if model_session is None:
        return []
    with model_lock:
        blob, scale, pw, ph, w, h = preprocess(img_bgr)
        outs = model_session.run(output_names, {input_name: blob})
        return postprocess(outs, scale, pw, ph, w, h, conf, iou)


def iter_sliding_tiles(img_bgr, tile_size, overlap):
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
    orig_w = int(payload["orig_w"])
    orig_h = int(payload["orig_h"])
    all_boxes = []
    all_scores = []
    for t in payload.get("tiles", []):
        x0, y0 = int(t["x0"]), int(t["y0"])
        for p in t.get("persons", []):
            all_boxes.append([
                float(p["x1"]) + x0,
                float(p["y1"]) + y0,
                float(p["x2"]) + x0,
                float(p["y2"]) + y0,
            ])
            all_scores.append(float(p["conf"]))
    if not all_boxes:
        return [], orig_w, orig_h
    boxes = np.array(all_boxes, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    keep = nms_xyxy(boxes, scores, float(merge_iou))
    merged = [(boxes[i].copy(), float(scores[i])) for i in keep]
    return merged, orig_w, orig_h


def merged_boxes_to_yolo_txt(merged_pairs, orig_w, orig_h):
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
        cv2.rectangle(img, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 0), 2)
        cv2.putText(img, f"person {p['conf']:.2f}", (p["x1"], p["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return {
        "count": len(persons),
        "persons": persons,
        "image_data": to_base64(img),
        "infer_time": round(time.time() - t0, 3)
    }, None


@app.route("/")
def index():
    return send_from_directory(TEMPLATE_DIR, "index.html")


@app.route("/templates/<path:filename>")
def serve_templates(filename):
    return send_from_directory(TEMPLATE_DIR, filename)


@app.route('/api/config', methods=['GET'])
def api_config():
    """供前端拼出本机 images 子目录绝对路径（随仓库位置迁移，勿写死盘符）。"""
    root = os.path.join(BASE_DIR, "images")
    return jsonify({"images_root": root.replace("\\", "/")})


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


# ==================== ✅ 最终修复：文件夹检测（显示图片 + 实时进度条） ====================
@app.route('/detect_folder', methods=['POST'])
def detect_folder():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效请求数据", "results": [], "total": 0, "total_count": 0, "progress": 100})

        folder = data.get('folder_path', '').strip()
        conf = float(data.get('conf', 0.5))
        iou = float(data.get('iou', 0.5))

        conf = max(0, min(1, conf))
        iou = max(0, min(1, iou))

        if not folder or not os.path.isdir(folder):
            return jsonify({"error": "文件夹不存在", "results": [], "total": 0, "total_count": 0, "progress": 100})

        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_paths = []
        seen = set()
        for ext in exts:
            for p in glob.glob(os.path.join(folder, ext)):
                k = os.path.basename(p).lower()
                if k not in seen:
                    seen.add(k)
                    img_paths.append(p)

        if not img_paths:
            return jsonify({"error": "无图片", "results": [], "total": 0, "total_count": 0, "progress": 100})

        total = len(img_paths)
        results = []
        total_count = 0

        for idx, path in enumerate(img_paths):
            try:
                with open(path, 'rb') as f:
                    res, err = detect_img_bytes(f.read(), conf, iou)
                    progress = int((idx + 1) / total * 100)

                    if err:
                        item = {
                            "name": os.path.basename(path),
                            "count": 0,
                            "persons": [],
                            "image_data": "",
                            "progress": progress
                        }
                    else:
                        item = {
                            "name": os.path.basename(path),
                            "count": res["count"],
                            "persons": res["persons"],
                            "image_data": res["image_data"],
                            "progress": progress
                        }
                        total_count += res["count"]

                    results.append(item)
            except:
                results.append({
                    "name": os.path.basename(path),
                    "count": 0,
                    "persons": [],
                    "image_data": "",
                    "progress": int((idx + 1) / total * 100)
                })

        return jsonify({
            "results": results,
            "total": total,
            "total_count": total_count,
            "error": None,
            "progress": 100
        })

    except Exception as e:
        return jsonify({"error": f"异常：{str(e)}", "results": [], "total": 0, "total_count": 0, "progress": 100})


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


@app.route("/a")
def pipeline_page_a():
    return send_from_directory(TEMPLATE_DIR, "page_a.html")


@app.route("/b")
def pipeline_page_b():
    return send_from_directory(TEMPLATE_DIR, "page_b.html")


@app.route('/api/pipeline_a', methods=['POST'])
def api_pipeline_a():
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400
    f = request.files['image']
    if not f.filename:
        return jsonify({'error': '空文件名'}), 400
    try:
        overlap = int(request.form.get('overlap', 200))
    except ValueError:
        overlap = 200
    overlap = max(0, min(overlap, IMG_SIZE - 1))
    try:
        conf = float(request.form.get('conf', 0.35))
        iou = float(request.form.get('iou', 0.25))
    except ValueError:
        conf, iou = 0.35, 0.25
    conf = max(0.0, min(1.0, conf))
    iou = max(0.0, min(1.0, iou))

    try:
        nparr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': '图片解码失败'}), 400
        if model_session is None:
            return jsonify({'error': '模型未加载'}), 503

        H, W = img.shape[:2]
        stride = max(1, IMG_SIZE - overlap)
        tiles_out = []
        sum_raw = 0
        idx = 0

        for x0, y0, tile in iter_sliding_tiles(img, IMG_SIZE, overlap):
            persons = run_inference_tile(tile, conf, iou)
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
            'conf': conf,
            'iou': iou,
            'tiles': tiles_out,
            'sum_raw': sum_raw,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline_b', methods=['POST'])
def api_pipeline_b():
    merge_iou = GLOBAL_MERGE_IOU
    image_bytes = None
    data = None

    if request.is_json:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': '无效 JSON'}), 400
        merge_iou = float(data.get('merge_iou', GLOBAL_MERGE_IOU))
        b64 = data.get('image_base64')
        if b64:
            s = b64.split(',', 1)[-1] if isinstance(b64, str) else ''
            try:
                image_bytes = base64.b64decode(s)
            except Exception:
                return jsonify({'error': 'image_base64 无效'}), 400
    else:
        raw = request.form.get('payload')
        if not raw:
            return jsonify({'error': '缺少表单字段 payload（JSON 字符串）'}), 400
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'payload 非合法 JSON: {e}'}), 400
        try:
            merge_iou = float(request.form.get('merge_iou', GLOBAL_MERGE_IOU))
        except ValueError:
            merge_iou = GLOBAL_MERGE_IOU
        if 'image' in request.files and request.files['image'].filename:
            image_bytes = request.files['image'].read()

    if not isinstance(data, dict) or 'tiles' not in data:
        return jsonify({'error': 'JSON 须包含 tiles'}), 400

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
                out['image_error'] = '原图解码失败，跳过画框'
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
                out['image_data'] = to_base64(vis)

        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:{PORT}/")


if __name__ == '__main__':
    print("=" * 60)
    print("  体育场人群检测系统 ✅ ")
    print(f"  模型目录：{MODEL_ROOT_DIR}")
    print(f"  保存目录：{SAVE_ROOT_DIR}")
    print(f"  当前模型：{current_model_name}")
    print(f"  主页: http://127.0.0.1:{PORT}/")
    print(f"  网页 A（滑窗）: http://127.0.0.1:{PORT}/a")
    print(f"  网页 B（拼接）: http://127.0.0.1:{PORT}/b")
    print("  (默认端口 5050，避免与占用 5000 的其它服务冲突；可设 APP_WIN_PORT)")
    print("=" * 60)

    if model_session is None:
        print("⚠️  警告: 未加载任何模型")
        print("   请将ONNX模型文件放入以下目录:")
        print(f"   {MODEL_ROOT_DIR}")

    threading.Timer(1.2, open_browser).start()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)