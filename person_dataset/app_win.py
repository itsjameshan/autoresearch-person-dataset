# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory, send_file, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import onnxruntime as ort
import os
import warnings
import webbrowser
import threading
import time
import glob
import json
import base64
import pandas as pd
from datetime import datetime
import tempfile
import shutil
import zipfile
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
app.config['SECRET_KEY'] = 'person_detect'  # 生产环境请改为随机字符串
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

# ==================== 用户数据文件 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, 'users.json')

def load_users():
    """从文件加载用户字典"""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    """保存用户字典到文件"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def init_default_user():
    """如果用户文件不存在，创建默认管理员账户"""
    if not os.path.exists(USERS_FILE):
        default_users = {'admin': 'admin123'}
        save_users(default_users)
        print("✅ 已创建默认用户: admin / admin123")

# 在启动时初始化默认用户
init_default_user()

# ==================== 核心参数（双模型兼容） ====================
MODEL_ROOT_DIR = os.path.join(BASE_DIR, "onnx_data")
SAVE_ROOT_DIR = r"D:\pythonProject\person_dataset\result"
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

# 全局变量声明
model_session = None
input_name = None
output_names = None
current_model_name = "未加载"
is_dynamic_model = False

# 固定尺寸模型参数
FIXED_IMG_SIZE = 1280
STRIDE = 32
PORT = 5000

MIN_BOX_W = 15
MIN_BOX_H = 30
GLOBAL_NMS_IOU = 0.75

model_lock = threading.Lock()


# ==================== 登录拦截器 ====================
@app.before_request
def check_login():
    # 允许访问的路径白名单
    public_paths = ['/login', '/static', '/api/login', '/api/logout', '/api/register', '/api/check_login', '/health']
    path = request.path
    if any(path.startswith(p) for p in public_paths):
        return None
    # 检查 session 中是否有 user
    if 'user' not in session:
        # 如果是 API 请求，返回 401
        if path.startswith('/api/'):
            return jsonify({"error": "未登录"}), 401
        # 页面请求重定向到登录页
        return redirect('/login')


# ==================== 模型加载（自动识别动态/固定） ====================
def get_available_models():
    models = []
    if not os.path.exists(MODEL_ROOT_DIR):
        os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
        return models
    for f in os.listdir(MODEL_ROOT_DIR):
        if f.lower().endswith(".onnx"):
            models.append({"name": f, "path": os.path.join(MODEL_ROOT_DIR, f)})
    print(f"✅ 找到 {len(models)} 个ONNX模型: {[m['name'] for m in models]}")
    return models


def load_model(model_path):
    global model_session, input_name, output_names, current_model_name, is_dynamic_model
    try:
        with model_lock:
            if model_session:
                del model_session
                model_session = None

            available_providers = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print("✅ 启用GPU加速 (CUDA)")
            else:
                providers = ["CPUExecutionProvider"]
                print("⚠️ 未检测到CUDA，使用CPU推理")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_profiling = False
            sess_options.intra_op_num_threads = 8

            model_session = ort.InferenceSession(model_path, sess_options, providers=providers)
            input_shape = model_session.get_inputs()[0].shape
            is_dynamic_model = any(s == -1 for s in input_shape)
            input_name = model_session.get_inputs()[0].name
            output_names = [o.name for o in model_session.get_outputs()]
            current_model_name = os.path.basename(model_path)
            print(f"✅ 模型加载成功: {current_model_name}, 输入形状: {input_shape}, 动态模型: {is_dynamic_model}")

        return True, f"✅ 模型加载成功：{current_model_name}"
    except Exception as e:
        model_session = None
        print(f"❌ 模型加载失败: {str(e)}")
        return False, f"❌ 加载失败：{str(e)}"


models = get_available_models()
target_model = "best.onnx"
loaded = False

for m in models:
    if m["name"] == target_model:
        ok, msg = load_model(m["path"])
        if ok:
            loaded = True
            break

if not loaded:
    print(f"❌ 错误: 未找到或无法加载 {target_model}，请检查模型！")
    current_model_name = "未加载"
else:
    print(f"✅ 系统启动完成，当前模型: {current_model_name}")


# ==================== 预处理（自动适配动态/固定） ====================
def preprocess(image):
    h, w = image.shape[:2]
    if is_dynamic_model:
        scale = min(1920 / w, 1920 / h)
        new_w = int(round(w * scale / STRIDE) * STRIDE)
        new_h = int(round(h * scale / STRIDE) * STRIDE)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[None], w, h, new_w, new_h
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = min(FIXED_IMG_SIZE / w, FIXED_IMG_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        pad_w = (FIXED_IMG_SIZE - new_w) / 2
        pad_h = (FIXED_IMG_SIZE - new_h) / 2
        top = int(np.floor(pad_h))
        bottom = int(np.ceil(pad_h))
        left = int(np.floor(pad_w))
        right = int(np.ceil(pad_w))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[None], scale, pad_w, pad_h, w, h


# ==================== xywh转xyxy ====================
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# ==================== 后处理（自动适配动态/固定） ====================
def postprocess(outputs, conf_thres, *args):
    scale, pad_w, pad_h, w, h = args
    # outputs[0] shape: (1, 5, 33600)  -> 转置成 (33600, 5)
    pred = outputs[0][0].T

    # 5列含义：cx, cy, bw, bh, obj_conf
    obj_conf = pred[:, 4]
    keep = obj_conf >= conf_thres
    boxes = pred[keep, :4]
    scores = pred[keep, 4]

    if len(boxes) == 0:
        return []

    # 将 cx,cy,w,h 转为 xyxy
    boxes_xyxy = xywh2xyxy(boxes)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        scores.tolist(),
        conf_thres,
        GLOBAL_NMS_IOU
    )
    if len(indices) == 0:
        return []

    indices = indices.flatten()
    dets = []
    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i]
        # 将归一化坐标（相对于1280x1280）还原到原图尺寸
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        x1 = max(0, min(int(x1), w))
        y1 = max(0, min(int(y1), h))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < MIN_BOX_W or box_h < MIN_BOX_H:
            continue

        dets.append([x1, y1, x2, y2, scores[i]])
    return dets


# ==================== base64编码 ====================
def to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


# ==================== 检测函数（双模型兼容） ====================
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
        if is_dynamic_model:
            blob, orig_w, orig_h, new_w, new_h = preprocess(img)
            outs = model_session.run(output_names, {input_name: blob})
            dets = postprocess(outs, conf, orig_w, orig_h, new_w, new_h)
        else:
            blob, scale, pw, ph, w, h = preprocess(img)
            outs = model_session.run(output_names, {input_name: blob})
            dets = postprocess(outs, conf, scale, pw, ph, w, h)

    # ====================== 【我加的打印，不影响任何逻辑】 ======================
    print("\n📌 快速诊断打印 ↓")
    print(f"模型输出结果数量 (outs): {len(outs)}")
    print(f"后处理输出框数量 (dets): {len(dets)}")
    # ==========================================================================
    persons = []
    if dets:
        for x1, y1, x2, y2, score in dets:
            persons.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "conf": round(float(score), 3),
                "class_name": "person"
            })

    out_img = img.copy()
    for p in persons:
        cv2.rectangle(out_img, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 0), 2)

    print(f"✅ 检测完成，耗时: {round(time.time() - t0, 3)}s，检测到: {len(persons)}人")
    return {
        "count": len(persons),
        "persons": persons,
        "image_data": to_base64(out_img),
        "infer_time": round(time.time() - t0, 3)
    }, None

# ==================== 接口 ====================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/login')
def login_page():
    return send_from_directory('static', 'login.html')


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    users = load_users()
    if username in users and users[username] == password:
        session['user'] = username
        return jsonify({"ok": True, "msg": "登录成功"})
    else:
        return jsonify({"ok": False, "msg": "用户名或密码错误"}), 401

@app.route('/api/check_login', methods=['GET'])
def check_login_status():
    if 'user' in session:
        return jsonify({"logged_in": True, "user": session['user']})
    else:
        return jsonify({"logged_in": False})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"ok": False, "msg": "用户名和密码不能为空"}), 400
    if len(username) < 3 or len(password) < 6:
        return jsonify({"ok": False, "msg": "用户名至少3位，密码至少6位"}), 400

    users = load_users()
    if username in users:
        return jsonify({"ok": False, "msg": "用户名已存在"}), 409

    users[username] = password
    save_users(users)
    return jsonify({"ok": True, "msg": "注册成功，请登录"})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop('user', None)
    return jsonify({"ok": True, "msg": "已退出登录"})


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
        # 固定置信度和 IoU 为 0.3
        conf = 0.3
        iou = 0.3

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
        folder = data.get('folder_path', '').strip()
        # 固定置信度和 IoU 为 0.3，忽略前端传递的值
        conf = 0.08
        iou = 0.75

        if not os.path.isdir(folder):
            return jsonify({"error": "文件夹不存在", "results": []})

        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_paths = []
        for ext in exts:
            img_paths += glob.glob(os.path.join(folder, '**', ext), recursive=True)

        if not img_paths:
            return jsonify({"error": "无图片", "results": []})

        results = []
        total_count = 0
        for path in img_paths:
            try:
                with open(path, 'rb') as f:
                    res, err = detect_img_bytes(f.read(), conf, iou)
                    if err:
                        results.append({"name": os.path.basename(path), "count": 0, "persons": [], "image_data": ""})
                    else:
                        results.append({"name": os.path.basename(path), **res})
                        total_count += res["count"]
                        json_save_path = os.path.splitext(path)[0] + '.json'
                        with open(json_save_path, 'w', encoding='utf-8') as jf:
                            json.dump(res["persons"], jf, ensure_ascii=False, indent=2)
            except:
                results.append({"name": os.path.basename(path), "count": 0, "persons": [], "image_data": ""})

        return jsonify({
            "results": results,
            "total": len(img_paths),
            "total_count": total_count,
            "error": None
        })
    except Exception as e:
        return jsonify({"error": f"异常：{str(e)}", "results": []})


@app.route('/save_results', methods=['POST'])
def save_results():
    try:
        data = request.get_json()
        persons = data.get('persons', [])
        img_b64 = data.get('image_base64', '')

        if not persons:
            return jsonify({"success": False, "msg": "没有检测结果可保存"})

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(SAVE_ROOT_DIR, f"result_{ts}.json")
        with open(json_path, 'w', encoding='utf-8', newline='') as f:
            json.dump(persons, f, ensure_ascii=False, indent=2)

        csv_rows = [{
            "id": i + 1, "class": p["class_name"], "conf": p["conf"],
            "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]
        } for i, p in enumerate(persons)]
        pd.DataFrame(csv_rows).to_csv(os.path.join(SAVE_ROOT_DIR, f"result_{ts}.csv"), index=False,
                                      encoding='utf-8-sig')

        return jsonify({"success": True, "msg": f"保存成功：{SAVE_ROOT_DIR}"})
    except Exception as e:
        return jsonify({"success": False, "msg": f"保存失败：{str(e)}"})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_session is not None,
        "current": current_model_name,
        "models_available": len(get_available_models()),
        "is_dynamic": is_dynamic_model
    })


# ==================== 裁剪核心函数（供多个接口复用） ====================
def crop_images(src_folder, dst_folder, tile_size, overlap, filter_empty):
    """裁剪核心逻辑，返回 (total_tiles, mapping_path)"""
    mapping_records = []
    total_tiles = 0

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(src_folder, ext)))

    print(f"🔍 裁剪核心：找到 {len(img_paths)} 个图片文件")

    for img_path in img_paths:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ 无法解码图片: {img_path}")
            continue

        h, w = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        stand_folder = os.path.join(dst_folder, base_name)
        os.makedirs(stand_folder, exist_ok=True)

        stride = tile_size - overlap

        if h < tile_size or w < tile_size:
            y_steps = [0]
            x_steps = [0]
        else:
            y_steps = list(range(0, h - tile_size + 1, stride))
            if y_steps and y_steps[-1] + tile_size < h:
                y_steps.append(h - tile_size)
            x_steps = list(range(0, w - tile_size + 1, stride))
            if x_steps and x_steps[-1] + tile_size < w:
                x_steps.append(w - tile_size)

        for i, y in enumerate(y_steps):
            for j, x in enumerate(x_steps):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = img[y:y_end, x:x_end]
                tile_name = f"{base_name}_{i}_{j}_{x}_{y}.jpg"
                tile_path = os.path.join(stand_folder, tile_name)

                is_empty = False
                if filter_empty:
                    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                    if np.var(gray) < 30:
                        is_empty = True

                if is_empty:
                    continue

                _, buf = cv2.imencode('.jpg', tile)
                with open(tile_path, 'wb') as f_out:
                    f_out.write(buf)

                total_tiles += 1
                mapping_records.append({
                    "stand": base_name,
                    "tile": tile_name,
                    "x_offset": x,
                    "y_offset": y
                })

    mapping_df = pd.DataFrame(mapping_records)
    mapping_path = os.path.join(dst_folder, "crop_mapping.xlsx")
    mapping_df.to_excel(mapping_path, index=False)

    print(f"✅ 裁剪核心完成，共生成 {total_tiles} 张小图，映射表：{mapping_path}")
    return total_tiles, mapping_path


# ==================== 裁剪 API（手动路径版本，保留兼容） ====================
@app.route('/api/crop', methods=['POST'])
def api_crop():
    try:
        data = request.get_json()
        src_folder = data.get('src_folder', '').strip()
        dst_folder = data.get('dst_folder', '').strip()
        overlap = int(data.get('overlap', 200))
        tile_size = int(data.get('tile_size', 1280))
        filter_empty = data.get('filter_empty', True)

        print(f"\n📂 裁剪任务开始（手动路径）")
        print(f"   源文件夹: {src_folder}")
        print(f"   输出文件夹: {dst_folder}")

        if not os.path.isdir(src_folder):
            return jsonify({"ok": False, "msg": "源文件夹不存在"})

        os.makedirs(dst_folder, exist_ok=True)

        total_tiles, mapping_path = crop_images(src_folder, dst_folder, tile_size, overlap, filter_empty)

        return jsonify({
            "ok": True,
            "total_tiles": total_tiles,
            "mapping_file": mapping_path,
            "dst_folder": dst_folder
        })
    except Exception as e:
        print(f"❌ 裁剪过程出错: {str(e)}")
        return jsonify({"ok": False, "msg": f"裁剪过程出错: {str(e)}"})


# ==================== 上传文件夹并裁剪（本地文件夹选择版，固定目录） ====================
@app.route('/api/upload_and_crop', methods=['POST'])
def upload_and_crop():
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({"ok": False, "msg": "未接收到文件"})

        # 固定根目录
        root_dir = r"D:\pythonProject\person_dataset\rentoujieguo"
        os.makedirs(root_dir, exist_ok=True)

        # 生成项目文件夹名（例如：项目_20260421_131536）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"项目_{timestamp}"
        project_dir = os.path.join(root_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        # 创建子文件夹
        src_folder = os.path.join(project_dir, "原始大图")
        dst_folder = os.path.join(project_dir, "裁剪结果")
        os.makedirs(src_folder, exist_ok=True)
        os.makedirs(dst_folder, exist_ok=True)

        saved_count = 0
        for file in files:
            if file.filename:
                filename = secure_filename(os.path.basename(file.filename))
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file.save(os.path.join(src_folder, filename))
                    saved_count += 1
                else:
                    print(f"⚠️ 跳过非图片文件：{file.filename}")

        if saved_count == 0:
            shutil.rmtree(project_dir, ignore_errors=True)
            return jsonify({"ok": False, "msg": "上传的文件中没有有效图片"})

        tile_size = int(request.form.get('tile_size', 1280))
        overlap = int(request.form.get('overlap', 200))
        filter_empty = request.form.get('filter_empty', 'true') == 'true'

        total_tiles, mapping_path = crop_images(src_folder, dst_folder, tile_size, overlap, filter_empty)

        return jsonify({
            "ok": True,
            "total_tiles": total_tiles,
            "mapping_file": mapping_path,
            "dst_folder": dst_folder,
            "src_folder": src_folder,
            "timestamp": timestamp,
            "work_root": project_dir      # 项目根目录
        })
    except Exception as e:
        print(f"❌ 上传裁剪出错: {str(e)}")
        return jsonify({"ok": False, "msg": f"处理失败: {str(e)}"})


@app.route('/api/upload_folder_for_detect', methods=['POST'])
def upload_folder_for_detect():
    """上传文件夹用于检测，返回服务器临时目录路径"""
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({"ok": False, "msg": "未接收到文件"})

        root_dir = r"D:\pythonProject\person_dataset\rentoujieguo"
        os.makedirs(root_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"项目_检测_{timestamp}"
        project_dir = os.path.join(root_dir, project_name)
        upload_dir = os.path.join(project_dir, "原始图片")
        os.makedirs(upload_dir, exist_ok=True)

        saved_count = 0
        for file in files:
            if file.filename:
                filename = secure_filename(os.path.basename(file.filename))
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    save_path = os.path.join(upload_dir, filename)
                    file.save(save_path)
                    saved_count += 1
                else:
                    print(f"⚠️ 跳过非图片文件：{filename}")

        if saved_count == 0:
            shutil.rmtree(project_dir, ignore_errors=True)
            return jsonify({"ok": False, "msg": "上传的文件中没有有效图片"})

        print(f"📁 检测文件夹已上传至: {upload_dir}，共 {saved_count} 张图片")
        return jsonify({
            "ok": True,
            "uploaded_path": upload_dir,
            "file_count": saved_count,
            "timestamp": timestamp,
            "work_root": project_dir
        })
    except Exception as e:
        print(f"❌ 上传检测文件夹出错: {str(e)}")
        return jsonify({"ok": False, "msg": f"上传失败: {str(e)}"})


# ==================== 上传裁剪（旧版，保留兼容） ====================
@app.route('/api/crop_upload', methods=['POST'])
def api_crop_upload():
    try:
        files = request.files.getlist('images')
        dst_folder = request.form.get('dst_folder', '').strip()
        overlap = int(request.form.get('overlap', 200))
        tile_size = int(request.form.get('tile_size', 1280))
        filter_empty = request.form.get('filter_empty', 'true') == 'true'

        if not files or len(files) == 0:
            return jsonify({"ok": False, "msg": "未接收到图片文件"})
        if not dst_folder:
            return jsonify({"ok": False, "msg": "请填写输出目录"})

        temp_src = tempfile.mkdtemp(prefix='crop_upload_')
        print(f"📁 临时目录：{temp_src}")

        saved_count = 0
        for file in files:
            if file.filename:
                filename = os.path.basename(file.filename)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    save_path = os.path.join(temp_src, filename)
                    file.save(save_path)
                    saved_count += 1
                else:
                    print(f"⚠️ 跳过非图片文件：{file.filename}")

        if saved_count == 0:
            shutil.rmtree(temp_src, ignore_errors=True)
            return jsonify({"ok": False, "msg": "上传的文件中没有有效图片"})

        print(f"📸 已保存 {saved_count} 张图片到临时目录，开始裁剪...")

        total_tiles, mapping_path = crop_images(temp_src, dst_folder, tile_size, overlap, filter_empty)

        shutil.rmtree(temp_src, ignore_errors=True)
        try:
            shutil.rmtree(temp_src)
            print(f"🧹 临时目录 {temp_src} 已清理")
        except Exception as e:
            print(f"⚠️ 临时目录 {temp_src} 清理失败: {str(e)}")

        return jsonify({
            "ok": True,
            "total_tiles": total_tiles,
            "mapping_file": mapping_path,
            "dst_folder": dst_folder
        })
    except Exception as e:
        print(f"❌ 上传裁剪出错: {str(e)}")
        return jsonify({"ok": False, "msg": f"上传裁剪出错: {str(e)}"})


# ==================== 重建 API ====================
@app.route('/api/rebuild', methods=['POST'])
def api_rebuild():
    try:
        data = request.get_json()
        detect_folder = data.get('detect_result_folder', '').strip()
        mapping_file = data.get('mapping_file', '').strip()
        original_folder = data.get('original_img_folder', '').strip()
        output_folder = data.get('output_folder', '').strip()
        nms_iou = float(data.get('nms_iou', 0.4))
        skip_nms = data.get('skip_nms', False)

        output_folder = os.path.abspath(output_folder)

        print(f"\n🗺️ 重建任务开始")
        print(f"   检测结果文件夹: {detect_folder}")
        print(f"   映射文件: {mapping_file}")
        print(f"   原始大图文件夹: {original_folder}")
        print(f"   输出目录: {output_folder}")
        print(f"   NMS 阈值: {nms_iou}")
        print(f"   跳过 NMS: {skip_nms}")

        if not os.path.isdir(detect_folder):
            return jsonify({"ok": False, "msg": "检测结果文件夹不存在"})
        if not os.path.isfile(mapping_file):
            return jsonify({"ok": False, "msg": "映射文件不存在"})
        if not os.path.isdir(original_folder):
            return jsonify({"ok": False, "msg": "原始大图文件夹不存在"})

        os.makedirs(output_folder, exist_ok=True)

        df_map = pd.read_excel(mapping_file)

        all_boxes = []
        json_count = 0
        used_flat_mode = False

        for _, row in df_map.iterrows():
            stand = row['stand']                # 正确的看台名称
            tile_name = row['tile']
            x_off = row['x_offset']
            y_off = row['y_offset']

            # 1. 先尝试标准目录结构：detect_folder/stand/tile_name.json
            json_path = os.path.join(detect_folder, stand, tile_name.replace('.jpg', '.json'))

            # 2. 若不存在，尝试平铺模式：detect_folder/tile_name.json（去掉子文件夹）
            if not os.path.exists(json_path):
                json_path_flat = os.path.join(detect_folder, tile_name.replace('.jpg', '.json'))
                if os.path.exists(json_path_flat):
                    json_path = json_path_flat
                    used_flat_mode = True
                else:
                    continue

            json_count += 1

            with open(json_path, 'r', encoding='utf-8') as f:
                dets = json.load(f)

            for d in dets:
                x1 = d['x1'] + x_off
                y1 = d['y1'] + y_off
                x2 = d['x2'] + x_off
                y2 = d['y2'] + y_off
                all_boxes.append({
                    'stand': stand,          # 使用映射表中的看台名
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': d['conf']
                })

        if used_flat_mode:
            print("📌 检测结果采用平铺模式（JSON 位于根目录）")
        print(f"📁 读取了 {json_count} 个 JSON 文件，总计 {len(all_boxes)} 个检测框（还原前）")

        if not all_boxes:
            return jsonify({"ok": False, "msg": "未找到任何检测结果"})

        df_all = pd.DataFrame(all_boxes)
        summary = []

        print(f"📊 检测到 {len(df_all['stand'].unique())} 个看台")

        for stand, group in df_all.groupby('stand'):
            boxes = group[['x1', 'y1', 'x2', 'y2']].values.astype(float)
            scores = group['conf'].values.astype(float)
            boxes_before = len(boxes)

            if not skip_nms:
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.25, nms_threshold=nms_iou)
                if len(indices) > 0:
                    indices = indices.flatten()
                    final_boxes = boxes[indices]
                    final_scores = scores[indices]
                else:
                    final_boxes = []
                    final_scores = []
            else:
                final_boxes = boxes
                final_scores = scores

            print(f"  看台 '{stand}': 还原框 {boxes_before} 个, NMS后 {len(final_boxes)} 个")

            orig_img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                p = os.path.join(original_folder, f"{stand}{ext}")
                if os.path.exists(p):
                    orig_img_path = p
                    break
            if orig_img_path is None:
                print(f"      ⚠️ 警告：找不到原始大图 {stand}，跳过绘图，但结果仍计入CSV")
            else:
                try:
                    with open(orig_img_path, 'rb') as f:
                        img_bytes = f.read()
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"      ❌ 无法解码图片: {orig_img_path}")
                    else:
                        for (x1, y1, x2, y2), conf in zip(final_boxes, final_scores):
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(img, f"Total: {len(final_boxes)}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                        out_img_path = os.path.join(output_folder, f"{stand}_detected.jpg")
                        _, buf = cv2.imencode('.jpg', img)
                        with open(out_img_path, 'wb') as f_out:
                            f_out.write(buf)
                        print(f"      ✅ 已保存带框大图: {out_img_path}")
                except Exception as e:
                    print(f"      ❌ 处理原始大图时出错: {e}")

            for (x1, y1, x2, y2), conf in zip(final_boxes, final_scores):
                summary.append({
                    'stand': stand,
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'conf': float(conf)
                })

        summary_df = pd.DataFrame(summary)
        csv_path = os.path.join(output_folder, "detection_summary.csv")
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 汇总 CSV 已保存: {csv_path}")
        print(f"📈 最终统计：处理看台数 {len(summary_df['stand'].unique())}，总检测人数 {len(summary_df)}")

        return jsonify({
            "ok": True,
            "output_folder": output_folder,
            "total_stands": len(summary_df['stand'].unique()),
            "total_persons": len(summary_df)
        })
    except Exception as e:
        print(f"❌ 重建过程出错: {str(e)}")
        return jsonify({"ok": False, "msg": f"重建过程出错: {str(e)}"})


# ==================== 文件夹列表 API ====================
@app.route('/api/list_folders', methods=['POST'])
def list_folders():
    data = request.get_json()
    base_path = data.get('path', '').strip()
    if not os.path.isdir(base_path):
        base_path = 'D:\\'
    try:
        items = []
        for item in os.listdir(base_path):
            full = os.path.join(base_path, item)
            if os.path.isdir(full):
                items.append({'name': item, 'path': full})
        return jsonify({'ok': True, 'items': items, 'parent': os.path.dirname(base_path)})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})


# ==================== 下载结果打包 ====================
@app.route('/api/download_result', methods=['POST'])
def download_result():
    data = request.get_json()
    output_folder = data.get('output_folder', '')
    if not output_folder or not os.path.isdir(output_folder):
        return jsonify({"ok": False, "msg": "输出目录不存在"})

    zip_path = output_folder + '.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, os.path.dirname(output_folder))
                zf.write(full_path, arcname)

    return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))


# ==================== 工作流页面路由 ====================
@app.route('/crop')
def crop_page():
    return send_from_directory('static', 'crop.html')


@app.route('/detect')
def detect_page():
    return send_from_directory('static', 'detect.html')


@app.route('/rebuild')
def rebuild_page():
    return send_from_directory('static', 'rebuild.html')


def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}/login")


if __name__ == '__main__':
    print("=" * 60)
    print("  体育场人群检测系统 ✅ ")
    print(f"  模型目录：{MODEL_ROOT_DIR}")
    print(f"  保存目录：{SAVE_ROOT_DIR}")
    print(f"  当前模型：{current_model_name}")
    print(f"  动态模型：{is_dynamic_model}")
    print("=" * 60)

    threading.Timer(1.2, open_browser).start()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)