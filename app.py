from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import pandas as pd
import io
import base64
from PIL import Image
import onnxruntime as ort
import uuid
from datetime import datetime
import os
import traceback
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# ==================== 模型参数 ====================
MODEL_PATH = "best.onnx"
IMG_SIZE = 1280                # 与训练/导出时一致
CONF_THRESHOLD = 0.25          # 置信度阈值
IOU_THRESHOLD = 0.45           # NMS的IOU阈值
CLASSES = ["person"]           # 单类别

# ==================== 加载ONNX模型（启动时加载一次） ====================
if not os.path.exists(MODEL_PATH):
    print(f"警告: {MODEL_PATH} 文件不存在，请确保模型文件在正确位置")
    model_session = None
else:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model_session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = model_session.get_inputs()[0].name
    output_names = [o.name for o in model_session.get_outputs()]
    print(f"ONNX模型加载成功, 输入: {model_session.get_inputs()[0].shape}")


# ==================== 预处理：letterbox（等比缩放+黑边填充） ====================
def preprocess_image(image):
    """与YOLOv8训练时一致的letterbox预处理，返回输入张量和坐标还原参数"""
    h, w = image.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized_img = cv2.resize(image, (new_w, new_h))

    # 计算填充量
    pad_w = (IMG_SIZE - new_w) / 2
    pad_h = (IMG_SIZE - new_h) / 2
    top, bottom = int(np.floor(pad_h)), int(np.ceil(pad_h))
    left, right = int(np.floor(pad_w)), int(np.ceil(pad_w))
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 转换为模型输入格式: (1, 3, H, W), float32, [0,1]
    input_img = np.transpose(padded_img, (2, 0, 1)) / 255.0
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
    return input_img, scale, pad_w, pad_h, w, h


# ==================== 工具函数 ====================
def xywh2xyxy(x):
    """将 [x, y, w, h] 格式转换为 [x1, y1, x2, y2] 格式"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_threshold):
    """非极大值抑制"""
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


# ==================== 后处理：解析模型输出 ====================
def postprocess_output(outputs, scale, pad_w, pad_h, orig_w, orig_h,
                       conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """
    解析YOLOv8 ONNX输出
    模型输出格式: [1, 5, 33600] -> 转置为 [33600, 5]
    每行: [x_center, y_center, width, height, score]  (单类别nc=1)
    """
    detections = []
    preds = outputs[0]  # shape: [1, 5, 33600]

    # 转置: [1, 5, 33600] -> [33600, 5]
    if preds.ndim == 3:
        preds = preds[0]                    # [5, 33600]
        preds = np.transpose(preds)         # [33600, 5]

    # 分离 boxes 和 scores
    boxes_xywh = preds[:, :4]               # [x, y, w, h] 在1280尺度下
    scores = preds[:, 4]                    # person类别得分

    # 置信度过滤
    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    if boxes_xywh.size == 0:
        return detections

    # xywh -> xyxy
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # 坐标从1280尺度还原到原图尺度（减去padding，除以缩放比例）
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale

    # 裁剪到原图边界内
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    # NMS
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


# ==================== 图片转base64 ====================
def image_to_base64(image):
    """将OpenCV图片转换为base64字符串"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


# ==================== 检测单张图片 ====================
def detect_single_image(image_bytes, draw_boxes=True):
    """检测单张图片，返回检测结果"""
    try:
        # 读取图片
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "图片读取失败"

        # letterbox预处理
        input_img, scale, pad_w, pad_h, orig_w, orig_h = preprocess_image(img)

        # 推理
        if model_session:
            outputs = model_session.run(output_names, {input_name: input_img})

            # 后处理
            boxes = postprocess_output(outputs, scale, pad_w, pad_h, orig_w, orig_h)

            # 绘制检测框
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
            return None, "模型未加载"

    except Exception as e:
        print(f"检测失败详情:\n{traceback.format_exc()}")
        return None, f"检测失败: {str(e)}"


# 静态文件路由
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
    """单张图片检测接口"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

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
    """批量图片检测接口"""
    if 'images' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': '文件列表为空'}), 400

    results = []

    for file in files:
        if file.filename:
            try:
                image_bytes = file.read()
                # 批量检测不绘制框以提高速度
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

    # 生成Excel报告
    excel_data = []
    for r in results:
        excel_data.append({
            '文件名': r['name'],
            '检测人数': r['count'],
            '检测时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '状态': '成功' if 'error' not in r else f"失败: {r.get('error', '未知错误')}"
        })

    df = pd.DataFrame(excel_data)

    # 保存Excel到内存并转换为base64
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='检测结果')

    output.seek(0)
    excel_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
    excel_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # 返回结果
    return jsonify({
        'results': results,
        'excel_data': excel_base64,
        'excel_filename': excel_filename
    })


if __name__ == '__main__':
    print(f"模型加载状态: {'已加载' if model_session else '未加载'}")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)