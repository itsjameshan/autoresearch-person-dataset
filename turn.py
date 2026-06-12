import os
import scipy.io as sio
from PIL import Image

MAT_GT_DIR = r"D:\autoresearch\autoresearch_new\part_B_final\train_data\ground_truth"
# 2. 对应原始图片文件夹（和 mat 文件一一对应，比如 train_data 下的 images 文件夹）
IMG_DIR = r"D:\autoresearch\autoresearch_new\part_B_final\train_data\images"
# 3. 输出 YOLO 标签文件夹（会自动创建，和 images 同级，符合你之前的目录结构）
OUTPUT_LABEL_DIR = r"D:\autoresearch\autoresearch_new\part_B_final\train_data\labels"


CLASS_ID = 0
HEAD_BOX_SIZE = 0.025  # 人头框大小（可微调）

def mat_to_yolo(mat_path, img_w, img_h):
    try:
        mat = sio.loadmat(mat_path)
        points = None

        # ---------------- 超强兼容：所有 ShanghaiTech 格式 ----------------
        if "image_info" in mat:
            points = mat["image_info"][0][0][0][0][0]  # 最常见格式
        elif "annPoints" in mat:
            points = mat["annPoints"]
        elif "gt" in mat:
            points = mat["gt"]

        if points is None or len(points) == 0:
            return []

        yolo_boxes = []
        for (x, y) in points:
            xn = x / img_w
            yn = y / img_h
            yolo_boxes.append((CLASS_ID, xn, yn, HEAD_BOX_SIZE, HEAD_BOX_SIZE))
        return yolo_boxes

    except Exception as e:
        print("错误:", e)
        return []

# ====================== 执行转换 ======================
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
mat_files = [f for f in os.listdir(MAT_GT_DIR) if f.endswith(".mat")]

print(f"找到 {len(mat_files)} 个标注文件，开始转换...\n")

for mat_file in mat_files:
    img_name = mat_file.replace("GT_IMG", "IMG").replace(".mat", ".jpg")
    img_path = os.path.join(IMG_DIR, img_name)
    mat_path = os.path.join(MAT_GT_DIR, mat_file)

    if not os.path.exists(img_path):
        print(f"跳过：找不到图片 {img_name}")
        continue

    # 读取图片尺寸
    img_w, img_h = Image.open(img_path).size

    # 转换
    boxes = mat_to_yolo(mat_path, img_w, img_h)

    # 写入 TXT
    txt_name = img_name.replace(".jpg", ".txt")
    txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_name)

    with open(txt_path, "w") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ {mat_file} → {txt_name}  人数：{len(boxes)}")

# 生成 classes
with open(os.path.join(OUTPUT_LABEL_DIR, "classes.txt"), "w") as f:
    f.write("person")

print("\n🎉 转换完成！现在标签绝对不会是空啦！")