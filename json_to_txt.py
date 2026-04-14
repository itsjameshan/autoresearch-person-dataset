import os
import json
from pathlib import Path
from PIL import Image

# ====================== 【关键：把你 3 个图片文件夹都填进来】 ======================
IMAGE_FOLDERS = [
    r"D:\autoresearch\autoresearch_new\NWPU-Crowd\images_part2",
    r"D:\autoresearch\autoresearch_new\NWPU-Crowd\images_part4",
    r"D:\autoresearch\autoresearch_new\NWPU-Crowd\images_part5b"
]

JSON_DIR = r"D:\autoresearch\autoresearch_new\NWPU-Crowd\jsons"
OUT_TXT_DIR = r"D:\autoresearch\autoresearch_new\NWPU-Crowd\labels"
# ==================================================================================

os.makedirs(OUT_TXT_DIR, exist_ok=True)


# 自动在所有图片文件夹里找同名图片
def find_image(name):
    for folder in IMAGE_FOLDERS:
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            path = os.path.join(folder, name + ext)
            if os.path.exists(path):
                return path
    return None


# 开始转换
for json_file in os.listdir(JSON_DIR):
    if not json_file.endswith(".json"):
        continue

    name = Path(json_file).stem
    json_path = os.path.join(JSON_DIR, json_file)

    # 找图片
    img_path = find_image(name)
    if not img_path:
        print(f"❌ 找不到图片：{name}")
        continue

    # 读取宽高
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    # 读取标注
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []

    # NWPU-Crowd 是人头点标注，直接转 YOLO 点格式
    if "points" in data:
        for (x, y) in data["points"]:
            xn = x / img_w
            yn = y / img_h
            lines.append(f"0 {xn:.6f} {yn:.6f}")

    # 保存 txt
    txt_path = os.path.join(OUT_TXT_DIR, name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("✅ ✅ ✅ 全部成功转换！labels 文件夹已生成！")