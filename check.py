import os

# -------------------------- 配置路径 --------------------------
IMAGE_DIR = r"D:\pythonProject\person_dataset\images\train"
LABEL_DIR = r"D:\pythonProject\person_dataset\labels\train"
# 支持的图片后缀（可根据需要添加）
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
# -----------------------------------------------------------

def sync_image_and_label():
    # 1. 获取所有 label 文件名（不含后缀）
    label_files = [
        os.path.splitext(f)[0]
        for f in os.listdir(LABEL_DIR)
        if f.endswith('.txt') and f != 'classes.txt'  # 排除 classes.txt
    ]
    label_set = set(label_files)
    print(f"✅ 找到 {len(label_set)} 个标注文件")

    # 2. 遍历图片，找出无对应标注的图片
    image_files = os.listdir(IMAGE_DIR)
    to_delete = []

    for img_file in image_files:
        if not img_file.lower().endswith(IMAGE_SUFFIXES):
            continue  # 跳过非图片文件
        img_name = os.path.splitext(img_file)[0]
        if img_name not in label_set:
            to_delete.append(img_file)

    if not to_delete:
        print("🎉 所有图片都有对应的标注文件，无需删除！")
        return

    # 3. 确认并删除多余图片
    print(f"⚠️  发现 {len(to_delete)} 张无对应标注的图片：")
    for f in to_delete:
        print(f"   - {f}")

    # 执行删除（可注释掉这部分先预览结果）
    for f in to_delete:
        img_path = os.path.join(IMAGE_DIR, f)
        os.remove(img_path)
        print(f"🗑️  已删除：{f}")

    print(f"\n✅ 清理完成！剩余图片数：{len(image_files) - len(to_delete)}")

if __name__ == '__main__':
    sync_image_and_label()