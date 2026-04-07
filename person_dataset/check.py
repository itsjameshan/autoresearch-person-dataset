import os

# 配置路径
VAL_IMG_DIR = r"D:\PythonProject\person_dataset\image\val"
VAL_LABEL_DIR = r"D:\PythonProject\person_dataset\labels\val"

# 支持的图片后缀
IMG_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')

# 1. 检查文件名匹配
img_files = [f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(IMG_SUFFIXES)]
label_files = [f for f in os.listdir(VAL_LABEL_DIR) if f.endswith('.txt') and f != 'classes.txt']

img_stems = {os.path.splitext(f)[0] for f in img_files}
label_stems = {os.path.splitext(f)[0] for f in label_files}

missing_labels = img_stems - label_stems
extra_labels = label_stems - img_stems

print(f"📊 验证集图片数量: {len(img_files)}")
print(f"📊 验证集标签数量: {len(label_files)}")
print(f"❌ 缺失标签的图片: {missing_labels if missing_labels else '无'}")
print(f"❌ 多余的标签: {extra_labels if extra_labels else '无'}")

# 2. 检查标签格式/内容
print("\n🔍 检查标签文件格式:")
error_count = 0
for label_file in label_files:
    label_path = os.path.join(VAL_LABEL_DIR, label_file)
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 检查标签是否为空
        if not lines:
            print(f"❌ 标签为空: {label_file}")
            error_count += 1
            continue
        # 检查每行格式
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # YOLO格式要求: class x_center y_center width height (5个值)
            if len(parts) != 5:
                print(f"❌ 格式错误: {label_file}，行内容: {line}")
                error_count += 1
                continue
            # 检查class是否为0
            if parts[0] != '0':
                print(f"❌ class错误: {label_file}，class值: {parts[0]} (应为0)")
                error_count += 1
            # 检查坐标是否在0-1之间
            coords = list(map(float, parts[1:]))
            for coord in coords:
                if coord < 0 or coord > 1:
                    print(f"❌ 坐标越界: {label_file}，坐标值: {coord} (应在0-1之间)")
                    error_count += 1
    except Exception as e:
        print(f"❌ 读取失败: {label_file}，错误: {e}")
        error_count += 1

print(f"\n✅ 标签校验完成，共发现 {error_count} 个错误")
if error_count == 0:
    print("🎉 所有标签文件完全符合YOLO规范！")