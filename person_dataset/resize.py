import os
from PIL import Image


def crop_images_to_640(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理常见的图片格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                    # 计算横向和纵向可裁剪的数量（向下取整，保证无重叠）
                    num_cols = width // 1280
                    num_rows = height // 1280

                    # 逐块裁剪
                    for row in range(num_rows):
                        for col in range(num_cols):
                            # 计算裁剪区域坐标（无重叠）
                            left = col * 1280
                            upper = row * 1280
                            right = left + 1280
                            lower = upper + 1280

                            # 裁剪图片
                            cropped_img = img.crop((left, upper, right, lower))

                            # 生成输出文件名（原文件名_行_列.后缀）
                            base_name = os.path.splitext(filename)[0]
                            ext = os.path.splitext(filename)[1]
                            output_filename = f"{base_name}_{row}_{col}{ext}"
                            output_path = os.path.join(output_folder, output_filename)

                            # 保存裁剪后的图片
                            cropped_img.save(output_path)
                            print(f"✅ 已保存: {output_path}")
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")


# --------------------------
# 请修改为你的实际路径
# --------------------------
input_dir = r"D:\person\two"  # 你的原始图片文件夹路径
output_dir = r"D:\person\two_1280"  # 裁剪后图片的保存路径

# 执行裁剪
if __name__ == "__main__":
    crop_images_to_640(input_dir, output_dir)
    print("\n🎉 所有图片裁剪完成！")