import urllib.request
import ssl

# 忽略SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context


# 定义进度显示函数
def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')


# 多个镜像源（按优先级排序）
urls = [
    "https://hub.gitmirror.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
    "https://gh.api.99988866.xyz/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
    "https://download.fastgit.org/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",  # 官方源备用
]

# 如果需要使用代理，取消下面的注释并设置正确的代理地址
# proxy_handler = urllib.request.ProxyHandler({
#     'http': 'http://127.0.0.1:7890',
#     'https': 'https://127.0.0.1:7890'
# })
# opener = urllib.request.build_opener(proxy_handler)
# urllib.request.install_opener(opener)

print("开始下载 YOLOv12s 模型...")
print("=" * 50)

success = False
for i, url in enumerate(urls, 1):
    try:
        print(f"\n尝试源 {i}/{len(urls)}: {url}")
        print("下载中...")

        urllib.request.urlretrieve(url, "yolo12s.pt", reporthook=show_progress)

        print("\n✅ 下载完成！")
        print(f"文件保存为: yolo12s.pt")
        success = True
        break

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        continue

if not success:
    print("\n" + "=" * 50)
    print("所有下载源均失败！")
    print("\n建议：")
    print("1. 检查网络连接")
    print("2. 使用浏览器手动下载：")
    print("   https://hub.gitmirror.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt")
    print("3. 或使用 YOLOv8s 替代：")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov8s.pt')\"")
else:
    # 验证下载的文件
    import os

    file_size = os.path.getsize("yolo12s.pt") / (1024 * 1024)
    print(f"\n文件大小: {file_size:.2f} MB")

    # 测试加载模型
    print("\n测试加载模型...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolo12s.pt")
        print("✅ 模型加载成功！可以开始训练了。")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}")