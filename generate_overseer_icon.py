"""
generate_overseer_icon.py — 生成智能体总管图标 (v2 神经网络风格)
"""

import os, struct, math
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ICO = os.path.join(PROJECT_ROOT, "overseer.ico")
SIZES = [256, 128, 64, 48, 32, 16]


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,)


def draw_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2

    bg_dark = (13, 17, 35)
    bg_light = (18, 24, 55)
    accent_cyan = (64, 210, 255)
    accent_purple = (140, 100, 255)
    node_fill = (45, 180, 240)
    line_color = (50, 160, 240, 180)
    glow_cyan = (64, 210, 255, 30)

    half = size / 2
    r = size * 0.42
    corner_r = size * 0.12

    draw.rounded_rectangle(
        [cx - r, cy - r, cx + r, cy + r],
        radius=int(corner_r),
        fill=bg_dark
    )

    inner_margin = size * 0.03
    draw.rounded_rectangle(
        [cx - r + inner_margin, cy - r + inner_margin,
         cx + r - inner_margin, cy + r - inner_margin],
        radius=int(corner_r * 0.85),
        fill=None,
        outline=lerp_color(accent_cyan, accent_purple, 0.4)[:3] + (80,),
        width=max(1, int(size * 0.022))
    )

    node_count = 6
    hub_r = r * 0.25
    node_dist = r * 0.72
    node_small_r = max(1.5, size * 0.04)
    hub_dot_r = max(2, size * 0.065)

    import random
    random.seed(42)

    for i in range(node_count):
        angle = math.radians(i * (360 / node_count) - 90)
        nx = cx + node_dist * math.cos(angle)
        ny = cy + node_dist * math.sin(angle)

        t = i / node_count
        lc = lerp_color(accent_cyan, accent_purple, t)
        draw.line(
            [cx, cy, nx, ny],
            fill=lc[:3] + (120,),
            width=max(1, int(size * 0.025))
        )

    for i in range(node_count):
        angle = math.radians(i * (360 / node_count) - 90)
        nx = cx + node_dist * math.cos(angle)
        ny = cy + node_dist * math.sin(angle)
        t = i / node_count
        nc = lerp_color(accent_cyan, accent_purple, t)

        draw.ellipse(
            [nx - node_small_r * 1.6, ny - node_small_r * 1.6,
             nx + node_small_r * 1.6, ny + node_small_r * 1.6],
            fill=glow_cyan[:3] + (25,)
        )
        draw.ellipse(
            [nx - node_small_r, ny - node_small_r,
             nx + node_small_r, ny + node_small_r],
            fill=nc[:3] + (230,)
        )

    draw.ellipse(
        [cx - hub_dot_r * 2.0, cy - hub_dot_r * 2.0,
         cx + hub_dot_r * 2.0, cy + hub_dot_r * 2.0],
        fill=(64, 210, 255, 35)
    )
    draw.ellipse(
        [cx - hub_dot_r, cy - hub_dot_r,
         cx + hub_dot_r, cy + hub_dot_r],
        fill=(255, 255, 255, 255)
    )
    draw.ellipse(
        [cx - hub_dot_r * 0.55, cy - hub_dot_r * 0.55,
         cx + hub_dot_r * 0.55, cy + hub_dot_r * 0.55],
        fill=accent_cyan + (220,)
    )

    return img


def save_ico(images_dict, path):
    items = []
    for sz in sorted(images_dict.keys()):
        img = images_dict[sz]
        rgba = img.tobytes("raw", "BGRA")
        items.append((sz, rgba))

    header = struct.pack("<HHH", 0, 1, len(items))

    dir_entries = b""
    data_offset = 6 + 16 * len(items)
    for sz, _ in items:
        entry_sz = sz if sz < 256 else 0
        img_size = sz * sz * 4 + 40 + sz * sz // 8
        dir_entries += struct.pack("<BBBBHHII", entry_sz, entry_sz, 0, 0, 1, 32, img_size, data_offset)
        data_offset += img_size

    data = b""
    for sz, rgba in items:
        bmp_header = struct.pack("<IiiHHIIIIII", 40, sz, sz * 2, 1, 32, 0, sz * sz * 4, 0, 0, 0, 0)
        and_mask = b"\xff" * (sz * sz // 8)
        data += bmp_header + rgba + and_mask

    with open(path, "wb") as f:
        f.write(header + dir_entries + data)
    return True


if __name__ == "__main__":
    print("生成智能体总管图标 (v2 网络拓扑风格)...")
    imgs = {}
    for sz in SIZES:
        imgs[sz] = draw_icon(sz)
    save_ico(imgs, OUTPUT_ICO)
    print(f"完成: {OUTPUT_ICO}")