"""
generate_overseer_icon.py — 生成智能体总管图标
"""

import os, struct, math
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ICO = os.path.join(PROJECT_ROOT, "overseer.ico")
SIZES = [256, 128, 64, 48, 32, 16]


def draw_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2
    r = size * 0.4

    bg = (15, 25, 55, 255)
    acc = (80, 160, 255, 255)
    white = (230, 240, 255, 255)
    dark = (8, 18, 40, 255)

    draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], fill=bg)

    ring_w = max(2, int(size * 0.045))
    draw.ellipse(
        [cx - r + ring_w, cy - r + ring_w, cx + r - ring_w, cy + r - ring_w],
        fill=None, outline=acc, width=ring_w
    )

    pupil_r = r * 0.32
    px = cx - r * 0.12
    py = cy - r * 0.16
    draw.ellipse([px - pupil_r, py - pupil_r, px + pupil_r, py + pupil_r], fill=acc)
    draw.ellipse(
        [px - pupil_r * 0.55, py - pupil_r * 0.55,
         px + pupil_r * 0.55, py + pupil_r * 0.55],
        fill=dark
    )
    hl_r = pupil_r * 0.18
    draw.ellipse(
        [px + pupil_r * 0.25 - hl_r, py - pupil_r * 0.3 - hl_r,
         px + pupil_r * 0.25 + hl_r, py - pupil_r * 0.3 + hl_r],
        fill=white
    )

    cdr = r * 0.16
    draw.ellipse([cx - cdr, cy - cdr, cx + cdr, cy + cdr], fill=white)
    draw.ellipse(
        [cx - cdr * 0.55, cy - cdr * 0.55, cx + cdr * 0.55, cy + cdr * 0.55],
        fill=acc
    )

    seg_count = 4
    for i in range(seg_count):
        a = math.radians(i * 90 - 45)
        sx = cx + r * 0.75 * math.cos(a)
        sy = cy + r * 0.75 * math.sin(a)
        sr = max(2, int(size * 0.05))
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=acc)

    for i in range(seg_count):
        a = math.radians(i * 90 + 45 - 45)
        sx = cx + r * 0.94 * math.cos(a)
        sy = cy + r * 0.94 * math.sin(a)
        sr = max(1.5, int(size * 0.025))
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=acc)

    return img


def save_ico(images_dict, path):
    items = []
    img_data_blocks = []
    for sz in sorted(images_dict.keys()):
        img = images_dict[sz]
        rgba = img.tobytes("raw", "BGRA")
        items.append((sz, rgba))
        img_data_blocks.append(rgba)

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
    print("生成智能体总管图标...")
    imgs = {}
    for sz in SIZES:
        imgs[sz] = draw_icon(sz)
    save_ico(imgs, OUTPUT_ICO)
    print(f"完成: {OUTPUT_ICO}")