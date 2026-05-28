import os
import sys
import struct
import io

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BAT_PATH = os.path.join(REPO_ROOT, "start_all.bat")
ICO_PATH = os.path.join(REPO_ROOT, "autoresearch.ico")

def create_ico():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[WARN] Pillow not found — installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
        from PIL import Image, ImageDraw, ImageFont

    sizes = [256, 128, 64, 48, 32, 16]
    images = []

    for s in sizes:
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        margin = s * 0.08
        r = (s - 2 * margin) / 2
        cx, cy = s / 2, s / 2

        # ── Outer glow ring ──
        for i in range(3):
            rr = r + i * (s * 0.02)
            alpha = 40 - i * 12
            draw.ellipse(
                [cx - rr, cy - rr, cx + rr, cy + rr],
                fill=(99, 102, 241, max(alpha, 8)),
            )

        # ── Main circle with gradient (simulated via layered circles) ──
        steps = max(6, int(r))
        for i in range(steps, -1, -1):
            frac = i / steps
            rr = r * frac
            # blue → violet gradient
            r_col = int(59 + frac * 80)
            g_col = int(30 + frac * 50)
            b_col = int(180 - frac * 40)
            a = int(220 + frac * 35)
            draw.ellipse(
                [cx - rr, cy - rr, cx + rr, cy + rr],
                fill=(r_col, g_col, b_col, a),
            )

        # ── Inner highlight ──
        hl_r = r * 0.28
        draw.ellipse(
            [cx - hl_r * 1.5, cy - r * 0.6, cx - hl_r * 0.3, cy - r * 0.6 + hl_r * 2],
            fill=(255, 255, 255, 50),
        )

        # ── Text: "AR" ──
        text = "AR"
        font_size_ratio = 0.38
        for fname in ("segoeuib.ttf", "segoeui.ttf", "Segoe UI.ttf",
                       "arialbd.ttf", "Arial Bold.ttf", "arial.ttf", "Arial.ttf"):
            try:
                font = ImageFont.truetype(fname, max(8, int(s * font_size_ratio)))
                break
            except (IOError, OSError):
                font = None
        if font is None:
            try:
                font = ImageFont.truetype("courbd.ttf", max(8, int(s * font_size_ratio * 0.8)))
            except (IOError, OSError):
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx - tw / 2
        ty = cy - th / 2 - s * 0.02

        draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0, 80), font=font)
        draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)

        # ── Small gear dots around the top-right ──
        gear_cx = cx + r * 0.42
        gear_cy = cy - r * 0.38
        gear_r = r * 0.18
        num_teeth = 6
        import math
        for tooth in range(num_teeth):
            angle = tooth * (2 * math.pi / num_teeth) - math.pi / 2
            tx_dot = gear_cx + math.cos(angle) * gear_r
            ty_dot = gear_cy + math.sin(angle) * gear_r
            dot_r = max(2, s * 0.03)
            draw.ellipse(
                [tx_dot - dot_r, ty_dot - dot_r, tx_dot + dot_r, ty_dot + dot_r],
                fill=(255, 255, 255, 200),
            )

        images.append(img)

    # ── Save as .ico ──
    images[0].save(
        ICO_PATH,
        format="ICO",
        sizes=[(sz, sz) for sz in sizes],
        append_images=images[1:],
    )
    print(f"[OK] Icon saved: {ICO_PATH}")
    return ICO_PATH


def create_shortcut():
    import subprocess

    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    lnk_path = os.path.join(desktop, "AutoResearch.lnk")

    ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{lnk_path}")
$Shortcut.TargetPath = "{BAT_PATH}"
$Shortcut.WorkingDirectory = "{REPO_ROOT}"
$Shortcut.IconLocation = "{ICO_PATH},0"
$Shortcut.Description = "AutoResearch v2 — Stadium Crowd Detection One-Click Launcher"
$Shortcut.WindowStyle = 1
$Shortcut.Save()
Write-Host "[OK] Shortcut created: {lnk_path}"
'''

    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        return False
    return True


def main():
    print("=" * 55)
    print("  AutoResearch Desktop Shortcut Setup")
    print("=" * 55)

    if not os.path.exists(BAT_PATH):
        print(f"[ERROR] start_all.bat not found at {BAT_PATH}")
        sys.exit(1)

    create_ico()
    ok = create_shortcut()

    if ok:
        print()
        print("=" * 55)
        print("  DONE — Double-click 'AutoResearch' on your Desktop!")
        print("=" * 55)
    else:
        print()
        print("[ERROR] Failed to create shortcut. Try running as Administrator.")


if __name__ == "__main__":
    main()