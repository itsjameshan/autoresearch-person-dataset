"""
install_desktop_shortcut.py — 一键安装桌面快捷方式 (.lnk 版, 支持自定义图标)
"""

import os, sys, subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = r"D:\Python 3.12.0\python.exe"
ICO_PATH = os.path.join(PROJECT_ROOT, "overseer.ico")

if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = "python"

try:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(New-Object -ComObject Scripting.FileSystemObject).GetFile('{PYTHON_EXE}').ShortPath"],
        capture_output=True, text=True, timeout=5
    )
    short = result.stdout.strip()
    if short and os.path.exists(short):
        PYTHON_EXE = short
except Exception:
    pass


def create_shortcut():
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    lnk_path = os.path.join(desktop, "智能体总管.lnk")

    ps = f"""
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("{lnk_path}")
$s.TargetPath = "{PYTHON_EXE}"
$s.Arguments = "agent_overseer.py"
$s.WorkingDirectory = "{PROJECT_ROOT}"
$s.IconLocation = "{ICO_PATH},0"
$s.WindowStyle = 1
$s.Description = "智能体总管 — 一键启动全部AI智能体训练YOLO12"
$s.Save()
Write-Output "OK"
"""

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True, text=True, timeout=10
    )
    return result.returncode == 0, lnk_path


def main():
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")

    print("=" * 55)
    print("  智能体总管 · 桌面快捷方式安装器")
    print("=" * 55)

    ok, lnk_path = create_shortcut()
    if ok:
        print(f"\n  ✅ 桌面快捷方式已创建")
        print(f"     位置: {lnk_path}")
        print(f"     图标: {ICO_PATH}")
        print(f"\n  双击桌面上「智能体总管」即可启动")
        print(f"  浏览器会自动打开 http://localhost:5050")
    else:
        print("\n  ❌ 创建失败")

    for name in ["智能体总管.vbs", "智能体总管.bat"]:
        p = os.path.join(desktop, name)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"  已清理旧文件: {name}")
            except Exception:
                pass

    print("=" * 55)


if __name__ == "__main__":
    main()