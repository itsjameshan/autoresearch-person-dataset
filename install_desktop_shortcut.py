"""
install_desktop_shortcut.py — 一键安装桌面快捷方式 (VBS 版)
"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = r"D:\Python 3.12.0\python.exe"

if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = "python"

try:
    import subprocess
    short_path = subprocess.check_output(
        ['powershell', '-NoProfile', '-Command',
         '(New-Object -ComObject Scripting.FileSystemObject).GetFile(\''
         + PYTHON_EXE + '\').ShortPath'],
        text=True, timeout=5
    ).strip()
    if short_path and os.path.exists(short_path):
        PYTHON_EXE = short_path
except Exception:
    pass

VBS_CONTENT = f'''Set WshShell = CreateObject("WScript.Shell")
projectDir = "{PROJECT_ROOT}"
pythonExe = "{PYTHON_EXE}"
If Not CreateObject("Scripting.FileSystemObject").FileExists(pythonExe) Then
    pythonExe = "python"
End If
WshShell.CurrentDirectory = projectDir
WshShell.Run "cmd /k " & """" & pythonExe & """ agent_overseer.py", 1, False
Set WshShell = Nothing
'''


def main():
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    vbs_path = os.path.join(desktop, "智能体总管.vbs")

    print("=" * 55)
    print("  智能体总管 · 桌面快捷方式安装器")
    print("=" * 55)

    with open(vbs_path, "w", encoding="utf-8") as f:
        f.write(VBS_CONTENT)
    print(f"\n  ✅ 桌面快捷方式已创建")
    print(f"     位置: {vbs_path}")
    print(f"     Python: {PYTHON_EXE}")
    print(f"\n  双击桌面上「智能体总管.vbs」即可启动")
    print(f"  浏览器会自动打开 http://localhost:5050")
    print("=" * 55)

    old_lnk = os.path.join(desktop, "智能体总管.lnk")
    old_bat = os.path.join(desktop, "智能体总管.bat")
    for p in [old_lnk, old_bat]:
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"  已清理旧文件: {os.path.basename(p)}")
            except Exception:
                pass


if __name__ == "__main__":
    main()