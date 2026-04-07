# -*- coding: utf-8 -*-
"""
从仓库根目录启动检测服务（与在 person_dataset 内运行等价）。

数据与 ONNX 默认目录仍在 person_dataset/（onnx_data、result、images 等）。
用法: 在 autoresearch_new 下执行  python app_win.py
"""
from __future__ import annotations

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_ROOT, "person_dataset", "app_win.py")

if not os.path.isfile(_TARGET):
    print(f"[ERROR] 未找到 {_TARGET}")
    sys.exit(1)

if __name__ == "__main__":
    sys.argv[0] = _TARGET
    runpy.run_path(_TARGET, run_name="__main__")
