import os
import sys

import onnx
from onnx.external_data_helper import convert_model_to_external_data

DEFAULT_ONNX = "person_count/weights/best.onnx"
onnx_path = os.environ.get("ONNX_PATH", DEFAULT_ONNX)
if not os.path.isfile(onnx_path):
    print(
        f"[ERROR] 找不到 ONNX 文件: {onnx_path}\n"
        f"        请通过 ONNX_PATH 环境变量指定路径，"
        f"或将文件放至默认位置 ({DEFAULT_ONNX})。",
        file=sys.stderr,
    )
    sys.exit(1)

model = onnx.load(onnx_path)
onnx.save_model(model, onnx_path, save_as_external_data=False)
