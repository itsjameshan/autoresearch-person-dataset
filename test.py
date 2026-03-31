import onnx
from onnx.external_data_helper import convert_model_to_external_data

model = onnx.load("person_count/weights/best.onnx")
onnx.save_model(model, "person_count/weights/best.onnx", save_as_external_data=False)