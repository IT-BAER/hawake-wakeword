
import onnx
import os
import sys

def check_opset(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    m = onnx.load(path)
    print(f"File: {path}")
    print(f"Opset: {m.opset_import[0].version}")

if len(sys.argv) > 1:
    check_opset(sys.argv[1])
else:
    base = r"d:\Software\openwakeword-train\openwakeword\openwakeword\resources\models"
    check_opset(os.path.join(base, "melspectrogram.onnx"))
    check_opset(os.path.join(base, "embedding_model.onnx"))
    check_opset(r"d:\Software\openwakeword-train\my_custom_model\hey_mycroft.onnx")
