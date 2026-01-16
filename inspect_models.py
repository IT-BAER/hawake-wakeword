
import onnxruntime as ort
import os

model_path = r"d:\Software\openwakeword-train\openwakeword\openwakeword\resources\models\melspectrogram.onnx"
sess = ort.InferenceSession(model_path)

print(f"--- Inspecting {os.path.basename(model_path)} ---")
for i in sess.get_inputs():
    print(f"Input: {i.name}, Shape: {i.shape}, Type: {i.type}")
for o in sess.get_outputs():
    print(f"Output: {o.name}, Shape: {o.shape}, Type: {o.type}")

emb_path = r"d:\Software\openwakeword-train\openwakeword\openwakeword\resources\models\embedding_model.onnx"
sess_emb = ort.InferenceSession(emb_path)
print(f"\n--- Inspecting {os.path.basename(emb_path)} ---")
for i in sess_emb.get_inputs():
    print(f"Input: {i.name}, Shape: {i.shape}, Type: {i.type}")
for o in sess_emb.get_outputs():
    print(f"Output: {o.name}, Shape: {o.shape}, Type: {o.type}")
