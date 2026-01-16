import onnx
import os

# Load model with external data
model = onnx.load('my_custom_model/hey_schpustee.onnx', load_external_data=True)

# Downgrade IR version from 10 to 9
print(f"Original IR version: {model.ir_version}")
model.ir_version = 9
print(f"New IR version: {model.ir_version}")

# Save as single file with embedded data
output_path = 'my_custom_model/hey_schpustee_v9.onnx'
onnx.save_model(model, output_path, save_as_external_data=False)

print(f"Merged model saved: {output_path}")
print(f"Size: {os.path.getsize(output_path)} bytes")
