
import onnx
from onnx import version_converter
import os

def convert_model(input_path, output_path, target_opset=11):
    print(f"Converting {input_path} to Opset {target_opset}")
    try:
        model = onnx.load(input_path)
        print(f"Original Opset: {model.opset_import[0].version}")
        
        # Check if already at target opset
        if model.opset_import[0].version == target_opset:
            print(f"Model is already Opset {target_opset}. Copying...")
            with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            return True

        converted_model = version_converter.convert_version(model, target_opset)
        onnx.save(converted_model, output_path)
        print(f"Successfully converted and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to convert {input_path}: {e}")
        return False

# Paths
base_dir = r"d:\Software\openwakeword-train\openwakeword\openwakeword\resources\models"
mel_input = os.path.join(base_dir, "melspectrogram.onnx")
emb_input = os.path.join(base_dir, "embedding_model.onnx")

mel_output = r"d:\Software\openwakeword-train\melspectrogram_opset10.onnx"
emb_output = r"d:\Software\openwakeword-train\embedding_model_opset10.onnx"

# Run conversion
convert_model(mel_input, mel_output)
convert_model(emb_input, emb_output)
