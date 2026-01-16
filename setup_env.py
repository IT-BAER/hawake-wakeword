import os
import subprocess
import sys
from pathlib import Path

def run_command(command, shell=True):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=shell, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result

def setup():
    # 1. Clone repositories
    if not os.path.exists("piper-sample-generator"):
        run_command("git clone https://github.com/rhasspy/piper-sample-generator")
        run_command("cd piper-sample-generator && git checkout 213d4d5")

    if not os.path.exists("openwakeword"):
        run_command("git clone https://github.com/dscripka/openwakeword")

    # 2. Download essential models
    models_dir = Path("openwakeword/openwakeword/resources/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
    models = [
        "embedding_model.onnx",
        "embedding_model.tflite",
        "melspectrogram.onnx",
        "melspectrogram.tflite"
    ]
    
    for model in models:
        model_path = models_dir / model
        if not model_path.exists():
            print(f"Downloading {model}...")
            run_command(f"curl -L {base_url}{model} -o {model_path}")

    # Piper generator model
    piper_model_dir = Path("piper-sample-generator/models")
    piper_model_dir.mkdir(parents=True, exist_ok=True)
    piper_model = piper_model_dir / "en_US-libritts_r-medium.pt"
    if not piper_model.exists():
        print("Downloading piper model...")
        run_command(f"curl -L https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt -o {piper_model}")

    print("Setup complete.")

if __name__ == "__main__":
    setup()
