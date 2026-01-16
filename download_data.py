import os
import subprocess
from pathlib import Path

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

def download():
    # Feature files
    features = [
        ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
        ("validation_set_features.npy", "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy")
    ]
    
    for name, url in features:
        if not os.path.exists(name):
            print(f"Downloading {name} (this may take a while)...")
            run_command(f"curl -L {url} -o {name}")
        else:
            print(f"{name} already exists.")

    # Note: audioset and fma downloads are handled in the Colab script logic 
    # but for local run, we might want to ensure the directories exist even if empty
    # to avoid immediate crashes, though the train.py script handles generation.
    os.makedirs("audioset_16k", exist_ok=True)
    os.makedirs("fma", exist_ok=True)

if __name__ == "__main__":
    download()
