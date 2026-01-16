import os
import sys
import yaml
import subprocess
from pathlib import Path

# Force CPU to avoid CUDA kernel errors on RTX 5060 Ti
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configuration
target_word = 'hey mycroft'
number_of_examples = 1000 
number_of_training_steps = 1000
false_activation_penalty = 1500

python_exe = sys.executable

def run_command(cmd):
    print(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        # sys.exit(process.returncode)

def main():
    # 1. Prepare Paths
    project_root = Path(__file__).parent
    oww_dir = project_root / "openwakeword"
    
    # 2. Load and update config
    config_path = oww_dir / "examples" / "custom_model.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["target_phrase"] = [target_word]
    config["model_name"] = config["target_phrase"][0].replace(" ", "_")
    config["n_samples"] = number_of_examples
    config["n_samples_val"] = max(500, number_of_examples//10)
    config["steps"] = number_of_training_steps
    config["target_accuracy"] = 0.5
    config["target_recall"] = 0.25
    config["output_dir"] = str(project_root / "my_custom_model")
    config["max_negative_weight"] = false_activation_penalty

    # Set data paths relative to project root
    config["background_paths"] = [str(project_root / 'audioset_16k'), str(project_root / 'fma')]
    config["false_positive_validation_data_path"] = str(project_root / "validation_set_features.npy")
    config["feature_data_files"] = {"ACAV100M_sample": str(project_root / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")}

    # Update generation config if needed
    # The piper generator might fail on Windows if dependencies are missing.
    # We will try to run it but keep going.

    with open('my_model.yaml', 'w') as file:
        yaml.dump(config, file)

    print(f"Config generated for wake word: {target_word}")

    # 3. Training steps
    train_script = str(oww_dir / "openwakeword" / "train.py")
    
    print("\n--- Step 1: Generate Clips ---")
    run_command([python_exe, train_script, "--training_config", "my_model.yaml", "--generate_clips"])

    print("\n--- Step 2: Augment Clips ---")
    run_command([python_exe, train_script, "--training_config", "my_model.yaml", "--augment_clips"])

    print("\n--- Step 3: Train Model ---")
    run_command([python_exe, train_script, "--training_config", "my_model.yaml", "--train_model"])

    print("\n--- Step 4: Convert to TFLite (Optional) ---")
    onnx_path = Path(config["output_dir"]) / f"{config['model_name']}.onnx"
    if onnx_path.exists():
        # Using onnx2tf as in the colab script
        run_command(f"onnx2tf -i {onnx_path} -o {config['output_dir']} -kat onnx____Flatten_0")
        # Rename if needed (logic from colab)
        float32_tflite = Path(config["output_dir"]) / f"{config['model_name']}_float32.tflite"
        final_tflite = Path(config["output_dir"]) / f"{config['model_name']}.tflite"
        if float32_tflite.exists():
            if final_tflite.exists():
                os.remove(final_tflite)
            os.rename(float32_tflite, final_tflite)
            print(f"TFLite model created at {final_tflite}")

    print("\nTraining process finished.")

if __name__ == "__main__":
    main()
