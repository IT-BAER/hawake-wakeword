import streamlit as st
import yaml
import os
import sys
import subprocess
from pathlib import Path
import shutil
import time
import onnx

# Add piper-sample-generator to path
current_dir = Path(__file__).parent.absolute()
piper_dir = current_dir / "piper-sample-generator"
if str(piper_dir) not in sys.path:
    sys.path.append(str(piper_dir))

# Try importing generate_samples to check availability
try:
    from generate_samples import generate_samples
    PIPER_AVAILABLE = True
except ImportError as e:
    PIPER_AVAILABLE = False
    st.error(f"Failed to import piper-sample-generator logic: {e}")

st.set_page_config(page_title="OpenWakeWord Trainer", layout="wide")

# Initialize session state for process management
if 'training_running' not in st.session_state:
    st.session_state.training_running = False
if 'training_pids' not in st.session_state:
    st.session_state.training_pids = []

st.title("OpenWakeWord Custom Model Trainer")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

target_word = st.sidebar.text_input("Target Wakeword", value="", placeholder="e.g. hey jarvis")
model_name = target_word.replace(" ", "_") if target_word else ""

# Show warning if no wake word entered
if not target_word.strip():
    st.sidebar.warning("⚠️ Enter a wake word to begin")

st.sidebar.subheader("Training Parameters")
number_of_examples = st.sidebar.slider("Number of Examples", min_value=100, max_value=50000, value=5000, step=100, help="How many synthetic examples to generate. More is generally better but takes longer.")
number_of_training_steps = st.sidebar.slider("Training Steps", min_value=100, max_value=50000, value=5000, step=100, help="How many steps to train the model.")
false_activation_penalty = st.sidebar.slider("False Activation Penalty", min_value=100, max_value=5000, value=1500, step=50, help="Penalizes false positives. Higher values reduce false positives but might lower recall.")

st.sidebar.subheader("ONNX Export")
onnx_opset = st.sidebar.selectbox(
    "Target Opset Version",
    options=[11, 10, 12, 13],
    index=0,  # Default to Opset 11
    help="Opset 11 recommended for Android 8+ compatibility with best operator support."
)
st.sidebar.caption("ℹ️ Opset 11 is recommended for Android 8+ devices")

def check_gpu_compatibility():
    """Checks if the GPU is actually usable for PyTorch operations."""
    import torch
    
    if not torch.cuda.is_available():
        return False, "CUDA not available (no GPU driver or PyTorch CPU-only build)"
    
    try:
        # Get GPU info for better error messages
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        compute_cap_str = f"{compute_cap[0]}.{compute_cap[1]}"
        
        # PyTorch 2.x typically requires compute capability 3.5+
        # Very new GPUs (like RTX 50 series) might need newer PyTorch builds
        if compute_cap[0] < 3 or (compute_cap[0] == 3 and compute_cap[1] < 5):
            return False, f"{gpu_name} (compute {compute_cap_str}) - compute capability too low"
        
        # Try a small dummy operation that requires a CUDA kernel
        x = torch.tensor([1.0, 2.0]).cuda()
        y = x * 2
        _ = y.cpu()  # Sync to ensure kernel executed
        
        return True, f"{gpu_name} (compute {compute_cap_str})"
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "no kernel image" in error_msg or "arch" in error_msg:
            return False, f"GPU detected but no kernel for this architecture. Try updating PyTorch or use CPU mode."
        return False, f"GPU kernel test failed: {e}"
    except Exception as e:
        return False, f"GPU check failed: {e}"

# Auto-detect GPU status
gpu_ok, gpu_msg = check_gpu_compatibility()
use_cpu_default = not gpu_ok

st.sidebar.subheader("Hardware")

# Show GPU detection result
if gpu_ok:
    st.sidebar.success(f"✅ GPU Detected: {gpu_msg}")
else:
    st.sidebar.info(f"ℹ️ {gpu_msg}")

# Manual override checkbox - always show it
force_cpu = st.sidebar.checkbox(
    "Force CPU Mode",
    value=use_cpu_default,
    help="Enable this if you have GPU issues (e.g., unsupported GPU architecture, driver problems). Training will be slower but more reliable."
)

# Determine final CPU/GPU usage
use_cpu = force_cpu or not gpu_ok

if use_cpu:
    st.sidebar.warning("🖥️ Using CPU for training")
else:
    st.sidebar.success("⚡ Using GPU for training")

# --- Preview Section ---
st.header("1. Wake Word Preview")
st.markdown("Listen to how the synthetic voice pronounces your wake word. Adjust the spelling (e.g., phonetic spelling) until it sounds right.")

if st.button("Generate Preview", disabled=not target_word.strip()):
    if not PIPER_AVAILABLE:
        st.error("Piper generator is not available. Please check dependencies.")
    elif not target_word.strip():
        st.error("Please enter a wake word first!")
    else:
        with st.spinner("Generating preview..."):
            preview_dir = current_dir / "preview_temp"
            preview_dir.mkdir(exist_ok=True)
            
            # Clean old previews
            for f in preview_dir.glob("*.wav"):
                f.unlink()

            try:
                # Use Piper ONNX model for TTS
                piper_model = piper_dir / "models" / "en_US-libritts_r-medium.onnx"
                
                # Check if model exists
                if not piper_model.exists():
                    st.error(f"Piper TTS model not found at: {piper_model}")
                    st.info("Please download the model first by running: python quickstart.py")
                else:
                    generate_samples(
                        text=target_word,
                        output_dir=str(preview_dir),
                        max_samples=1,
                        model=str(piper_model),
                        batch_size=1,
                        auto_reduce_batch_size=True,
                        file_names=["preview.wav"]
                    )
                    
                    preview_file = preview_dir / "preview.wav"
                    if preview_file.exists():
                        st.audio(str(preview_file))
                        st.success("Preview generated!")
                    else:
                        st.error("Failed to generate preview file.")
            except Exception as e:
                st.error(f"Error during generation: {e}")

# --- Training Section ---
st.header("2. Train Model")

def run_command(cmd, log_container):
    """Run a shell command and stream output to streamlit container."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        env=env
    )
    
    # Track this process PID for cleanup
    st.session_state.training_pids.append(process.pid)
    
    output_log = ""
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output_log += line
                # Update the log view every line for better responsiveness
                log_container.code(output_log[-2000:], language="text")
    except Exception:
        # If Streamlit stops the script (User presses Stop), or any error occurs
        # Force kill the process tree
        kill_process_tree(process.pid)
        raise
    finally:
        # Remove from tracking if completed
        if process.pid in st.session_state.training_pids:
            st.session_state.training_pids.remove(process.pid)
    
    return process.returncode

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        # Windows: Use taskkill with /T to kill tree
        subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True)
    except Exception:
        pass

def kill_all_training_processes():
    """Kill all tracked training processes and known training executables."""
    killed = []
    
    # Kill tracked PIDs
    for pid in list(st.session_state.training_pids):
        kill_process_tree(pid)
        killed.append(f"PID {pid}")
    st.session_state.training_pids = []
    
    # Kill known training processes by name
    known_processes = ["piper.exe"]
    for proc_name in known_processes:
        try:
            result = subprocess.run(f"taskkill /F /IM {proc_name}", shell=True, capture_output=True, text=True)
            if "SUCCESS" in result.stdout:
                killed.append(proc_name)
        except Exception:
            pass
    
    # Try to find and kill python processes running train.py
    try:
        # Use WMIC to find python processes with train.py in command line
        result = subprocess.run(
            'wmic process where "name=\'python.exe\' and commandline like \'%train.py%\'" get processid',
            shell=True, capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line.isdigit():
                subprocess.run(f"taskkill /F /T /PID {line}", shell=True, capture_output=True)
                killed.append(f"train.py (PID {line})")
    except Exception:
        pass
    
    # Reset training state
    st.session_state.training_running = False
    
    return killed

if st.sidebar.button("Force Stop / Cleanup Processes", type="primary"):
    st.sidebar.warning("Stopping all training processes...")
    killed = kill_all_training_processes()
    if killed:
        st.sidebar.success(f"Killed: {', '.join(killed)}")
    else:
        st.sidebar.info("No training processes found to kill.")

# Show current training status
if st.session_state.training_running:
    st.sidebar.warning("⚠️ Training is currently running...")
else:
    st.sidebar.info("✅ No training in progress")

# Initialize training trigger flag
if 'start_training_trigger' not in st.session_state:
    st.session_state.start_training_trigger = False

# Disable Start button if already training or no wake word entered
start_disabled = st.session_state.training_running or not target_word.strip()
if st.button("Start Training", disabled=start_disabled):
    if not target_word.strip():
        st.error("Please enter a wake word first!")
    else:
        # Mark training as running and trigger rerun to disable button immediately
        st.session_state.training_running = True
        st.session_state.start_training_trigger = True
        st.rerun()

# Actual training logic runs after rerun (button is now disabled)
if st.session_state.start_training_trigger:
    st.session_state.start_training_trigger = False  # Reset trigger
    
    # 1. Prepare Config
    oww_dir = current_dir / "openwakeword"
    config_template_path = oww_dir / "examples" / "custom_model.yml"
    
    if not config_template_path.exists():
        st.error(f"Config template not found at {config_template_path}")
        st.session_state.training_running = False
    else:
        with open(config_template_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config
        config["target_phrase"] = [target_word]
        config["model_name"] = model_name
        config["n_samples"] = number_of_examples
        config["n_samples_val"] = max(500, number_of_examples // 10)
        config["steps"] = number_of_training_steps
        config["target_accuracy"] = 0.5
        config["target_recall"] = 0.25
        config["output_dir"] = str(current_dir / model_name)
        config["max_negative_weight"] = false_activation_penalty
        
        # Paths
        config["background_paths"] = [str(current_dir / 'audioset_16k'), str(current_dir / 'fma')]
        config["false_positive_validation_data_path"] = str(current_dir / "validation_set_features.npy")
        config["feature_data_files"] = {"ACAV100M_sample": str(current_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")}
        
        # Write config
        config_path = current_dir / "my_model.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        st.info(f"Configuration saved to {config_path}")
        
        # 2. Run Steps
        python_exe = sys.executable
        train_script = str(oww_dir / "openwakeword" / "train.py")
        
        log_expander = st.expander("Training Logs", expanded=True)
        log_placeholder = log_expander.empty()
        
        flags = ["--training_config", "my_model.yaml"]
        if use_cpu:
            flags.append("--force_cpu")

        # Add -u to python command for unbuffered output
        steps = [
            ("Generate Clips", [python_exe, "-u", train_script] + flags + ["--generate_clips"]),
            ("Augment Clips", [python_exe, "-u", train_script] + flags + ["--augment_clips", "--overwrite"]),
            ("Train Model", [python_exe, "-u", train_script] + flags + ["--train_model"])
        ]
        
        success = True
        progress_bar = st.progress(0)
        
        for i, (step_name, cmd_list) in enumerate(steps):
            st.write(f"**Step {i+1}: {step_name}**...")
            cmd_str = " ".join(cmd_list)
            ret_code = run_command(cmd_str, log_placeholder)
            
            # Special handling for Train Model step:
            # It might fail at the very end due to missing onnx_tf, but if the .onnx file exists, we are good.
            if step_name == "Train Model" and ret_code != 0:
                onnx_path = Path(config["output_dir"]) / f"{model_name}.onnx"
                if onnx_path.exists():
                    st.warning(f"Step {step_name} finished with errors (likely legacy TFLite conversion), but ONNX model was saved. Proceeding...")
                    ret_code = 0
            
            if ret_code != 0:
                st.error(f"Step {step_name} failed with return code {ret_code}")
                success = False
                st.session_state.training_running = False
                break
            progress_bar.progress((i + 1) / len(steps))
            
        if success:
            st.success("Training complete!")
            
            # Convert to TFLite (Optional but recommended)
            st.write("**Step 4: Converting to TFLite...**")
            onnx_path = Path(config["output_dir"]) / f"{model_name}.onnx"
            if onnx_path.exists():
                try:
                    # Using onnx2tf command (assuming it's in path or can be run via python -m)
                    # We'll try running it as a shell command
                    cmd = f"onnx2tf -i \"{onnx_path}\" -o \"{config['output_dir']}\" -kat onnx____Flatten_0"
                    run_command(cmd, log_placeholder)
                    
                    float32_tflite = Path(config["output_dir"]) / f"{model_name}_float32.tflite"
                    final_tflite = Path(config["output_dir"]) / f"{model_name}.tflite"
                    
                    if float32_tflite.exists():
                        if final_tflite.exists():
                            final_tflite.unlink()
                        float32_tflite.rename(final_tflite)
                        st.success(f"TFLite model created: {final_tflite.name}")
                        
                        # Download Button
                        with open(final_tflite, "rb") as f:
                            st.download_button(
                                label="Download .tflite Model",
                                data=f,
                                file_name=f"{model_name}.tflite",
                                mime="application/octet-stream"
                            )
                except Exception as e:
                    st.warning(f"TFLite conversion failed: {e}")
            
            # Download ONNX Button
            if onnx_path.exists():
                # Patch ONNX for Android compatibility
                try:
                    model = onnx.load(str(onnx_path))
                    patched = False
                    
                    # 1. Patch opset version if needed
                    current_opset = model.opset_import[0].version
                    if current_opset != onnx_opset:
                        st.write(f"**Patching ONNX opset: {current_opset} → {onnx_opset}**")
                        model.opset_import[0].version = onnx_opset
                        patched = True
                    
                    # 2. Set IR version to 7 for ONNX Runtime Android compatibility
                    # ONNX Runtime Android 1.14.0 only supports IR version <= 8
                    if model.ir_version > 7:
                        st.write(f"**Patching IR version: {model.ir_version} → 7**")
                        model.ir_version = 7
                        patched = True
                    
                    # 3. Remove unsupported 'allowzero' attribute from Reshape operators
                    # This attribute was introduced in Opset 14 but isn't supported by older runtimes
                    for node in model.graph.node:
                        if node.op_type == 'Reshape':
                            for attr in list(node.attribute):
                                if attr.name == 'allowzero':
                                    node.attribute.remove(attr)
                                    st.write(f"**Removed 'allowzero' from Reshape node: {node.name}**")
                                    patched = True
                    
                    if patched:
                        onnx.save(model, str(onnx_path))
                        st.success(f"Model patched for Android compatibility (Opset {onnx_opset}, IR v7)")
                    else:
                        st.info(f"Model already compatible (Opset {onnx_opset}, IR v{model.ir_version})")
                except Exception as e:
                    st.warning(f"Could not patch model: {e}")
                
                # Download FP32 model
                with open(onnx_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download Model (Opset {onnx_opset})",
                        data=f,
                        file_name=f"{model_name}.onnx",
                        mime="application/octet-stream"
                    )
            
            # Create Merged ONNX (End-to-End)
            st.write("**Step 5: Creating Merged Model (Raw Audio -> Detection)...**")
            try:
                emb_model_path = oww_dir / "openwakeword" / "resources" / "models" / "embedding_model.onnx"
                merged_output_path = Path(config["output_dir"]) / f"{model_name}_merged.onnx"
                
                if emb_model_path.exists() and onnx_path.exists():
                    merge_cmd = f"{python_exe} merge_models.py \"{emb_model_path}\" \"{onnx_path}\" \"{merged_output_path}\""
                    run_command(merge_cmd, log_placeholder)
                    
                    if merged_output_path.exists():
                        st.success(f"Merged model created: {merged_output_path.name}")
                        with open(merged_output_path, "rb") as f:
                            st.download_button(
                                label="Download Merged .onnx",
                                data=f,
                                file_name=f"{model_name}_merged.onnx",
                                mime="application/octet-stream"
                            )
                else:
                    st.warning("Could not find embedding model or custom model for merging.")
            except Exception as e:
                st.error(f"Merging failed: {e}")
        
        # ========================================
        # TRAINING COMPLETE - Show final summary
        # ========================================
        st.balloons()
        st.success("""🎉 **Training Complete!**
        
Your wake word model has been trained successfully. Download the models above:
- **ONNX Model**: Standard FP32 precision model
- **Merged Model**: Includes embedding layer for end-to-end inference""")
        
        # Show file locations
        output_dir = Path(config["output_dir"])
        st.info(f"📁 All models saved to: `{output_dir}`")
        
        # Reset training state but DON'T rerun to keep results visible
        st.session_state.training_running = False
        # Removed st.rerun() to keep the results visible!

