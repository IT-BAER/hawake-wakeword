#!/usr/bin/env python3
"""
HAwake WakeWord Training - Update Script

Usage:
    python update.py              # Pull latest and update dependencies
    python update.py --force      # Force reinstall all dependencies

This script:
1. Pulls latest changes from git
2. Updates pip dependencies
3. Downloads any missing TTS models
4. Verifies the installation
"""

import os
import sys
import subprocess
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_step(msg):
    print(f"{CYAN}[*]{RESET} {msg}")

def print_success(msg):
    print(f"{GREEN}[✓]{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}[!]{RESET} {msg}")

def print_error(msg):
    print(f"{RED}[✗]{RESET} {msg}")

def run_cmd(cmd, check=True, capture=False):
    """Run a command and return output"""
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return subprocess.run(cmd, shell=True, check=check).returncode == 0

def remove_onnxruntime_gpu(pip_exe):
    """Older installs have onnxruntime-gpu next to onnxruntime; both write the same module folder.
    Its CUDA provider does not load with current PyTorch builds, so keep only the CPU package."""
    if not run_cmd(f'"{pip_exe}" show onnxruntime-gpu', capture=True, check=False):
        return
    info = run_cmd(f'"{pip_exe}" show onnxruntime', capture=True, check=False) or ""
    version = next((l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("Version:")), "")
    print_step("Removing onnxruntime-gpu (no longer used)...")
    run_cmd(f'"{pip_exe}" uninstall -y onnxruntime-gpu -q', check=False)
    # the uninstall deleted files the CPU package shares, so reinstall it
    spec = f"onnxruntime=={version}" if version else "onnxruntime"
    run_cmd(f'"{pip_exe}" install --force-reinstall --no-deps "{spec}" -q', check=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Update HAwake WakeWord Training")
    parser.add_argument('--force', action='store_true', help="Force reinstall all dependencies")
    args = parser.parse_args()
    
    print(f"\n{BOLD}🔄 HAwake WakeWord Training - Update{RESET}\n")
    
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    is_windows = sys.platform == "win32"
    venv_dir = script_dir / ".venv"
    
    # Determine Python paths
    if is_windows:
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    
    # Check if venv exists
    if not venv_dir.exists():
        print_error("Virtual environment not found. Please run quickstart.py first.")
        print(f"  {CYAN}python quickstart.py{RESET}")
        return 1
    
    # Step 1: Pull latest changes
    print_step("Pulling latest changes from git...")
    result = run_cmd("git pull", capture=True, check=False)
    if result:
        if "Already up to date" in result:
            print_success("Already up to date")
        else:
            print_success(f"Updated: {result[:100]}...")
    else:
        print_warning("Git pull failed - continuing with local files")
    
    # Step 2: Update pip
    print_step("Updating pip...")
    run_cmd(f'"{pip_exe}" install --upgrade pip -q', check=False)
    
    # Step 3: Update dependencies
    print_step("Updating dependencies...")
    
    if args.force:
        run_cmd(f'"{pip_exe}" install -r requirements.txt --upgrade', check=False)
    else:
        run_cmd(f'"{pip_exe}" install -r requirements.txt -q', check=False)
    
    # Update piper requirements
    piper_req = script_dir / "piper-sample-generator" / "requirements.txt"
    if piper_req.exists():
        if args.force:
            run_cmd(f'"{pip_exe}" install -r "{piper_req}" --upgrade', check=False)
        else:
            run_cmd(f'"{pip_exe}" install -r "{piper_req}" -q', check=False)
    
    remove_onnxruntime_gpu(pip_exe)
    print_success("Dependencies updated")

    # Step 4: Download Piper TTS model if needed
    piper_models_dir = script_dir / "piper-sample-generator" / "models"
    piper_model = piper_models_dir / "en_US-libritts_r-medium.onnx"
    piper_config = piper_models_dir / "en_US-libritts_r-medium.onnx.json"
    
    if not piper_model.exists() or not piper_config.exists():
        print_step("Downloading Piper TTS model...")
        piper_models_dir.mkdir(parents=True, exist_ok=True)
        
        model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx"
        config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json"
        
        try:
            import urllib.request
            
            if not piper_model.exists():
                print_step("  Downloading ONNX model...")
                urllib.request.urlretrieve(model_url, piper_model)
            
            if not piper_config.exists():
                print_step("  Downloading config file...")
                urllib.request.urlretrieve(config_url, piper_config)
            
            print_success("Piper TTS model downloaded")
        except Exception as e:
            print_warning(f"Could not download model: {e}")
    else:
        print_success("Piper TTS model exists")
    
    # Step 5: Verify installation
    print_step("Verifying installation...")
    
    # Check if streamlit is installed
    result = run_cmd(f'"{pip_exe}" show streamlit', capture=True, check=False)
    if result:
        print_success("Streamlit: OK")
    else:
        print_warning("Streamlit not found - installing...")
        run_cmd(f'"{pip_exe}" install streamlit -q', check=False)
    
    # Check if piper-tts is installed  
    result = run_cmd(f'"{pip_exe}" show piper-tts', capture=True, check=False)
    if result:
        print_success("Piper TTS: OK")
    else:
        print_warning("Piper TTS not found - installing...")
        run_cmd(f'"{pip_exe}" install piper-tts>=1.2.0 -q', check=False)
    
    print(f"\n{GREEN}{BOLD}✅ Update complete!{RESET}\n")
    print(f"Run {CYAN}streamlit run app.py{RESET} to start the WebUI.\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
