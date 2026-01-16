#!/usr/bin/env python3
"""
HAwake WakeWord Training - Quick Setup Script

Usage (one-liner):
    curl -sSL https://raw.githubusercontent.com/IT-BAER/hawake-wakeword/master/setup.py | python3 -

Or after cloning:
    python setup.py

This script:
1. Creates a virtual environment
2. Installs all dependencies
3. Checks GPU availability
4. Downloads required TTS model
5. Launches the WebUI
"""

import os
import sys
import subprocess
import platform
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

def main():
    print(f"\n{BOLD}🎤 HAwake WakeWord Training Setup{RESET}\n")
    
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    is_windows = platform.system() == "Windows"
    venv_dir = script_dir / ".venv"
    
    # Determine Python paths
    if is_windows:
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
        activate_hint = ".venv\\Scripts\\activate"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
        activate_hint = "source .venv/bin/activate"
    
    # Step 1: Create venv if needed
    if not venv_dir.exists():
        print_step("Creating virtual environment...")
        run_cmd(f"{sys.executable} -m venv .venv")
        print_success("Virtual environment created")
    else:
        print_success("Virtual environment exists")
    
    # Step 2: Install/upgrade pip
    print_step("Upgrading pip...")
    run_cmd(f'"{pip_exe}" install --upgrade pip -q')
    
    # Step 3: Install dependencies
    print_step("Installing dependencies (this may take a few minutes)...")
    
    # Check for CUDA
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    if not cuda_available:
        # Try to detect CUDA from nvidia-smi
        cuda_available = run_cmd("nvidia-smi", check=False, capture=True) is not None
    
    if cuda_available:
        print_success("GPU detected - installing GPU-accelerated packages")
        run_cmd(f'"{pip_exe}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q')
    else:
        print_warning("No GPU detected - using CPU (training will be slower)")
        run_cmd(f'"{pip_exe}" install torch torchvision torchaudio -q')
    
    # Install remaining dependencies
    run_cmd(f'"{pip_exe}" install -r requirements.txt -q')
    run_cmd(f'"{pip_exe}" install streamlit -q')
    
    # Install piper requirements
    piper_req = script_dir / "piper-sample-generator" / "requirements.txt"
    if piper_req.exists():
        run_cmd(f'"{pip_exe}" install -r "{piper_req}" -q')
    
    print_success("Dependencies installed")
    
    # Step 4: Download Piper TTS model if needed
    piper_models_dir = script_dir / "piper-sample-generator" / "models"
    piper_model = piper_models_dir / "en_US-libritts_r-medium.pt"
    
    if not piper_model.exists():
        print_step("Downloading Piper TTS model (~200MB)...")
        piper_models_dir.mkdir(parents=True, exist_ok=True)
        model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.pt"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, piper_model)
            print_success("Piper TTS model downloaded")
        except Exception as e:
            print_warning(f"Could not download model: {e}")
            print_warning("You can download it manually from: " + model_url)
    else:
        print_success("Piper TTS model exists")
    
    # Step 5: Verify opset versions
    print_step("Verifying ONNX model compatibility...")
    try:
        result = run_cmd(f'"{python_exe}" check_opset.py', capture=True)
        if "Opset: 11" in (result or ""):
            print_success("Base models are Opset 11 (Android compatible)")
        else:
            print_warning("Base models may need conversion - check with: python check_opset.py")
    except:
        pass
    
    # Final summary
    print(f"\n{GREEN}{BOLD}✅ Setup complete!{RESET}\n")
    
    if cuda_available:
        print(f"  GPU: {GREEN}Enabled{RESET}")
    else:
        print(f"  GPU: {YELLOW}Not detected (using CPU){RESET}")
    
    print(f"\n{BOLD}To start the WebUI:{RESET}")
    print(f"  {CYAN}cd {script_dir}{RESET}")
    print(f"  {CYAN}{activate_hint}{RESET}")
    print(f"  {CYAN}streamlit run app.py{RESET}")
    
    # Ask if user wants to launch now
    print(f"\n{BOLD}Launch WebUI now? [Y/n]{RESET} ", end="")
    try:
        response = input().strip().lower()
        if response in ('', 'y', 'yes'):
            print_step("Starting WebUI...")
            os.execv(str(python_exe), [str(python_exe), "-m", "streamlit", "run", "app.py"])
    except (KeyboardInterrupt, EOFError):
        print("\n")
    
    print(f"\n{GREEN}Done! Run 'streamlit run app.py' anytime to start training.{RESET}\n")

if __name__ == "__main__":
    main()
