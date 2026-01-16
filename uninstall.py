#!/usr/bin/env python3
"""
HAwake WakeWord Training - Uninstall Script

Usage:
    python uninstall.py              # Interactive uninstall (removes cache + temp files)
    python uninstall.py --full       # Full cleanup including cloned repos, models, trained outputs
    python uninstall.py --keep-venv  # Keep virtual environment
    python uninstall.py --keep-trained  # Keep trained model output folders

This script removes:
- Virtual environment (.venv)
- Generated training data and configs
- Preview/test output files
- Cache and temporary files
- [--full] Cloned repositories (openwakeword, piper-sample-generator)
- [--full] Downloaded RIR files (mit_rirs)
- [--full] Generated ONNX models
- [--full] Trained model output folders (unless --keep-trained)
"""

import os
import sys
import shutil
import argparse
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

def get_size_str(path):
    """Get human-readable size of a directory"""
    total = 0
    if path.is_file():
        total = path.stat().st_size
    elif path.is_dir():
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"

def remove_directory(path, name):
    """Safely remove a directory"""
    if path.exists():
        size = get_size_str(path)
        print_step(f"Removing {name} ({size})...")
        try:
            shutil.rmtree(path)
            print_success(f"Removed {name}")
            return True
        except Exception as e:
            print_error(f"Failed to remove {name}: {e}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Uninstall HAwake WakeWord Training")
    parser.add_argument('--full', action='store_true', help="Full cleanup including cloned repos and all generated files")
    parser.add_argument('--keep-venv', action='store_true', help="Keep the virtual environment")
    parser.add_argument('--keep-trained', action='store_true', help="Keep trained model output folders")
    parser.add_argument('-y', '--yes', action='store_true', help="Skip confirmation prompts")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    
    print(f"\n{BOLD}🗑️  HAwake WakeWord Training - Uninstaller{RESET}\n")
    
    # Identify what will be removed
    items_to_remove = []
    
    venv_dir = script_dir / ".venv"
    if venv_dir.exists() and not args.keep_venv:
        items_to_remove.append(("Virtual environment", venv_dir))
    
    # Cache directories (always clean these)
    cache_patterns = [
        ("preview_temp", "Preview cache"),
        (".streamlit", "Streamlit cache"),
    ]
    
    for pattern, name in cache_patterns:
        path = script_dir / pattern
        if path.exists():
            items_to_remove.append((name, path))
    
    # Find all __pycache__ directories
    for cache_dir in script_dir.rglob("__pycache__"):
        if cache_dir.exists() and cache_dir not in [p for _, p in items_to_remove]:
            items_to_remove.append(("Python cache", cache_dir))
    
    # Find .pyc files
    for pyc_file in script_dir.rglob("*.pyc"):
        if pyc_file.exists():
            items_to_remove.append(("Compiled Python", pyc_file))
    
    # Generated config files (always remove)
    generated_files = [
        "my_model.yaml",
        "test_output.wav",
    ]
    for filename in generated_files:
        path = script_dir / filename
        if path.exists():
            items_to_remove.append((f"Generated: {filename}", path))
    
    # Full cleanup mode
    if args.full:
        # Cloned repositories
        cloned_repos = [
            ("openwakeword", "OpenWakeWord repository"),
            ("piper-sample-generator", "Piper sample generator"),
        ]
        for folder, name in cloned_repos:
            path = script_dir / folder
            if path.exists():
                items_to_remove.append((name, path))
        
        # Downloaded RIR files
        rir_dir = script_dir / "mit_rirs"
        if rir_dir.exists():
            items_to_remove.append(("RIR impulse responses", rir_dir))
        
        # Trained model output folders (unless --keep-trained)
        if not args.keep_trained:
            # Look for directories that look like trained model outputs
            # These typically contain checkpoints/, onnx files, etc.
            exclude_dirs = {".venv", ".git", ".streamlit", "openwakeword", 
                          "piper-sample-generator", "mit_rirs", "__pycache__",
                          "preview_temp", "docs", ".github"}
            
            for path in script_dir.iterdir():
                if path.is_dir() and path.name not in exclude_dirs:
                    # Check if it looks like a model output folder
                    has_onnx = any(path.glob("*.onnx"))
                    has_checkpoints = (path / "checkpoints").exists()
                    has_positive = (path / "positive").exists()
                    has_negative = (path / "negative").exists()
                    
                    if has_onnx or has_checkpoints or has_positive or has_negative:
                        items_to_remove.append((f"Trained model: {path.name}", path))
        
        # Generated ONNX files in root directory (not base models)
        base_models = {"melspectrogram.onnx", "embedding_model.onnx"}
        for onnx_file in script_dir.glob("*.onnx"):
            if onnx_file.name not in base_models:
                items_to_remove.append((f"Generated model: {onnx_file.name}", onnx_file))
        
        # Downloaded Piper TTS models
        piper_models_dir = script_dir / "piper-sample-generator" / "models"
        if piper_models_dir.exists():
            for onnx_file in piper_models_dir.glob("*.onnx"):
                items_to_remove.append((f"Piper model: {onnx_file.name}", onnx_file))
            for pt_file in piper_models_dir.glob("*.pt"):
                items_to_remove.append((f"Piper model: {pt_file.name}", pt_file))
    else:
        # Standard mode: also remove my_custom_model output directories
        output_patterns = [
            ("my_custom_model", "Training outputs"),
            ("openwakeword/openwakeword/my_custom_model", "OpenWakeWord outputs"),
        ]
        
        for pattern, name in output_patterns:
            path = script_dir / pattern
            if path.exists():
                items_to_remove.append((name, path))
    
    if not items_to_remove:
        print_success("Nothing to uninstall - environment is clean!")
        if not args.full:
            print(f"\n{YELLOW}Tip:{RESET} Use {CYAN}--full{RESET} to also remove cloned repos, trained models, and downloads.")
        return
    
    # Show what will be removed
    print(f"{BOLD}The following will be removed:{RESET}\n")
    total_size = 0
    for name, path in items_to_remove:
        size = get_size_str(path)
        try:
            rel_path = path.relative_to(script_dir)
        except ValueError:
            rel_path = path
        print(f"  • {name}: {CYAN}{rel_path}{RESET} ({size})")
    
    if not args.full:
        print(f"\n{YELLOW}Tip:{RESET} Use {CYAN}--full{RESET} to also remove cloned repos, trained models, and downloads.")
    
    if args.full and not args.keep_trained:
        print(f"\n{YELLOW}Note:{RESET} Use {CYAN}--keep-trained{RESET} to preserve your trained model outputs.")
    
    print()
    
    # Confirmation
    if not args.yes:
        print(f"{YELLOW}This action cannot be undone.{RESET}")
        print(f"\n{BOLD}Continue with uninstall? [y/N]{RESET} ", end="")
        try:
            response = input().strip().lower()
            if response not in ('y', 'yes'):
                print_warning("Uninstall cancelled.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n")
            print_warning("Uninstall cancelled.")
            return
    
    print()
    
    # Remove items
    success_count = 0
    for name, path in items_to_remove:
        if path.is_file():
            print_step(f"Removing {name}...")
            try:
                path.unlink()
                print_success(f"Removed {name}")
                success_count += 1
            except Exception as e:
                print_error(f"Failed to remove {name}: {e}")
        elif path.is_dir():
            if remove_directory(path, name):
                success_count += 1
    
    print(f"\n{GREEN}{BOLD}✅ Uninstall complete!{RESET}")
    print(f"   Removed {success_count}/{len(items_to_remove)} items\n")
    
    if args.keep_venv:
        print_warning("Virtual environment was kept (--keep-venv flag)")
    
    print(f"\n{BOLD}To reinstall:{RESET}")
    print(f"  {CYAN}python quickstart.py{RESET}")
    print()

if __name__ == "__main__":
    main()
