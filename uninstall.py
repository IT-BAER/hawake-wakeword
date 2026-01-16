#!/usr/bin/env python3
"""
HAwake WakeWord Training - Uninstall Script

Usage:
    python uninstall.py              # Interactive uninstall
    python uninstall.py --full       # Full cleanup including models
    python uninstall.py --keep-venv  # Keep virtual environment

This script removes:
- Virtual environment (.venv)
- Generated training data
- Output models (optional)
- Cache and temporary files
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
    parser.add_argument('--full', action='store_true', help="Full cleanup including downloaded models")
    parser.add_argument('--keep-venv', action='store_true', help="Keep the virtual environment")
    parser.add_argument('-y', '--yes', action='store_true', help="Skip confirmation prompts")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    
    print(f"\n{BOLD}🗑️  HAwake WakeWord Training - Uninstaller{RESET}\n")
    
    # Identify what will be removed
    items_to_remove = []
    
    venv_dir = script_dir / ".venv"
    if venv_dir.exists() and not args.keep_venv:
        items_to_remove.append(("Virtual environment", venv_dir))
    
    # Training output directories
    output_patterns = [
        ("my_custom_model", "Training outputs"),
        ("preview_temp", "Preview cache"),
        ("openwakeword/openwakeword/my_custom_model", "OpenWakeWord outputs"),
        ("__pycache__", "Python cache"),
        ("openwakeword/__pycache__", "OpenWakeWord cache"),
        ("openwakeword/openwakeword/__pycache__", "OpenWakeWord cache"),
        (".streamlit", "Streamlit cache"),
    ]
    
    for pattern, name in output_patterns:
        path = script_dir / pattern
        if path.exists():
            items_to_remove.append((name, path))
    
    # Find all __pycache__ directories
    for cache_dir in script_dir.rglob("__pycache__"):
        if cache_dir.exists() and cache_dir not in [p for _, p in items_to_remove]:
            items_to_remove.append(("Python cache", cache_dir))
    
    # Full cleanup includes downloaded models
    if args.full:
        piper_models = script_dir / "piper-sample-generator" / "models"
        if piper_models.exists():
            items_to_remove.append(("Piper TTS models (downloaded)", piper_models))
        
        # Any generated .onnx files in root (not the base models)
        for onnx_file in script_dir.glob("*.onnx"):
            if onnx_file.name not in ["melspectrogram.onnx", "embedding_model.onnx"]:
                items_to_remove.append((f"Generated model: {onnx_file.name}", onnx_file))
    
    if not items_to_remove:
        print_success("Nothing to uninstall - environment is clean!")
        return
    
    # Show what will be removed
    print(f"{BOLD}The following will be removed:{RESET}\n")
    total_size = 0
    for name, path in items_to_remove:
        size = get_size_str(path)
        print(f"  • {name}: {CYAN}{path.relative_to(script_dir)}{RESET} ({size})")
    
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
