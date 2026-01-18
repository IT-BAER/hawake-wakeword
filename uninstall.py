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
    
    # Identify what will be removed - group by category
    items_to_remove = []  # List of (name, path, is_directory)
    
    venv_dir = script_dir / ".venv"
    if venv_dir.exists() and not args.keep_venv:
        items_to_remove.append(("Virtual environment (.venv)", venv_dir, True))
    
    # Cache directories (always clean these)
    cache_patterns = [
        ("preview_temp", "Preview cache"),
        (".streamlit", "Streamlit cache"),
    ]
    
    for pattern, name in cache_patterns:
        path = script_dir / pattern
        if path.exists():
            items_to_remove.append((name, path, True))
    
    # Count __pycache__ directories (don't list individually)
    pycache_dirs = list(script_dir.rglob("__pycache__"))
    if pycache_dirs:
        # Calculate total size
        total_size = sum(
            sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            for d in pycache_dirs if d.exists()
        )
        items_to_remove.append((f"Python cache ({len(pycache_dirs)} __pycache__ dirs)", pycache_dirs, "pycache"))
    
    # Generated config files (always remove)
    generated_files = [
        "my_model.yaml",
        "test_output.wav",
    ]
    for filename in generated_files:
        path = script_dir / filename
        if path.exists():
            items_to_remove.append((f"Generated: {filename}", path, False))
    
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
                items_to_remove.append((name, path, True))
        
        # Downloaded data files
        downloaded_data = [
            ("mit_rirs", "RIR impulse responses"),
            ("audioset_16k", "AudioSet background audio"),
            ("fma", "FMA music samples"),
            ("downloaded_models", "Downloaded models cache"),
        ]
        for folder, name in downloaded_data:
            path = script_dir / folder
            if path.exists():
                items_to_remove.append((name, path, True))
        
        # Large validation/feature files
        large_files = [
            ("validation_set_features.npy", "Validation features"),
            ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", "ACAV100M feature data"),
        ]
        for filename, name in large_files:
            path = script_dir / filename
            if path.exists():
                items_to_remove.append((name, path, False))
        
        # Pip cache directory (can be very large)
        pip_cache_paths = [
            Path.home() / "AppData" / "Local" / "pip" / "cache",  # Windows
            Path.home() / ".cache" / "pip",  # Linux/Mac
        ]
        for pip_cache in pip_cache_paths:
            if pip_cache.exists():
                items_to_remove.append(("Pip download cache", pip_cache, True))
                break  # Only add once
        
        # Trained model output folders (unless --keep-trained)
        if not args.keep_trained:
            # Look for directories that look like trained model outputs
            # These typically contain checkpoints/, onnx files, etc.
            exclude_dirs = {".venv", ".git", ".streamlit", "openwakeword", 
                          "piper-sample-generator", "mit_rirs", "__pycache__",
                          "preview_temp", "docs", ".github"}
            
            trained_models = []
            for path in script_dir.iterdir():
                if path.is_dir() and path.name not in exclude_dirs:
                    # Check if it looks like a model output folder
                    has_onnx = any(path.glob("*.onnx"))
                    has_checkpoints = (path / "checkpoints").exists()
                    has_positive = (path / "positive").exists()
                    has_negative = (path / "negative").exists()
                    
                    if has_onnx or has_checkpoints or has_positive or has_negative:
                        trained_models.append(path)
            
            if trained_models:
                for model_path in trained_models:
                    items_to_remove.append((f"Trained model: {model_path.name}", model_path, True))
        
        # Generated ONNX files in root directory (not base models)
        base_models = {"melspectrogram.onnx", "embedding_model.onnx"}
        generated_onnx = [f for f in script_dir.glob("*.onnx") if f.name not in base_models]
        if generated_onnx:
            items_to_remove.append((f"Generated ONNX models ({len(generated_onnx)} files)", generated_onnx, "onnx"))
        
        # Downloaded Piper TTS models
        piper_models_dir = script_dir / "piper-sample-generator" / "models"
        if piper_models_dir.exists():
            piper_files = list(piper_models_dir.glob("*.onnx")) + list(piper_models_dir.glob("*.pt"))
            if piper_files:
                items_to_remove.append((f"Piper TTS models ({len(piper_files)} files)", piper_files, "piper"))
    else:
        # Standard mode: also remove my_custom_model output directories
        output_patterns = [
            ("my_custom_model", "Training outputs"),
            ("openwakeword/openwakeword/my_custom_model", "OpenWakeWord outputs"),
        ]
        
        for pattern, name in output_patterns:
            path = script_dir / pattern
            if path.exists():
                items_to_remove.append((name, path, True))
    
    if not items_to_remove:
        print_success("Nothing to uninstall - environment is clean!")
        if not args.full:
            print(f"\n{YELLOW}Tip:{RESET} Use {CYAN}--full{RESET} to also remove cloned repos, trained models, and downloads.")
        return
    
    # Calculate total sizes for grouped items
    def get_total_size(paths):
        """Get total size for a list of paths"""
        total = 0
        if isinstance(paths, list):
            for p in paths:
                if p.is_file():
                    total += p.stat().st_size
                elif p.is_dir():
                    for f in p.rglob('*'):
                        if f.is_file():
                            total += f.stat().st_size
        else:
            total = get_size_str(paths)
            return total
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"
    
    # Show what will be removed
    print(f"{BOLD}The following will be removed:{RESET}\n")
    for name, path_or_list, item_type in items_to_remove:
        if isinstance(path_or_list, list):
            size = get_total_size(path_or_list)
            print(f"  • {name} ({size})")
        else:
            size = get_size_str(path_or_list)
            try:
                rel_path = path_or_list.relative_to(script_dir)
            except (ValueError, AttributeError):
                rel_path = path_or_list
            print(f"  • {name} ({size})")
    
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
    for name, path_or_list, item_type in items_to_remove:
        if isinstance(path_or_list, list):
            # Handle grouped items (pycache dirs, onnx files, piper models)
            print_step(f"Removing {name}...")
            removed = 0
            for path in path_or_list:
                try:
                    if path.is_file():
                        path.unlink()
                        removed += 1
                    elif path.is_dir():
                        shutil.rmtree(path)
                        removed += 1
                except Exception:
                    pass
            if removed > 0:
                print_success(f"Removed {name}")
                success_count += 1
        elif item_type == False:  # Single file
            print_step(f"Removing {name}...")
            try:
                path_or_list.unlink()
                print_success(f"Removed {name}")
                success_count += 1
            except Exception as e:
                print_error(f"Failed to remove {name}: {e}")
        elif item_type == True:  # Directory
            if remove_directory(path_or_list, name):
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
