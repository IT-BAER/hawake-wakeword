"""
Patch DeepPhonemizer (dp) package for PyTorch 2.6+ compatibility.

PyTorch 2.6 changed the default value of `weights_only` in `torch.load` from 
`False` to `True`. This breaks loading checkpoints that contain custom classes
like `dp.preprocessing.text.Preprocessor`.

This patch adds `weights_only=False` to the torch.load call in dp/model/model.py.
"""

import os
import sys
import re

def find_dp_model_file():
    """Find dp/model/model.py in the current environment."""
    try:
        import dp
        dp_dir = os.path.dirname(dp.__file__)
        target_file = os.path.join(dp_dir, "model", "model.py")
        if os.path.exists(target_file):
            return target_file
    except ImportError:
        pass
    
    # Fallback: try common venv locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = os.path.join(script_dir, ".venv", "Lib", "site-packages", "dp", "model", "model.py")
    if os.path.exists(venv_path):
        return venv_path
    
    return None

def patch_dp():
    target_file = find_dp_model_file()
    
    if not target_file:
        print("[!] dp package not found - skipping patch")
        return False
    
    print(f"[*] Found dp at: {target_file}")
    
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if already patched
    if "weights_only=False" in content:
        print("[✓] dp/model/model.py already patched")
        return True
    
    # Pattern 1: Simple replacement
    old_code = "checkpoint = torch.load(checkpoint_path, map_location=device)"
    new_code = "checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)"
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("[✓] Patched dp/model/model.py for PyTorch 2.6+ compatibility")
        return True
    
    # Pattern 2: Regex fallback for different formatting
    pattern = r"torch\.load\(checkpoint_path,\s*map_location\s*=\s*device\)"
    replacement = "torch.load(checkpoint_path, map_location=device, weights_only=False)"
    
    new_content, count = re.subn(pattern, replacement, content)
    if count > 0:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"[✓] Patched {count} occurrence(s) in dp/model/model.py (regex)")
        return True
    
    print("[!] Could not find expected torch.load pattern in dp/model/model.py")
    print("    The file may have a different format. Manual patching may be required.")
    return False

if __name__ == "__main__":
    patch_dp()
