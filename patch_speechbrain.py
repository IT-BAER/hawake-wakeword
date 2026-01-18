"""
Patch speechbrain for torchaudio 2.1+ compatibility.

In newer torchaudio versions (2.1+), torchaudio.list_audio_backends() was removed.
This script patches speechbrain to handle this gracefully.
"""

import os
import sys
from pathlib import Path


def find_speechbrain_file():
    """Find speechbrain/utils/torch_audio_backend.py in the current environment."""
    try:
        import speechbrain
        sb_dir = os.path.dirname(speechbrain.__file__)
        target_file = os.path.join(sb_dir, "utils", "torch_audio_backend.py")
        if os.path.exists(target_file):
            return Path(target_file)
    except ImportError:
        pass
    
    # Fallback: try common venv locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = Path(script_dir) / ".venv" / "Lib" / "site-packages" / "speechbrain" / "utils" / "torch_audio_backend.py"
    if venv_path.exists():
        return venv_path
    
    return None


def patch_speechbrain():
    target_file = find_speechbrain_file()
    
    if target_file is None:
        print("[!] speechbrain not found - skipping patch")
        return False
    
    print(f"[*] Found speechbrain at: {target_file}")
    
    content = target_file.read_text(encoding='utf-8')
    
    # Check if already patched
    if "# Patched for torchaudio 2.1+" in content or "torchaudio.set_audio_backend" not in content:
        print("[✓] speechbrain doesn't need patching or already patched")
        return True
    
    # Apply patch - remove or wrap the problematic calls
    new_content = content.replace(
        'torchaudio.set_audio_backend(backend)',
        'pass  # Patched for torchaudio 2.1+: torchaudio.set_audio_backend(backend)'
    )
    
    if new_content != content:
        target_file.write_text(new_content, encoding='utf-8')
        print("[✓] Patched speechbrain for torchaudio 2.1+ compatibility")
        return True
    
    print("[!] Could not find expected code pattern in speechbrain")
    return False


if __name__ == "__main__":
    success = patch_speechbrain()
    sys.exit(0 if success else 1)

