# HAwake Wake Word Training - Quick Start Guide

## 🚀 Fast Installation (15-30 minutes vs 1+ hour)

### Pre-requisites
- Python 3.11+ installed
- NVIDIA GPU recommended (CPU works but 5-10x slower)
- 2-3 GB disk space for dependencies
- ~1–2 GB for training data (plus optional 16GB features)

### Option 1: Full Installation (Recommended for first time)
```powershell
# Windows PowerShell
.\install.ps1
```

### Option 2: Minimal Installation (Fastest, ~10 minutes)
If you already have PyTorch installed or want to skip data downloads:

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch (skip if already installed globally)
# RTX 20xx / 30xx / 40xx (CUDA 12.1):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# RTX 50xx / Blackwell (requires nightly, CUDA 12.8):
# pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install required dependencies
pip install -r requirements.txt
pip install streamlit

# Install OpenWakeWord (editable)
pip install -e openwakeword

# Apply compatibility patches
python patch_dp.py
python patch_torch_audiomentations.py
```

### Option 3: Using Pre-bundled Data (Fastest for returning users)
If you have the data files from a previous installation, just copy them:
- `audioset_16k/` folder (background audio)
- `mit_rirs/` folder (room impulse responses)
- `validation_set_features.npy` (validation data)

## 🏃 Quick Training (30-60 minutes vs 2+ hours)

### Fast Training Settings (Good for testing)
In the WebUI sidebar, use these settings for faster training:

| Setting | Fast Value | Quality Value |
|---------|------------|---------------|
| Number of Examples | 1000 | 5000 |
| Training Steps | 1000 | 5000 |
| False Activation Penalty | 500 | 1500 |

**Estimated times with GPU:**
- Fast settings: ~20-30 minutes
- Quality settings: ~1-2 hours

### GPU vs CPU Training
| GPU | Approximate Training Time |
|-----|---------------------------|
| RTX 5060 Ti / 50xx (nightly cu128) | 15-30 minutes |
| RTX 4090 | 15-30 minutes |
| RTX 3060 | 30-60 minutes |
| GTX 1060 | 1-2 hours |
| CPU only | 4-8 hours |

## 🔄 Resume Training (NEW!)

Training now supports checkpoints! If training fails or you need to stop:

1. Enable **"Resume Training"** checkbox (default: ON)
2. Click **"Start Training"** again
3. Completed steps (clips, features) will be skipped automatically

### What Gets Saved
- Generated audio clips → `{model_name}/positive_train/`
- Computed features → `{model_name}/features/*.npy`
- Trained model → `{model_name}/{model_name}.onnx`

## ⚡ Performance Tips

### 1. Skip Data Download (Use Your Own)
If you have background audio files (.wav, 16kHz):
```powershell
# Copy your files to audioset_16k folder
mkdir audioset_16k
# Copy 500+ wav files here
```

### 2. Reduce Training Examples
For testing, 1000-2000 examples is often sufficient. Production models benefit from 5000+.

### 3. Use GPU
Ensure CUDA is working:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU
```

### 4. Skip Large Feature File
The 16GB `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` is optional. Training works without it using the AudioSet background audio.

## 🐛 Common Issues

### "No CUDA devices available" or "no kernel image for this device"
```powershell
# Reinstall PyTorch with CUDA
pip uninstall torch torchaudio torchvision -y

# RTX 20xx / 30xx / 40xx:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# RTX 50xx / Blackwell (sm_120) — requires PyTorch nightly:
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### "weights_only" Error
```powershell
python patch_dp.py
```

### "torchaudio.info" Error
```powershell
python patch_torch_audiomentations.py
```

### Training Crashes at Augmentation
Usually a sample rate conversion issue. Fixed in latest version:
```powershell
git pull origin main
python patch_torch_audiomentations.py
```
