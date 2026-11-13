# HAwake Wake Word Training - Quick Start

The short path is in the [README](README.md#quick-start-for-hawake-users): install, train in the WebUI, load the `.onnx` file in HAwake. This page adds manual setup and tips.

## Requirements

- Python 3.11 exactly (newer versions lack wheels for piper-phonemize and other pinned packages)
- Git
- About 10 GB of free disk space
- Optional: NVIDIA GPU. RTX 50 series (Blackwell) needs NVIDIA driver 580 or newer for the CUDA 13.0 PyTorch build the installers pick.

## Manual installation

Use this only when you cannot run `install.ps1`, `install.bat` or `install.sh`; the installers pick the right PyTorch build for your GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch: pick ONE line
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130   # RTX 50 series, driver 580+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126   # older NVIDIA GPUs
pip install torch torchvision torchaudio                                                       # no NVIDIA GPU

pip install -r requirements.txt
pip install streamlit
pip install -r piper-sample-generator\requirements.txt

# Compatibility patches (each stops with an error if it cannot patch)
python patch_torch_audiomentations.py
python patch_dp.py

# Training data: AudioSet background audio, room impulse responses, validation features
python download_data.py
```

Then start the WebUI with `.\run_webui.ps1` or `streamlit run app.py`.

## Reusing data from an earlier installation

Copy these from the old folder into the new one to skip the downloads:
- `audioset_16k/` (background audio, files named `audioset_*.wav`)
- `mit_rirs/` (room impulse responses)
- `validation_set_features.npy`

## Training settings

The defaults are 5000 examples, 5000 training steps and a False Activation Penalty of 1500. Test runs on 13 November 2026, wake word "hey nova", all with the same 5000 generated clips:

| Run | Training features (17 GB) | Steps | Detected test clips | Test negatives that fire | False activations per hour (validation set) |
|-----|---------------------------|-------|---------------------|--------------------------|---------------------------------------------|
| A | no | 5000 | 69 % | 11 % | 9727 |
| B (default) | yes | 5000 | 35 % | 0.2 % | 8.7 |
| C | yes | 20000 | 55 % | 3.2 % | 96.9 |

Detection was measured with ONNX Runtime 1.22.0 (the version in HAwake) at threshold 0.5 on the augmented test clips, which are harder than clean speech. The HAwake built-in models had 7 to 14 false activations per hour in the same validation.

- **Training features:** without `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` the model reacts to normal speech (run A). `download_data.py` and the installers download it.
- **Training Steps:** more steps detect the wake word more often, but also fire more often on other speech (run C). If HAwake misses your wake word, train again with more steps; with **Resume Training** on, this reuses the clips and took 12 minutes for 20000 steps on an RTX 5060 Ti.
- **False Activation Penalty:** raise it when the model fires too often.
- **Number of Examples:** clip generation runs on the CPU, about 25 minutes per 1000 examples on a 6-core Ryzen 5 7500F.

## Resume

With **Resume Training** enabled (default), a new run skips the steps whose output already exists:
- generated clips in `<wake_word>/positive_train/`, `positive_test/`, `negative_train/`, `negative_test/`
- computed features in `<wake_word>/*_features_*.npy`
- the model is always trained again and saved as `<wake_word>/<wake_word>.onnx`

Disable it, or enable **Force Recompute Features**, after you change the wake word spelling or augmentation settings.

## Common issues

| Symptom | Fix |
|---------|-----|
| Sidebar: "NVIDIA GPU found, but PyTorch (CUDA 13.0) cannot use it" | Update the NVIDIA driver to 580 or newer and restart the WebUI, or train on CPU |
| Sidebar or installer: synthetic background audio / "no real background audio" | `python download_data.py` (the AudioSet download failed earlier) |
| "weights_only" error during training | `python patch_dp.py` |
| `torchaudio` errors in `torch_audiomentations` | `python patch_torch_audiomentations.py` |
| HAwake: "Invalid ONNX file (incompatible version)" | `python check_opset.py <wake_word>/<wake_word>.onnx` and keep the defaults under *Advanced export* |
