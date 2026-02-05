# HAwake WakeWord Training

Custom OpenWakeWord training pipeline for generating wake word models compatible with [HAwake Android](https://play.google.com/store/apps/details?id=com.baer.hawake).

## Quick Start (One-Liner)

### Windows (PowerShell)
```powershell
git clone https://github.com/IT-BAER/hawake-wakeword.git; cd hawake-wakeword; .\install.ps1
```

### Windows (CMD)
```cmd
git clone https://github.com/IT-BAER/hawake-wakeword.git && cd hawake-wakeword && install.bat
```

### Linux/macOS
```bash
git clone https://github.com/IT-BAER/hawake-wakeword.git && cd hawake-wakeword && chmod +x install.sh && ./install.sh
```

The setup script automatically:
- Creates a virtual environment
- Detects GPU and verifies CUDA kernel compatibility
- Installs appropriate PyTorch version (GPU or CPU)
- Installs all dependencies
- Downloads Room Impulse Responses for audio augmentation
- Downloads background audio data (~500MB–1GB) for training
- Launches the WebUI at http://localhost:8501

**Note:** First-time setup downloads ~1-2 GB of required training data and may take 10-20 minutes.

> **Optional:** For best training quality, download the full feature file (~16GB) separately:
> ```bash
> curl -L -o openwakeword_features_ACAV100M_2000_hrs_16bit.npy https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
> ```
> Or run the install/download with the flag enabled:
> ```bash
> HAWAKE_DOWNLOAD_LARGE_FEATURES=1 ./install.sh
> ```

## Uninstall

### Windows (PowerShell)
```powershell
.\uninstall.ps1
```

### Windows (CMD)
```cmd
uninstall.bat
```

### Linux/macOS
```bash
./uninstall.sh
```

### Options
| Flag | Description |
|------|-------------|
| `--full` | Also remove downloaded TTS models (~200MB) |
| `--keep-venv` | Keep the virtual environment |
| `-y` | Skip confirmation prompts |

Example full cleanup:
```bash
./uninstall.sh --full -y
```

## Features

- **Streamlit WebUI** - Easy-to-use interface for training custom wake words
- **ONNX Opset 11** - Default opset for Android 8+ compatibility (selectable in WebUI)
- **Piper TTS** - High-quality synthetic voice generation
- **Auto-Patching** - Automatic IR version and attribute fixes for ONNX Runtime 1.14.0

## Requirements

- Python 3.11+ (Linux/WSL: use 3.11; Python 3.12 lacks piper-phonemize manylinux wheels and requires a source build)
- CUDA (optional, for GPU acceleration)
- ~10GB disk space (training data + models)

## Manual Setup

If you prefer manual installation:

### 1. Clone and Setup

```bash
git clone https://github.com/IT-BAER/hawake-wakeword.git
cd hawake-wakeword
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Download Training Data

```bash
python download_data.py     # Background audio (AudioSet, FMA)
python download_rirs.py     # Room impulse responses
```

### 3. Start WebUI

```bash
# Windows
.\run_webui.ps1

# Or directly
streamlit run app.py
```

### 4. Train a Wake Word

1. Enter your wake word (e.g., "hey jarvis")
2. Click **Generate Preview** to hear pronunciation
3. Adjust spelling if needed for correct pronunciation
4. Click **Start Training**
5. Download the resulting `.onnx` model

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Target Opset | 11 | ONNX opset version (11 required for HAwake) |
| Number of Examples | 5000 | Synthetic training samples |
| Training Steps | 5000 | Model training iterations |
| False Activation Penalty | 1500 | Reduces false positives |

## CLI Training

For advanced users:

```bash
python train_local.py
```

Edit the file to configure:
- `target_word` - Your wake word
- `number_of_examples` - Training sample count
- `number_of_training_steps` - Training iterations

## Integration with HAwake Android

After training, copy your model to HAwake:

```bash
# Copy to HAwake Android assets
cp my_model/my_wake_word.onnx /path/to/hawake-android/app/src/main/assets/
```

Or upload via the HAwake app's "Manage Wake Words" dialog.

## ONNX Compatibility

Models are exported with:
- **Selected Opset** (default 11) - Best Android 8+ compatibility
- **IR Version 7** - Required for ONNX Runtime 1.14.0
- **No `allowzero`** - Attribute removed from Reshape nodes

Verify with:
```bash
python check_opset.py my_model.onnx
```

## Project Structure

```
├── app.py                 # Streamlit WebUI
├── train_local.py         # CLI training script
├── check_opset.py         # ONNX version checker
├── convert_models.py      # Opset conversion utility
├── merge_models.py        # Merge embedding + classifier
├── openwakeword/          # OpenWakeWord library (modified)
│   └── openwakeword/
│       ├── train.py       # Training logic
│       └── resources/
│           └── models/    # Base feature extraction models
└── piper-sample-generator/ # TTS for synthetic audio
```

## Troubleshooting

### GPU Issues

| Issue | Solution |
|-------|----------|
| GPU detected but "no kernel image" | Very new GPU (e.g., RTX 50 series). Use "Force CPU Mode" checkbox in WebUI |
| Training fails on GPU | Enable "Force CPU Mode" in WebUI sidebar |
| CUDA not available | Install NVIDIA drivers or use CPU mode |
| Wrong GPU detected | Set `CUDA_VISIBLE_DEVICES=0` environment variable |

The WebUI has a **"Force CPU Mode"** checkbox under Hardware settings that lets you disable GPU even if one is detected. This is useful for:
- Very new GPUs that don't have PyTorch kernels yet
- GPUs with insufficient memory
- Troubleshooting CUDA issues

### Other Issues

| Issue | Solution |
|-------|----------|
| Model crashes on Android | Run `check_opset.py` - must be Opset 11 |
| TTS sounds wrong | Adjust spelling phonetically |
| Low detection accuracy | Increase training examples to 10000+ |

## Credits

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Original project
- [Piper](https://github.com/rhasspy/piper) - TTS engine
- [ONNX Runtime](https://onnxruntime.ai/) - Model inference

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
