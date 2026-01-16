# HAwake WakeWord Training

Custom OpenWakeWord training pipeline for generating wake word models compatible with [HAwake Android](https://github.com/IT-BAER/hawake-android).

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
- Detects GPU and installs appropriate PyTorch version
- Installs all dependencies
- Launches the WebUI at http://localhost:8501

## Features

- **Streamlit WebUI** - Easy-to-use interface for training custom wake words
- **ONNX Opset 11** - All models optimized for Android 8+ compatibility
- **Piper TTS** - High-quality synthetic voice generation
- **Auto-Patching** - Automatic IR version and attribute fixes for ONNX Runtime 1.14.0

## Requirements

- Python 3.10+
- CUDA (optional, for GPU acceleration)
- ~10GB disk space for training data

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

All models are exported with:
- **Opset 11** - Best Android 8+ compatibility
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

| Issue | Solution |
|-------|----------|
| Model crashes on Android | Run `check_opset.py` - must be Opset 11 |
| Training fails on GPU | Add `--force_cpu` flag |
| TTS sounds wrong | Adjust spelling phonetically |
| Low detection accuracy | Increase training examples to 10000+ |

## Credits

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Original project
- [Piper](https://github.com/rhasspy/piper) - TTS engine
- [ONNX Runtime](https://onnxruntime.ai/) - Model inference

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
