#!/bin/bash
# HAwake WakeWord Training - One-liner Setup
# Usage: curl -sSL https://raw.githubusercontent.com/IT-BAER/hawake-wakeword/main/install.sh | bash
# Or: ./install.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "  HAwake WakeWord Training Setup"
echo "========================================"
echo ""

# Python 3.11, like install.ps1: piper-phonemize and the pinned packages have wheels for 3.11
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3 &> /dev/null && python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
    PYTHON_BIN="python3"
else
    echo -e "${RED}[✗] Python 3.11 not found (python3.11 or python3). Install python3.11 and python3.11-venv, then re-run.${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] Found $("$PYTHON_BIN" --version) ($PYTHON_BIN)${NC}"

# Clone required repositories if not present
echo ""
echo -e "${CYAN}[*] Checking required repositories...${NC}"

# Clone OpenWakeWord if needed
if [ ! -d "openwakeword" ]; then
    echo -e "${CYAN}[*] Cloning openwakeword repository...${NC}"
    git clone https://github.com/dscripka/openwakeword.git openwakeword
    echo -e "${GREEN}[✓] OpenWakeWord cloned${NC}"
else
    echo -e "${GREEN}[✓] OpenWakeWord repository exists${NC}"
fi

# Clone piper-sample-generator if needed
if [ ! -d "piper-sample-generator" ]; then
    echo -e "${CYAN}[*] Cloning piper-sample-generator repository...${NC}"
    git clone https://github.com/rhasspy/piper-sample-generator.git piper-sample-generator
    echo -e "${GREEN}[✓] Piper sample generator cloned${NC}"
else
    echo -e "${GREEN}[✓] Piper sample generator exists${NC}"
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo -e "${CYAN}[*] Creating virtual environment...${NC}"
    "$PYTHON_BIN" -m venv .venv
elif ! grep -qE '^version(_info)? *= *3\.11\.' .venv/pyvenv.cfg 2>/dev/null; then
    # A .venv from another Python brings back the wheel failures above
    echo -e "${RED}[✗] .venv was not created with Python 3.11. Delete the .venv folder and re-run.${NC}"
    exit 1
fi

# Activate
source .venv/bin/activate

PIP_TIMEOUT=${PIP_TIMEOUT:-120}
PIP_RETRIES=${PIP_RETRIES:-5}
FORCE_CPU=${FORCE_CPU:-0}
SKIP_DATA_DOWNLOADS=${SKIP_DATA_DOWNLOADS:-0}
SKIP_PIPER_MODEL=${SKIP_PIPER_MODEL:-0}
pip_install() {
    pip install --timeout "$PIP_TIMEOUT" --retries "$PIP_RETRIES" "$@" -q
}

echo -e "${CYAN}[*] Installing dependencies (this may take a few minutes)...${NC}"

# Blackwell (RTX 50, compute 10.0+) has kernels only in CUDA 12.8+ builds; cu126 covers older GPUs.
# PyPI torch on Linux is the CUDA build with several GB of NVIDIA libraries, so CPU-only uses the CPU index.
TORCH_INDEX=""
CC_MAJOR=""
if [ "$FORCE_CPU" != "1" ] && command -v nvidia-smi &> /dev/null; then
    CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true)
    if [ -n "$CC_MAJOR" ] && [ "$CC_MAJOR" -ge 10 ] 2>/dev/null; then
        TORCH_INDEX="cu130"
    elif [ -n "$CC_MAJOR" ]; then
        TORCH_INDEX="cu126"
    fi
fi
if [ -n "$TORCH_INDEX" ]; then
    echo -e "${GREEN}[✓] GPU detected (compute ${CC_MAJOR}.x) - installing CUDA-enabled PyTorch ($TORCH_INDEX)...${NC}"
    pip_install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
elif [ "$(uname -s)" = "Linux" ]; then
    echo -e "${YELLOW}[!] No GPU or FORCE_CPU=1 - installing CPU PyTorch (training will be slower)${NC}"
    pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo -e "${YELLOW}[!] No GPU detected - using CPU version (training will be slower)${NC}"
    pip_install torch torchvision torchaudio
fi

pip_install -r requirements.txt
pip_install streamlit

# Compatibility patches, same as install.ps1 (torchaudio 2.1+, PyTorch 2.6+); set -e stops on failure
echo -e "${CYAN}[*] Checking torch_audiomentations compatibility...${NC}"
python patch_torch_audiomentations.py
echo -e "${CYAN}[*] Checking DeepPhonemizer compatibility...${NC}"
python patch_dp.py

# Patch speechbrain if needed (for torchaudio 2.1+ compatibility)
SB_FILE=".venv/lib/python*/site-packages/speechbrain/utils/torch_audio_backend.py"
for f in $SB_FILE; do
    if [ -f "$f" ] && grep -q "torchaudio.set_audio_backend" "$f"; then
        echo -e "${CYAN}[*] Patching speechbrain for torchaudio 2.1+ compatibility...${NC}"
        sed -i 's/torchaudio\.set_audio_backend([^)]*)/pass  # Removed: torchaudio.set_audio_backend (deprecated in 2.1+)/g' "$f"
        echo -e "${GREEN}[✓] Patched speechbrain${NC}"
    fi
done

if [ "$SKIP_PIPER_MODEL" != "1" ]; then
    if [ -f "piper-sample-generator/requirements.txt" ]; then
        PIPER_REQ_TMP="$(mktemp)"
        # Avoid torch/torchaudio pins that conflict with newer torch wheels
        grep -vE '^(torch|torchaudio)\b' "piper-sample-generator/requirements.txt" > "$PIPER_REQ_TMP"
        pip_install -r "$PIPER_REQ_TMP"
        rm -f "$PIPER_REQ_TMP"
    fi
else
    echo -e "${YELLOW}[!] SKIP_PIPER_MODEL=1 set - skipping piper-sample-generator requirements.${NC}"
fi

# Download Piper TTS ONNX model if not present
PIPER_MODEL_DIR="piper-sample-generator/models"
PIPER_MODEL="$PIPER_MODEL_DIR/en_US-libritts_r-medium.onnx"
PIPER_CONFIG="$PIPER_MODEL_DIR/en_US-libritts_r-medium.onnx.json"

# Create models directory if needed
mkdir -p "$PIPER_MODEL_DIR"

if [ "$SKIP_PIPER_MODEL" = "1" ]; then
    echo -e "${YELLOW}[!] SKIP_PIPER_MODEL=1 set - skipping Piper TTS model download.${NC}"
else
    # Download ONNX model
    if [ ! -f "$PIPER_MODEL" ]; then
        echo -e "${CYAN}[*] Downloading Piper TTS model (~75MB)...${NC}"
        MODEL_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx"
        if curl -sL "$MODEL_URL" -o "$PIPER_MODEL"; then
            echo -e "${GREEN}[✓] Piper TTS model downloaded${NC}"
        else
            echo -e "${YELLOW}[!] Could not download Piper model${NC}"
            echo "    Please download manually from: $MODEL_URL"
        fi
    else
        echo -e "${GREEN}[✓] Piper TTS model exists${NC}"
    fi

    # Download config file
    if [ ! -f "$PIPER_CONFIG" ]; then
        echo -e "${CYAN}[*] Downloading Piper TTS config...${NC}"
        CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json"
        if curl -sL "$CONFIG_URL" -o "$PIPER_CONFIG"; then
            echo -e "${GREEN}[✓] Piper TTS config downloaded${NC}"
        else
            echo -e "${YELLOW}[!] Could not download config${NC}"
        fi
    fi
fi
if [ "$SKIP_DATA_DOWNLOADS" = "1" ]; then
    echo -e "${YELLOW}[!] SKIP_DATA_DOWNLOADS=1 set - skipping RIR and background audio downloads.${NC}"
else
    # Download Room Impulse Responses for audio augmentation
    if [ ! -d "mit_rirs" ] || [ -z "$(ls -A mit_rirs 2>/dev/null)" ]; then
        echo -e "${CYAN}[*] Downloading Room Impulse Responses (~10MB)...${NC}"
        echo "    This improves training quality with realistic audio augmentation"
        if python download_rirs.py; then
            echo -e "${GREEN}[✓] RIR files downloaded${NC}"
        else
            echo -e "${YELLOW}[!] Could not download RIR files${NC}"
            echo "    Training will still work, but audio augmentation will be limited"
        fi
    else
        echo -e "${GREEN}[✓] RIR files exist${NC}"
    fi

    # Download background audio data (required for training)
    if [ ! -d "audioset_16k" ] || [ -z "$(ls -A audioset_16k/*.wav 2>/dev/null)" ]; then
        echo ""
        echo -e "${CYAN}[*] Downloading background audio data (~3-5 GB)...${NC}"
        echo "    This is required for training and may take 10-30 minutes"
        echo ""
        if python download_data.py; then
            echo -e "${GREEN}[✓] Background audio downloaded${NC}"
        else
            echo -e "${YELLOW}[!] Background audio download failed. Retrying with smaller settings...${NC}"
            if HAWAKE_AUDIOSET_WORKERS=1 HAWAKE_AUDIOSET_PARQUET_FILES=1 HAWAKE_AUDIOSET_MAX_CLIPS=200 python download_data.py; then
                echo -e "${GREEN}[✓] Background audio downloaded with reduced settings${NC}"
            else
                echo -e "${YELLOW}[!] Reduced download failed. Generating synthetic background audio...${NC}"
                python - <<'PY'
from download_data import download_background_audio_fallback
ok = download_background_audio_fallback()
raise SystemExit(0 if ok else 1)
PY
            fi
        fi
    else
        AUDIOSET_COUNT=$(ls -1 audioset_16k/*.wav 2>/dev/null | wc -l)
        echo -e "${GREEN}[✓] Background audio exists ($AUDIOSET_COUNT files)${NC}"
    fi
fi

echo ""
echo -e "${GREEN}========================================"
echo "  Setup Complete!"
echo "========================================${NC}"
echo ""
echo "Starting WebUI..."
echo "(Press Ctrl+C to stop)"
echo ""

streamlit run app.py
