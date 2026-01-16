#!/bin/bash
# HAwake WakeWord Training - One-liner Setup
# Usage: curl -sSL https://raw.githubusercontent.com/IT-BAER/hawake-wakeword/master/install.sh | bash
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

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}[!] Python3 not found! Please install Python 3.10+ first.${NC}"
    exit 1
fi

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
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

echo -e "${CYAN}[*] Installing dependencies (this may take a few minutes)...${NC}"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[✓] GPU detected - installing CUDA-enabled PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
else
    echo -e "${YELLOW}[!] No GPU detected - using CPU version (training will be slower)${NC}"
    pip install torch torchvision torchaudio -q
fi

pip install -r requirements.txt -q
pip install streamlit -q

if [ -f "piper-sample-generator/requirements.txt" ]; then
    pip install -r piper-sample-generator/requirements.txt -q
fi

# Download Piper TTS ONNX model if not present
PIPER_MODEL_DIR="piper-sample-generator/models"
PIPER_MODEL="$PIPER_MODEL_DIR/en_US-libritts_r-medium.onnx"
PIPER_CONFIG="$PIPER_MODEL_DIR/en_US-libritts_r-medium.onnx.json"

# Create models directory if needed
mkdir -p "$PIPER_MODEL_DIR"

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
echo ""
echo -e "${GREEN}========================================"
echo "  Setup Complete!"
echo "========================================${NC}"
echo ""
echo "Starting WebUI..."
echo "(Press Ctrl+C to stop)"
echo ""

streamlit run app.py
