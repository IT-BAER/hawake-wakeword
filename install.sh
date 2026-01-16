#!/bin/bash
# HAwake WakeWord Training - One-liner Setup
# Usage: curl -sSL https://raw.githubusercontent.com/IT-BAER/hawake-wakeword/master/install.sh | bash
# Or: ./install.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
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

echo ""
echo -e "${GREEN}========================================"
echo "  Setup Complete!"
echo "========================================${NC}"
echo ""
echo "Starting WebUI..."
echo "(Press Ctrl+C to stop)"
echo ""

streamlit run app.py
