# HAwake WakeWord Training - One-liner PowerShell Setup
# Usage: git clone https://github.com/IT-BAER/hawake-wakeword.git; cd hawake-wakeword; .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HAwake WakeWord Training Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[✓] Found $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] Python not found! Please install Python 3.10+ first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Create venv if not exists
if (-not (Test-Path ".venv")) {
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "[✓] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[✓] Virtual environment exists" -ForegroundColor Green
}

# Activate venv
Write-Host "[*] Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "[*] Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan

# Check for GPU
$hasGpu = $false
try {
    $null = nvidia-smi 2>&1
    $hasGpu = $LASTEXITCODE -eq 0
} catch {
    $hasGpu = $false
}

if ($hasGpu) {
    Write-Host "[✓] GPU detected - installing CUDA-enabled PyTorch..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
} else {
    Write-Host "[!] No GPU detected - using CPU version (training will be slower)" -ForegroundColor Yellow
    pip install torch torchvision torchaudio -q
}

pip install -r requirements.txt -q
pip install streamlit -q

if (Test-Path "piper-sample-generator\requirements.txt") {
    pip install -r piper-sample-generator\requirements.txt -q
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting WebUI at http://localhost:8501" -ForegroundColor Cyan
Write-Host "(Press Ctrl+C to stop)" -ForegroundColor Gray
Write-Host ""

streamlit run app.py
