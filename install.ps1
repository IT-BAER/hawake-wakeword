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

# Clone required repositories if not present
Write-Host ""
Write-Host "[*] Checking required repositories..." -ForegroundColor Cyan

# Clone OpenWakeWord if needed
if (-not (Test-Path "openwakeword")) {
    Write-Host "[*] Cloning openwakeword repository..." -ForegroundColor Cyan
    git clone https://github.com/dscripka/openwakeword.git openwakeword
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[✗] Failed to clone openwakeword!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[✓] OpenWakeWord cloned" -ForegroundColor Green
} else {
    Write-Host "[✓] OpenWakeWord repository exists" -ForegroundColor Green
}

# Clone piper-sample-generator if needed
if (-not (Test-Path "piper-sample-generator")) {
    Write-Host "[*] Cloning piper-sample-generator repository..." -ForegroundColor Cyan
    git clone https://github.com/rhasspy/piper-sample-generator.git piper-sample-generator
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[✗] Failed to clone piper-sample-generator!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[✓] Piper sample generator cloned" -ForegroundColor Green
} else {
    Write-Host "[✓] Piper sample generator exists" -ForegroundColor Green
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

# Check for GPU and determine compute capability
$hasGpu = $false
$needsNightly = $false
$computeCap = ""
try {
    $gpuInfo = nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasGpu = $true
        $computeCap = ($gpuInfo -split ",")[0].Trim()
        $gpuName = ($gpuInfo -split ",")[1].Trim()
        
        # Check if GPU is RTX 50-series (Blackwell, compute capability 12.0+)
        # or RTX 40-series (Ada Lovelace, compute capability 8.9)
        $majorCap = [int]($computeCap -split "\.")[0]
        if ($majorCap -ge 12) {
            $needsNightly = $true
            Write-Host "[!] Detected $gpuName (compute $computeCap) - RTX 50-series/Blackwell detected" -ForegroundColor Yellow
            Write-Host "    Using PyTorch nightly for best GPU support" -ForegroundColor Yellow
        } else {
            Write-Host "[✓] Detected $gpuName (compute $computeCap)" -ForegroundColor Green
        }
    }
} catch {
    $hasGpu = $false
}

if ($hasGpu) {
    if ($needsNightly) {
        Write-Host "[*] Installing PyTorch nightly with CUDA 12.8 (for RTX 50-series support)..." -ForegroundColor Cyan
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -q
    } else {
        Write-Host "[✓] GPU detected - installing CUDA-enabled PyTorch..." -ForegroundColor Green
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    }
} else {
    Write-Host "[!] No GPU detected - using CPU version (training will be slower)" -ForegroundColor Yellow
    pip install torch torchvision torchaudio -q
}

pip install -r requirements.txt -q
pip install streamlit -q

if (Test-Path "piper-sample-generator\requirements.txt") {
    pip install -r piper-sample-generator\requirements.txt -q
}

# Download Piper TTS ONNX model if not present
$piperModelDir = "piper-sample-generator\models"
$piperModel = "$piperModelDir\en_US-libritts_r-medium.onnx"
$piperConfig = "$piperModelDir\en_US-libritts_r-medium.onnx.json"

# Create models directory if needed
if (-not (Test-Path $piperModelDir)) {
    New-Item -ItemType Directory -Path $piperModelDir -Force | Out-Null
}

# Download ONNX model
if (-not (Test-Path $piperModel)) {
    Write-Host "[*] Downloading Piper TTS model (~75MB)..." -ForegroundColor Cyan
    $modelUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx"
    try {
        Invoke-WebRequest -Uri $modelUrl -OutFile $piperModel -UseBasicParsing
        Write-Host "[✓] Piper TTS model downloaded" -ForegroundColor Green
    } catch {
        Write-Host "[!] Could not download Piper model: $_" -ForegroundColor Yellow
        Write-Host "    Please download manually from: $modelUrl" -ForegroundColor Yellow
    }
} else {
    Write-Host "[✓] Piper TTS model exists" -ForegroundColor Green
}

# Download config file
if (-not (Test-Path $piperConfig)) {
    Write-Host "[*] Downloading Piper TTS config..." -ForegroundColor Cyan
    $configUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json"
    try {
        Invoke-WebRequest -Uri $configUrl -OutFile $piperConfig -UseBasicParsing
        Write-Host "[✓] Piper TTS config downloaded" -ForegroundColor Green
    } catch {
        Write-Host "[!] Could not download config: $_" -ForegroundColor Yellow
    }
}

# Download Room Impulse Responses for audio augmentation
if (-not (Test-Path "mit_rirs") -or (Get-ChildItem "mit_rirs" -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
    Write-Host "[*] Downloading Room Impulse Responses (~10MB)..." -ForegroundColor Cyan
    Write-Host "    This improves training quality with realistic audio augmentation" -ForegroundColor Gray
    try {
        python download_rirs.py
        Write-Host "[✓] RIR files downloaded" -ForegroundColor Green
    } catch {
        Write-Host "[!] Could not download RIR files: $_" -ForegroundColor Yellow
        Write-Host "    Training will still work, but audio augmentation will be limited" -ForegroundColor Yellow
    }
} else {
    Write-Host "[✓] RIR files exist" -ForegroundColor Green
}

# Download background audio data (required for training)
if (-not (Test-Path "audioset_16k") -or (Get-ChildItem "audioset_16k" -Filter "*.wav" -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
    Write-Host "" -ForegroundColor Cyan
    Write-Host "[*] Downloading background audio data (~3-5 GB)..." -ForegroundColor Cyan
    Write-Host "    This is required for training and may take 10-30 minutes" -ForegroundColor Gray
    Write-Host "" -ForegroundColor Gray
    try {
        python download_data.py
        Write-Host "[✓] Background audio downloaded" -ForegroundColor Green
    } catch {
        Write-Host "[!] Could not download background audio: $_" -ForegroundColor Yellow
        Write-Host "    Run 'python download_data.py' manually before training" -ForegroundColor Yellow
    }
} else {
    $audiosetCount = (Get-ChildItem "audioset_16k" -Filter "*.wav" | Measure-Object).Count
    Write-Host "[✓] Background audio exists ($audiosetCount files)" -ForegroundColor Green
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
