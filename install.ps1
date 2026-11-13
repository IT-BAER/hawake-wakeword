# HAwake WakeWord Training - One-liner PowerShell Setup
# Usage: git clone https://github.com/IT-BAER/hawake-wakeword.git; cd hawake-wakeword; .\install.ps1
# Options:
#   -Fast    Skip data downloads (use existing data or download later)
#   -Minimal Skip optional dependencies

param(
    [switch]$Fast,
    [switch]$Minimal
)

$ErrorActionPreference = "Stop"

# A failed native step (pip, python) stops the setup instead of ending in "Setup Complete!"
function Assert-Success([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[✗] $Step failed (exit code $LASTEXITCODE). Setup stopped." -ForegroundColor Red
        exit 1
    }
}

# Python 3.11, like install.sh: the py launcher's 3.11 first, else python on PATH if it is 3.11
function Find-Python311 {
    # Windows PowerShell 5.1 turns native stderr into terminating errors under "Stop"
    $ErrorActionPreference = "Continue"
    $probe = "import sys; print(sys.executable if sys.version_info[:2] == (3, 11) else '')"
    $exe = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        # Without a 3.11, py prints its error and runtime list to stdout and exits non-zero
        $exe = py -3.11 -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) { $exe = $null }
    }
    if (-not $exe -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $exe = python -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) { $exe = $null }
    }
    if ($exe) { return "$exe".Trim() }
    return $null
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HAwake WakeWord Training Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
if ($Fast) {
    Write-Host "  Mode: FAST (skipping data downloads)" -ForegroundColor Yellow
}
if ($Minimal) {
    Write-Host "  Mode: MINIMAL (core dependencies only)" -ForegroundColor Yellow
}
Write-Host ""

# Check Python
$python = Find-Python311
if (-not $python) {
    Write-Host "[✗] Python 3.11 not found (py -3.11 or python on PATH)! Please install Python 3.11 first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
$pythonVersion = & $python --version 2>&1
Write-Host "[✓] Found $pythonVersion ($python)" -ForegroundColor Green

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
    & $python -m venv .venv
    Assert-Success "Creating the virtual environment"
    Write-Host "[✓] Virtual environment created" -ForegroundColor Green
} else {
    # A .venv from another Python brings back the wheel failures above
    if (-not (Select-String -Path ".venv\pyvenv.cfg" -Pattern '^version(_info)?\s*=\s*3\.11\.' -Quiet -ErrorAction SilentlyContinue)) {
        Write-Host "[✗] .venv was not created with Python 3.11. Delete the .venv folder and re-run." -ForegroundColor Red
        exit 1
    }
    Write-Host "[✓] Virtual environment exists" -ForegroundColor Green
}

# Activate venv
Write-Host "[*] Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "[*] Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan

# Check for GPU and determine compute capability
$hasGpu = $false
$needsCuda13 = $false
$computeCap = ""
try {
    $gpuInfo = nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasGpu = $true
        $computeCap = ($gpuInfo -split ",")[0].Trim()
        $gpuName = ($gpuInfo -split ",")[1].Trim()
        
        # Blackwell (RTX 50, compute 10.0+) has kernels only in CUDA 12.8+ builds; cu126 covers older GPUs
        $majorCap = [int]($computeCap -split "\.")[0]
        if ($majorCap -ge 10) {
            $needsCuda13 = $true
            Write-Host "[✓] Detected $gpuName (compute $computeCap) - Blackwell, using the CUDA 13.0 build" -ForegroundColor Green
        } else {
            Write-Host "[✓] Detected $gpuName (compute $computeCap)" -ForegroundColor Green
        }
    }
} catch {
    $hasGpu = $false
}

if ($hasGpu) {
    if ($needsCuda13) {
        Write-Host "[*] Installing CUDA-enabled PyTorch (CUDA 13.0, needs NVIDIA driver 580+)..." -ForegroundColor Cyan
        Write-Host "    Note: PyTorch is ~2.5GB - this may take a few minutes" -ForegroundColor Gray
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    } else {
        Write-Host "[✓] GPU detected - installing CUDA-enabled PyTorch..." -ForegroundColor Green
        Write-Host "    Note: PyTorch is ~2.5GB - this may take a few minutes" -ForegroundColor Gray
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    }
} else {
    Write-Host "[!] No GPU detected - using CPU version (training will be slower)" -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}
Assert-Success "PyTorch install"

pip install -r requirements.txt -q
Assert-Success "pip install -r requirements.txt"
pip install streamlit -q
Assert-Success "pip install streamlit"

# Patch torch_audiomentations for torchaudio 2.1+ compatibility
# In newer torchaudio versions, these functions were removed:
# - torchaudio.set_audio_backend()
# - torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE
# - torchaudio.info()
Write-Host "[*] Checking torch_audiomentations compatibility..." -ForegroundColor Cyan
python patch_torch_audiomentations.py
Assert-Success "patch_torch_audiomentations.py"

# Patch DeepPhonemizer (dp) for PyTorch 2.6+ compatibility
# PyTorch 2.6 changed weights_only default to True, breaking checkpoint loading
Write-Host "[*] Checking DeepPhonemizer compatibility..." -ForegroundColor Cyan
python patch_dp.py
Assert-Success "patch_dp.py"

# Patch speechbrain for torchaudio 2.1+ compatibility
$sbFile = ".venv\Lib\site-packages\speechbrain\utils\torch_audio_backend.py"
if (Test-Path $sbFile) {
    $content = Get-Content $sbFile -Raw
    if ($content -match 'torchaudio\.set_audio_backend') {
        Write-Host "[*] Patching speechbrain for torchaudio 2.1+ compatibility..." -ForegroundColor Cyan
        $content = $content -replace 'torchaudio\.set_audio_backend\([^)]+\)', 'pass  # Removed: torchaudio.set_audio_backend (deprecated in 2.1+)'
        Set-Content $sbFile $content -NoNewline
        Write-Host "[✓] Patched speechbrain" -ForegroundColor Green
    }
}

if (Test-Path "piper-sample-generator\requirements.txt") {
    pip install -r piper-sample-generator\requirements.txt -q
    Assert-Success "pip install -r piper-sample-generator\requirements.txt"
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

# Skip data downloads in Fast mode
if ($Fast) {
    Write-Host ""
    Write-Host "[*] Fast mode: Skipping data downloads" -ForegroundColor Yellow
    Write-Host "    Run 'python download_data.py' manually before training" -ForegroundColor Gray
} else {
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
    # Only real AudioSet clips count: noise_*.wav from the synthetic fallback must not skip a retry
    if (-not (Test-Path "audioset_16k") -or (Get-ChildItem "audioset_16k" -Filter "audioset_*.wav" -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0 -or -not (Test-Path "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")) {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "[*] Downloading training data (about 19 GB)..." -ForegroundColor Cyan
        Write-Host "" -ForegroundColor Gray
        try {
            python download_data.py
            Write-Host "[✓] Background audio downloaded" -ForegroundColor Green
        } catch {
            Write-Host "[!] Could not download background audio: $_" -ForegroundColor Yellow
            Write-Host "    Run 'python download_data.py' manually before training" -ForegroundColor Yellow
        }
    } else {
        $audiosetCount = (Get-ChildItem "audioset_16k" -Filter "audioset_*.wav" | Measure-Object).Count
        Write-Host "[✓] Background audio exists ($audiosetCount files)" -ForegroundColor Green
    }
}

if ((Get-ChildItem "audioset_16k" -Filter "audioset_*.wav" -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
    Write-Host ""
    Write-Host "[!] WARNING: no real background audio (AudioSet download failed or skipped)." -ForegroundColor Yellow
    Write-Host "    A model trained now false-triggers often. Run: python download_data.py" -ForegroundColor Yellow
}
if (-not (Test-Path "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")) {
    Write-Host ""
    Write-Host "[!] WARNING: training features (17 GB) missing. A model trained now reacts to normal speech." -ForegroundColor Yellow
    Write-Host "    Run: python download_data.py" -ForegroundColor Yellow
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
