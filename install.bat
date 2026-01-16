@echo off
REM HAwake WakeWord Training - One-liner Windows Setup
REM Usage: install.bat

echo.
echo ========================================
echo   HAwake WakeWord Training Setup
echo ========================================
echo.

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Clone required repositories if not present
echo.
echo [*] Checking required repositories...

REM Clone OpenWakeWord if needed
if not exist "openwakeword" (
    echo [*] Cloning openwakeword repository...
    git clone https://github.com/dscripka/openwakeword.git openwakeword
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to clone openwakeword!
        pause
        exit /b 1
    )
    echo [OK] OpenWakeWord cloned
) else (
    echo [OK] OpenWakeWord repository exists
)

REM Clone piper-sample-generator if needed
if not exist "piper-sample-generator" (
    echo [*] Cloning piper-sample-generator repository...
    git clone https://github.com/rhasspy/piper-sample-generator.git piper-sample-generator
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to clone piper-sample-generator!
        pause
        exit /b 1
    )
    echo [OK] Piper sample generator cloned
) else (
    echo [OK] Piper sample generator exists
)

REM Create venv if not exists
if not exist ".venv" (
    echo [*] Creating virtual environment...
    python -m venv .venv
)

REM Activate and install
echo [*] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat

REM Check for GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] GPU detected - installing CUDA-enabled PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
) else (
    echo [*] No GPU detected - using CPU version...
    pip install torch torchvision torchaudio -q
)

pip install -r requirements.txt -q
pip install streamlit -q

if exist "piper-sample-generator\requirements.txt" (
    pip install -r piper-sample-generator\requirements.txt -q
)

REM Download Piper TTS model if not present
set PIPER_MODEL=piper-sample-generator\models\en_US-libritts_r-medium.onnx
set PIPER_CONFIG=piper-sample-generator\models\en_US-libritts_r-medium.onnx.json

REM Create models directory
if not exist "piper-sample-generator\models" mkdir piper-sample-generator\models

REM Download ONNX model
if not exist "%PIPER_MODEL%" (
    echo [*] Downloading Piper TTS model (~75MB)...
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx" -o "%PIPER_MODEL%"
    if exist "%PIPER_MODEL%" (
        echo [OK] Piper TTS model downloaded
    ) else (
        echo [WARNING] Could not download Piper model
        echo Please download manually from: https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx
    )
) else (
    echo [OK] Piper TTS model exists
)

REM Download config file
if not exist "%PIPER_CONFIG%" (
    echo [*] Downloading Piper TTS config...
    curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json" -o "%PIPER_CONFIG%"
    if exist "%PIPER_CONFIG%" (
        echo [OK] Piper TTS config downloaded
    )
)

REM Download Room Impulse Responses for audio augmentation
if not exist "mit_rirs" (
    echo [*] Downloading Room Impulse Responses (~10MB)...
    echo     This improves training quality with realistic audio augmentation
    python download_rirs.py
    if exist "mit_rirs" (
        echo [OK] RIR files downloaded
    ) else (
        echo [WARNING] Could not download RIR files
        echo Training will still work, but audio augmentation will be limited
    )
) else (
    echo [OK] RIR files exist
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Starting WebUI...
echo (Press Ctrl+C to stop)
echo.

streamlit run app.py
