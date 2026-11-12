@echo off
REM HAwake WakeWord Training - One-liner Windows Setup
REM Usage: install.bat

echo.
echo ========================================
echo   HAwake WakeWord Training Setup
echo ========================================
echo.

REM Python 3.11, like install.sh: py -3.11 first, else python on PATH if it is 3.11
set "PYTHON="
REM call: a py/python batch shim (pyenv-win) would otherwise end this script
call py -3.11 -c "import sys" >nul 2>&1 && set "PYTHON=py -3.11"
if not defined PYTHON call python -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" >nul 2>&1 && set "PYTHON=python"
if not defined PYTHON (
    echo [ERROR] Python 3.11 not found ^(py -3.11 or python on PATH^)! Please install Python 3.11 first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Using Python 3.11: %PYTHON%

REM Clone required repositories if not present
echo.
echo [*] Checking required repositories...

REM Clone OpenWakeWord if needed
if not exist "openwakeword" (
    echo [*] Cloning openwakeword repository...
    git clone https://github.com/dscripka/openwakeword.git openwakeword
    if errorlevel 1 (
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
    if errorlevel 1 (
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
    call %PYTHON% -m venv .venv || (set "STEP=Creating the virtual environment" & goto :fail)
) else (
    REM A .venv from another Python brings back the wheel failures
    findstr /r /b /c:"version *= *3\.11\." /c:"version_info *= *3\.11\." .venv\pyvenv.cfg >nul 2>&1 || (
        echo [ERROR] .venv was not created with Python 3.11. Delete the .venv folder and re-run.
        pause
        exit /b 1
    )
)

REM Activate and install
echo [*] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat

REM Check for GPU: Blackwell (RTX 50, compute 10.0+) has kernels only in CUDA 12.8+ builds; cu126 covers older GPUs
set "TORCH_INDEX="
set "CC_MAJOR="
for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do set "CC_MAJOR=%%a"
if defined CC_MAJOR (
    if %CC_MAJOR% GEQ 10 (set "TORCH_INDEX=cu130") else (set "TORCH_INDEX=cu126")
)
if defined TORCH_INDEX (
    echo [*] GPU detected - installing CUDA-enabled PyTorch ^(%TORCH_INDEX%^)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%TORCH_INDEX% -q
) else (
    echo [*] No GPU detected - using CPU version...
    pip install torch torchvision torchaudio -q
)
if errorlevel 1 (set "STEP=PyTorch install" & goto :fail)

pip install -r requirements.txt -q || (set "STEP=pip install -r requirements.txt" & goto :fail)
pip install streamlit -q || (set "STEP=pip install streamlit" & goto :fail)

REM Compatibility patches, same as install.ps1 (torchaudio 2.1+, PyTorch 2.6+)
echo [*] Checking torch_audiomentations compatibility...
python patch_torch_audiomentations.py || (set "STEP=patch_torch_audiomentations.py" & goto :fail)
echo [*] Checking DeepPhonemizer compatibility...
python patch_dp.py || (set "STEP=patch_dp.py" & goto :fail)
echo [*] Checking speechbrain compatibility...
python -c "import pathlib,re; p=pathlib.Path(r'.venv\Lib\site-packages\speechbrain\utils\torch_audio_backend.py'); s=p.read_bytes() if p.exists() else b''; n=re.sub(rb'torchaudio\.set_audio_backend\([^)]+\)', b'pass  # Removed: torchaudio.set_audio_backend (deprecated in 2.1+)', s); s == n or p.write_bytes(n)" || (set "STEP=speechbrain patch" & goto :fail)

if exist "piper-sample-generator\requirements.txt" (
    pip install -r piper-sample-generator\requirements.txt -q || (set "STEP=pip install -r piper-sample-generator\requirements.txt" & goto :fail)
)

REM Download Piper TTS model if not present
set PIPER_MODEL=piper-sample-generator\models\en_US-libritts_r-medium.onnx
set PIPER_CONFIG=piper-sample-generator\models\en_US-libritts_r-medium.onnx.json

REM Create models directory
if not exist "piper-sample-generator\models" mkdir piper-sample-generator\models

REM Download ONNX model
if not exist "%PIPER_MODEL%" (
    echo [*] Downloading Piper TTS model ^(~75MB^)...
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
    echo [*] Downloading Room Impulse Responses ^(~10MB^)...
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

REM Download background audio data (required for training)
if not exist "audioset_16k\*.wav" (
    echo.
    echo [*] Downloading background audio data ^(~3-5 GB^)...
    echo     This is required for training and may take 10-30 minutes
    echo.
    python download_data.py
    if exist "audioset_16k\*.wav" (
        echo [OK] Background audio downloaded
    ) else (
        echo [WARNING] Could not download background audio
        echo Run 'python download_data.py' manually before training
    )
) else (
    echo [OK] Background audio exists
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
exit /b

:fail
echo [ERROR] %STEP% failed. Setup stopped.
pause
exit /b 1
