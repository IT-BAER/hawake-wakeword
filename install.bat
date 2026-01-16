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

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Starting WebUI...
echo (Press Ctrl+C to stop)
echo.

streamlit run app.py
