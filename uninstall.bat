@echo off
REM HAwake WakeWord Training - Uninstall (Windows Batch)
REM Usage: uninstall.bat [--full] [--keep-venv] [-y]

setlocal enabledelayedexpansion

echo.
echo HAwake WakeWord Training - Uninstaller
echo =======================================
echo.

REM Check if Python is available
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

cd /d "%~dp0"
python uninstall.py %*

if errorlevel 1 (
    echo.
    echo [!] Uninstall encountered an error
    pause
    exit /b 1
)

pause
