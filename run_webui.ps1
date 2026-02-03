# OpenWakeWord Trainer WebUI launcher
# Press Ctrl+C to stop the server (no prompt)

Write-Host "Starting OpenWakeWord Trainer WebUI..." -ForegroundColor Cyan
Write-Host "(Press Ctrl+C to stop the server)" -ForegroundColor Gray
Write-Host ""

$root = $PSScriptRoot
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (Test-Path $venvActivate) {
    Write-Host "Using virtual environment: .venv" -ForegroundColor Green
    & $venvActivate
}

$pythonCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

try {
    & $pythonCmd -m streamlit run app.py
} catch {
    # Silently catch Ctrl+C
}

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Yellow
