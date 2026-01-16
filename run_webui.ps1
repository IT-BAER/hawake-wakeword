# OpenWakeWord Trainer WebUI launcher
# Press Ctrl+C to stop the server (no prompt)

Write-Host "Starting OpenWakeWord Trainer WebUI..." -ForegroundColor Cyan
Write-Host "(Press Ctrl+C to stop the server)" -ForegroundColor Gray
Write-Host ""

try {
    python -m streamlit run app.py
} catch {
    # Silently catch Ctrl+C
}

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Yellow
