@echo off
:: Launch via PowerShell to avoid "Terminate batch job (Y/N)?" prompt
powershell -ExecutionPolicy Bypass -File "%~dp0run_webui.ps1"
pause