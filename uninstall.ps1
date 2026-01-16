# HAwake WakeWord Training - Uninstall (PowerShell)
# Usage: .\uninstall.ps1 [-Full] [-KeepVenv] [-Yes]

param(
    [switch]$Full,
    [switch]$KeepVenv,
    [switch]$Yes,
    [Alias('y')]
    [switch]$Confirm
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "HAwake WakeWord Training - Uninstaller" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# Check for Python
$python = $null
foreach ($cmd in @("python", "python3", "py -3")) {
    try {
        $version = & ([scriptblock]::Create("$cmd --version 2>&1"))
        if ($version -match "Python 3\.(\d+)") {
            $minor = [int]$matches[1]
            if ($minor -ge 10) {
                $python = $cmd
                break
            }
        }
    } catch {}
}

if (-not $python) {
    Write-Host "[ERROR] Python 3.10+ is required but not found" -ForegroundColor Red
    Write-Host "Please install Python 3.10 or later from https://python.org"
    exit 1
}

# Navigate to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Build arguments
$args = @()
if ($Full) { $args += "--full" }
if ($KeepVenv) { $args += "--keep-venv" }
if ($Yes -or $Confirm) { $args += "-y" }

# Run the uninstall script
& $python uninstall.py @args
