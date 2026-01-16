#!/bin/bash
# HAwake WakeWord Training - Uninstall (macOS/Linux)
# Usage: ./uninstall.sh [--full] [--keep-venv] [-y]

set -e

echo ""
echo "HAwake WakeWord Training - Uninstaller"
echo "======================================="
echo ""

# Check for Python 3.10+
check_python() {
    if command -v python3 &> /dev/null; then
        version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            echo "python3"
            return 0
        fi
    fi
    
    if command -v python &> /dev/null; then
        version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            echo "python"
            return 0
        fi
    fi
    
    return 1
}

PYTHON=$(check_python)
if [ $? -ne 0 ]; then
    echo "[ERROR] Python 3.10+ is required but not found"
    echo "Please install Python 3.10 or later"
    exit 1
fi

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run the uninstall script
$PYTHON uninstall.py "$@"
