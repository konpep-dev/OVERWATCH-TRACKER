#!/bin/bash

echo ""
echo "========================================"
echo "     ESP TRACKER PRO - INSTALLER"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed!"
    echo "Please install Python 3.8+"
    exit 1
fi

echo "[OK] Python found"
echo ""

# Install dependencies
echo "[*] Installing dependencies..."
echo "    This may take a few minutes..."
echo ""

pip3 install --upgrade pip
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Installation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "     INSTALLATION COMPLETE!"
echo "========================================"
echo ""
echo "To run ESP Tracker:"
echo "  1. Open IP Webcam on your phone"
echo "  2. Tap 'Start Server'"
echo "  3. Edit config.py with your phone's IP"
echo "  4. Run: python3 main.py"
echo ""
