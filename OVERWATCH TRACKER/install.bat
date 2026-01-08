@echo off
title ESP Tracker Installer
color 0A

echo.
echo  ========================================
echo       ESP TRACKER PRO - INSTALLER
echo  ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
echo.

:: Create virtual environment (optional but recommended)
echo [*] Installing dependencies...
echo     This may take a few minutes...
echo.

pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed!
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo  ========================================
echo       INSTALLATION COMPLETE!
echo  ========================================
echo.
echo  To run ESP Tracker:
echo    1. Open IP Webcam on your phone
echo    2. Tap "Start Server"
echo    3. Edit config.py with your phone's IP
echo    4. Run: python main.py
echo.
echo  ========================================
echo.

pause
