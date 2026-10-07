@echo off
REM Double-click this file to install PebbleMapper. It runs install.ps1, which
REM installs conda if needed, builds the Python environment, downloads the
REM model weights and puts a PebbleMapper icon on the desktop.
title Install PebbleMapper
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
    echo.
    echo  The installation did not finish. The messages above say why.
    echo  Fix the problem and double-click "Install PebbleMapper" again:
    echo  it continues where it stopped.
) else (
    echo.
    echo  Done. You can close this window.
)
echo.
pause
