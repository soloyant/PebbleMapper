@echo off
rem PebbleMapper's starter. Written by the installer; an update never changes it.
cd /d "%~dp0"
if not exist ".conda-location" (
  echo PebbleMapper is not installed in this folder. Run "Install PebbleMapper.bat".
  pause
  exit /b 1
)
set /p CONDA_BAT=<".conda-location"
for %%I in ("%CONDA_BAT%\..\..") do set "CONDA_ROOT=%%~fI"
"%CONDA_ROOT%\python.exe" "%~dp0pm_starter.py" %*
if errorlevel 1 pause
