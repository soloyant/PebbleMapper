@echo off
REM ===========================================================================
REM  PebbleMapper - Windows Launcher
REM ===========================================================================
REM  Double-click this file to start the GUI. To create a desktop shortcut with
REM  a custom icon: right-click -> Send to -> Desktop (create shortcut), then
REM  right-click the shortcut -> Properties -> Change Icon -> point to gui\icon.ico.
REM ===========================================================================

setlocal

REM === BRANDING: product display name (single edit point for this script) ====
REM  Keep in sync with functions/branding.py APP_NAME. The banners below read it.
set "APP_NAME=PebbleMapper"
REM ===========================================================================

REM --- Locate the repo root. The script is normally placed at the repo root,
REM --- but if it ends up inside gui\ we transparently walk up one level so
REM --- the user doesn't have to figure out where it should live.
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"
REM Detect "I am inside gui\" by checking whether the basename equals 'gui'.
for %%I in ("%REPO_ROOT%") do set "REPO_BASENAME=%%~nxI"
if /I "%REPO_BASENAME%"=="gui" (
    REM Walk up one level - script is in gui\, repo root is the parent
    for %%I in ("%REPO_ROOT%\..") do set "REPO_ROOT=%%~fI"
)
cd /d "%REPO_ROOT%"

REM --- Sanity-check the package layout. Both gui/ and functions/ must be ---
REM --- importable Python packages, i.e. they must contain __init__.py.    ---
if not exist "%REPO_ROOT%\gui\__init__.py" (
    echo.
    echo  ERROR: Missing %REPO_ROOT%\gui\__init__.py
    echo  Create an empty file at that path so 'gui' is a Python package.
    echo  ^(touch the file or 'echo. ^> gui\__init__.py'^)
    echo.
    pause
    exit /b 1
)
if not exist "%REPO_ROOT%\functions\__init__.py" (
    echo.
    echo  ERROR: Missing %REPO_ROOT%\functions\__init__.py
    echo  Create an empty file at that path so 'functions' is a Python package.
    echo.
    pause
    exit /b 1
)
if not exist "%REPO_ROOT%\gui\app.py" (
    echo.
    echo  ERROR: Missing %REPO_ROOT%\gui\app.py
    echo  Has the repo been fully cloned?
    echo.
    pause
    exit /b 1
)

REM --- Locate conda. The installer records the conda it used in .conda-location;
REM --- without it, try the common Miniforge/Miniconda/Anaconda locations. ---
set "CONDA_BAT="
if exist "%REPO_ROOT%\.conda-location" set /p CONDA_BAT=<"%REPO_ROOT%\.conda-location"
if not defined CONDA_BAT goto :conda_search
if exist "%CONDA_BAT%" goto :conda_found
set "CONDA_BAT="
:conda_search
if exist "%SystemDrive%\miniforge3\condabin\conda.bat"      set "CONDA_BAT=%SystemDrive%\miniforge3\condabin\conda.bat"
if exist "%LOCALAPPDATA%\miniforge3\condabin\conda.bat"     set "CONDA_BAT=%LOCALAPPDATA%\miniforge3\condabin\conda.bat"
if exist "%USERPROFILE%\miniforge3\condabin\conda.bat"      set "CONDA_BAT=%USERPROFILE%\miniforge3\condabin\conda.bat"
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat"      set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat"       set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if exist "%PROGRAMDATA%\miniconda3\condabin\conda.bat"      set "CONDA_BAT=%PROGRAMDATA%\miniconda3\condabin\conda.bat"
if exist "%PROGRAMDATA%\anaconda3\condabin\conda.bat"       set "CONDA_BAT=%PROGRAMDATA%\anaconda3\condabin\conda.bat"
if exist "C:\ProgramData\miniconda3\condabin\conda.bat"     set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"

if "%CONDA_BAT%"=="" (
    echo.
    echo  ERROR: conda was not found.
    echo  Double-click "Install PebbleMapper" in the PebbleMapper folder first.
    echo  If conda is installed somewhere unusual, write the full path of its
    echo  condabin\conda.bat into a file named .conda-location in that folder.
    echo.
    pause
    exit /b 1
)

:conda_found
REM --- Activate the env ---
call "%CONDA_BAT%" activate maskrcnn
if errorlevel 1 (
    echo.
    echo  ERROR: Failed to activate the 'maskrcnn' conda environment.
    echo  Double-click "Install PebbleMapper" in the PebbleMapper folder first.
    echo.
    pause
    exit /b 1
)

REM --- Confirm nicegui is installed ---
python -c "import nicegui" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: nicegui is not installed in the maskrcnn env.
    echo  Double-click "Install PebbleMapper" again to repair the installation.
    echo.
    pause
    exit /b 1
)

REM --- HEIC/HEIF photographs need the pillow-heif plugin; warn, do not stop ---
python -c "import pillow_heif" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  NOTE: HEIC/HEIF photographs will not open: pillow-heif is not installed.
    echo  To add it:   conda activate maskrcnn ^&^& pip install pillow-heif
    echo.
)

REM --- Make sure the repo root is on sys.path so 'from functions.X import Y' ---
REM --- resolves. python -m gui.app implicitly does this when run from the    ---
REM --- repo root, but setting PYTHONPATH explicitly is belt-and-suspenders.  ---
set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"

REM --- Launch ---
echo.
echo  Starting %APP_NAME% GUI...
echo  The browser will open automatically. To stop the server, close this window
echo  or press Ctrl+C in this terminal.
echo.

REM -u: unbuffered stdout so the "Server starting at http://..." line shows
REM     up immediately. -m gui.app: run gui as a package so package-relative
REM     imports (from functions.X) work.
python -u -m gui.app

REM --- If we get here, the server stopped. Pause so the user can see any
REM --- final messages or tracebacks before the terminal closes.
echo.
echo  GUI server stopped. Press any key to close this window.
pause >nul

endlocal
