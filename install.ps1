<#
.SYNOPSIS
    Install PebbleMapper: conda (if missing), the Python environment, the model
    weights and a desktop shortcut.

.DESCRIPTION
    Normally started by double-clicking "Install PebbleMapper.bat".

    1. Finds conda. If there is none, downloads Miniforge (the conda-forge
       distribution of conda, BSD licence) and installs it for the current user
       only, without changing PATH or the system Python.
    2. Detects an NVIDIA GPU and builds the matching 'maskrcnn' environment from
       environment.yml (the CUDA packages are left out on a machine without one).
    3. Downloads the trained Mask R-CNN weights (367 MB) from Zenodo.
    4. Checks the result and puts a PebbleMapper shortcut on the desktop.

    Running it again is safe: an existing environment is left alone unless
    -Force is given, and the weights are not downloaded twice.

.PARAMETER Force
    Delete and rebuild the environment even if it already exists.

.PARAMETER SkipWeights
    Do not download the Mask R-CNN weights. Everything except the Detect tab
    works without them; fetch them later with `python -m detectors.download_weights`.

.PARAMETER Cpu
    Build the CPU-only environment even if an NVIDIA GPU is present.

.PARAMETER NoShortcut
    Do not create the desktop shortcut.

.PARAMETER CondaRoot
    Use the conda installed in this folder, or install Miniforge there when
    the folder holds none, instead of looking for an existing conda.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -Cpu -SkipWeights
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipWeights,
    [switch]$Cpu,
    [switch]$NoShortcut,
    [string]$CondaRoot
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$weightsPending = $false
$envName = 'maskrcnn'
$miniforgeUrl = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'

function Say([string]$text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Ok ([string]$text) { Write-Host "    $text" -ForegroundColor Green }
function Warn([string]$text) { Write-Host "    $text" -ForegroundColor Yellow }
function Fail([string]$text) { Write-Host ""; Write-Host $text -ForegroundColor Red; exit 1 }

Say "PebbleMapper install"
Write-Host "    folder: $repo"

# ---------------------------------------------------------------- conda ----
# A default Windows install does not put conda on PATH outside the Anaconda
# Prompt, so look where the installers put it before installing one.
function Find-Conda {
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) { return $cmd.Source }
    $candidates = @(
        $env:CONDA_EXE,
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniforge3\Scripts\conda.exe",
        "$env:SystemDrive\miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniforge3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

# Windows limits paths to 260 characters unless long paths are enabled (an
# administrator setting). The deepest file TensorFlow 2.10 installs lies 202
# characters below the conda folder, so without long paths that folder may
# be at most 57 characters long, or pip fails halfway through the build.
$maxRoot = 57
$longPaths = $false
try {
    $longPaths = ((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
        -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled -eq 1)
} catch { }
function Assert-ShortRoot([string]$root) {
    if ($longPaths -or $root.Length -le $maxRoot) { return }
    Fail ("The conda folder $root is $($root.Length) characters long; without Windows long " +
          "paths it must be at most $maxRoot. Run the installer again with a shorter one, " +
          "for example:  .\install.ps1 -CondaRoot $env:SystemDrive\miniforge3")
}

if ($CondaRoot) {
    Assert-ShortRoot $CondaRoot
    $conda = Join-Path $CondaRoot 'Scripts\conda.exe'
    if (-not (Test-Path $conda)) { $conda = $null }
}
else {
    $conda = Find-Conda
}
if ($null -eq $conda) {
    # conda breaks on install paths with spaces or non-ASCII characters, which
    # a user name can contain, and on over-long ones: fall back to the root of
    # the system drive then.
    $target = "$env:USERPROFILE\miniforge3"
    if ($target -match '[ ]' -or $target -match '[^\x00-\x7F]' -or
        (-not $longPaths -and $target.Length -gt $maxRoot)) { $target = "$env:SystemDrive\miniforge3" }
    if ($CondaRoot) { $target = $CondaRoot }
    Say "conda was not found: installing Miniforge into $target"
    $exe = Join-Path $env:TEMP 'Miniforge3-Windows-x86_64.exe'
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $miniforgeUrl -OutFile $exe -UseBasicParsing
    }
    catch {
        Fail "Could not download Miniforge. Check the internet connection and run the installer again.`n($($_.Exception.Message))"
    }
    Ok "downloaded the Miniforge installer"
    $p = Start-Process -FilePath $exe -Wait -PassThru -ArgumentList @(
        '/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', '/S', "/D=$target")
    Remove-Item $exe -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0 -or -not (Test-Path "$target\Scripts\conda.exe")) {
        Fail "Miniforge did not install (exit code $($p.ExitCode))."
    }
    $conda = "$target\Scripts\conda.exe"
}
Ok "conda: $conda"
Assert-ShortRoot (Split-Path (Split-Path $conda))
# Tell the launcher which conda holds the environment, so a machine with more
# than one conda starts the right one.
$condaBat = Join-Path (Split-Path (Split-Path $conda)) 'condabin\conda.bat'
Set-Content -Path (Join-Path $repo '.conda-location') -Value $condaBat -Encoding ascii

# ------------------------------------------------------------------ GPU ----
$useGpu = -not $Cpu
if ($useGpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $smi) {
        $useGpu = $false
        Warn "no NVIDIA GPU found: building the processor-only environment"
    }
    else {
        $name = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
        if ([string]::IsNullOrWhiteSpace($name)) {
            $useGpu = $false
            Warn "nvidia-smi reported no GPU: building the processor-only environment"
        }
        else {
            Ok "GPU: $($name.Trim())"
        }
    }
}
else {
    Warn "-Cpu given: building the processor-only environment"
}

# ------------------------------------------------------- environment file --
$envFile = Join-Path $repo 'environment.yml'
if (-not (Test-Path $envFile)) { Fail "environment.yml is missing from $repo" }
if (-not $useGpu) {
    # environment.yml pins cudatoolkit and cudnn for GPU machines. Drop exactly
    # those two lines rather than keep a second file that would drift.
    $src = Get-Content $envFile
    $out = $src | Where-Object { $_ -notmatch '^\s*-\s*(cudatoolkit|cudnn)\s*=' }
    $dropped = $src.Count - $out.Count
    if ($dropped -ne 2) {
        Fail "Expected to drop exactly 2 CUDA lines from environment.yml, dropped $dropped. The file has changed shape; fix this script."
    }
    $envFile = Join-Path $repo 'environment-cpu.yml'
    $out | Set-Content $envFile -Encoding utf8
    Ok "wrote environment-cpu.yml (CUDA packages removed)"
}

# ----------------------------------------------------------- environment ---
$existing = (& $conda env list) | Where-Object { $_ -match "^\s*$envName\s" }
if ($existing -and $Force) {
    Say "removing the existing '$envName' environment (-Force)"
    & $conda env remove -n $envName -y
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $existing = $null
}
if ($existing) {
    Ok "environment '$envName' already exists, left as it is (use -Force to rebuild)"
}
else {
    Say "building the '$envName' environment: 20 to 60 minutes and about 7 GB of disk"
    & $conda env create -n $envName -f $envFile
    if ($LASTEXITCODE -ne 0) { Fail "conda could not build the environment. See the messages above." }
    Ok "environment built"
}

# --------------------------------------------------------------- weights ---
# 'conda run' only from here on. Some conda installs carry plugins that break
# it (Anaconda's auth plugin), so it runs without plugins. Not earlier: the
# Miniforge installer and the libmamba solver of 'env create' are plugins.
$env:CONDA_NO_PLUGINS = 'true'
$weights = Join-Path $repo 'model_weights\mask_rcnn_clasts.h5'
# The folder the weights go in, so a user placing the file by hand finds it.
New-Item -ItemType Directory -Force (Join-Path $repo 'model_weights') | Out-Null
if ($SkipWeights) {
    Warn "-SkipWeights given: fetch them later with 'python -m detectors.download_weights'"
}
elseif (Test-Path $weights) {
    Ok "weights already present: model_weights\mask_rcnn_clasts.h5"
}
else {
    Say "downloading the Mask R-CNN weights (367 MB) from Zenodo"
    & $conda run -n $envName --no-capture-output --cwd $repo python -m detectors.download_weights
    if ($LASTEXITCODE -eq 4) {
        # The record is not public yet: the message above says how to get
        # the file. Not an install failure.
        Warn "the weights are not installed yet (see the message above);"
        Warn "every tab except Detect works without them."
        $weightsPending = $true
    }
    elseif ($LASTEXITCODE -ne 0) {
        Warn "the download did not finish. Run the installer again later to retry;"
        Warn "every tab except Detect works without the weights."
    }
    else {
        Ok "weights in model_weights\"
    }
}

# ----------------------------------------------------------------- check ---
Say "checking the install"
& $conda run -n $envName --no-capture-output --cwd $repo python (Join-Path $repo 'tools\check_install.py')
if ($LASTEXITCODE -ne 0) { Fail "The environment was built but the check failed. See the messages above." }

# -------------------------------------------------------------- shortcut ---
if (-not $NoShortcut) {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $lnk = Join-Path $desktop 'PebbleMapper.lnk'
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($lnk)
    $sc.TargetPath = Join-Path $repo 'gui\launch_gui.bat'
    $sc.WorkingDirectory = $repo
    $sc.IconLocation = (Join-Path $repo 'gui\icon.ico') + ',0'
    $sc.Description = 'Start PebbleMapper'
    $sc.Save()
    Ok "desktop shortcut: $lnk"
}

Write-Host ""
Say "PebbleMapper is installed."
if ($weightsPending) {
    Write-Host "    Detect needs the model weights: see the message above on how to obtain them." -ForegroundColor Yellow
}
if (-not $NoShortcut) {
    Write-Host "    Start it with the PebbleMapper icon on the desktop."
}
else {
    Write-Host "    Start it by double-clicking gui\launch_gui.bat."
}
Write-Host ""
