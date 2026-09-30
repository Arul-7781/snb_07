<#
.SYNOPSIS
Set up a Windows training environment for LF-VSN_V2 on RTX 4090.

.DESCRIPTION
Creates a Python virtual environment, installs CUDA-enabled PyTorch,
installs training dependencies, and prints a quick CUDA sanity check.

.PARAMETER PythonVersion
Python version passed to the Windows py launcher (default: 3.10).

.PARAMETER VenvPath
Virtual environment path (relative to code dir or absolute).

.PARAMETER CudaWheel
PyTorch CUDA wheel channel. Use cu121 (recommended) or cu124.

.PARAMETER InstallOptional
Also installs optional visualization/demo dependencies.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_4090_env.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_4090_env.ps1 -CudaWheel cu124 -InstallOptional
#>

param(
    [string]$PythonVersion = "3.10",
    [string]$VenvPath = ".venv",
    [ValidateSet("cu121", "cu124")]
    [string]$CudaWheel = "cu121",
    [switch]$InstallOptional
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path

if ([System.IO.Path]::IsPathRooted($VenvPath)) {
    $VenvFullPath = $VenvPath
} else {
    $VenvFullPath = Join-Path $CodeDir $VenvPath
}

$ReqDir = Join-Path $CodeDir "requirements"
$TrainReq = Join-Path $ReqDir "windows-train.txt"
$OptionalReq = Join-Path $ReqDir "windows-optional.txt"

if (-not (Test-Path $TrainReq)) {
    throw "Missing dependency file: $TrainReq"
}

Write-Host "[1/5] Creating virtual environment at: $VenvFullPath"
$usedPyLauncher = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    $usedPyLauncher = $true
    & py "-$PythonVersion" -m venv "$VenvFullPath"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "py launcher not found, using default python on PATH"
    & python -m venv "$VenvFullPath"
} else {
    throw "Neither 'py' nor 'python' command is available."
}

$PyExe = Join-Path $VenvFullPath "Scripts\python.exe"
if (-not (Test-Path $PyExe) -and $usedPyLauncher -and (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Requested py launcher version not available, retrying with default python"
    & python -m venv "$VenvFullPath"
}
if (-not (Test-Path $PyExe)) {
    throw "Virtual environment Python not found: $PyExe"
}

Write-Host "[2/5] Upgrading pip tooling"
& $PyExe -m pip install --upgrade pip setuptools wheel

if ($CudaWheel -eq "cu121") {
    $TorchVersion = "2.4.1"
    $TorchVisionVersion = "0.19.1"
    $IndexUrl = "https://download.pytorch.org/whl/cu121"
} else {
    $TorchVersion = "2.5.1"
    $TorchVisionVersion = "0.20.1"
    $IndexUrl = "https://download.pytorch.org/whl/cu124"
}

Write-Host "[3/5] Installing PyTorch ($CudaWheel)"
& $PyExe -m pip install --index-url $IndexUrl "torch==$TorchVersion" "torchvision==$TorchVisionVersion"

Write-Host "[4/5] Installing LF-VSN training dependencies"
& $PyExe -m pip install -r $TrainReq

if ($InstallOptional.IsPresent) {
    if (-not (Test-Path $OptionalReq)) {
        throw "Missing optional dependency file: $OptionalReq"
    }
    Write-Host "Installing optional plotting/demo dependencies"
    & $PyExe -m pip install -r $OptionalReq
}

Write-Host "[5/5] CUDA sanity check"
& $PyExe -c "import torch;print('torch',torch.__version__);print('torchvision',__import__('torchvision').__version__);print('cuda',torch.version.cuda);print('cuda_available',torch.cuda.is_available());print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host ""
Write-Host "Environment setup complete."
Write-Host "Activate with:"
Write-Host "  $VenvFullPath\Scripts\Activate.ps1"
