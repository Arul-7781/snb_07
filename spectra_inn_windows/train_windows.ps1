[CmdletBinding()]
param(
    [string]$AudioRoot = "",
    [string]$DataDirectory = "",
    [string]$ManifestPath = "data\manifests\all.jsonl",
    [string]$ConfigPath = "configs\one_day_librispeech.yaml",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [ValidateRange(1, 1000)]
    [int]$SmokeSteps = 10,
    [ValidateRange(0, 100000000)]
    [int]$MaxSteps = 0,
    [ValidateRange(0.1, 168.0)]
    [double]$TrainingHours = 20.0,
    [string]$Resume = "",
    [switch]$SkipInstall,
    [switch]$SkipDownload,
    [switch]$SkipTests,
    [switch]$SkipSmoke,
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [string[]]$Arguments = @()
    )

    Write-Host "`n> $Executable $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

function Require-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. $InstallHint"
    }
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedMd5
    )

    if (Test-Path $Destination -PathType Leaf) {
        $ExistingMd5 = (Get-FileHash -Path $Destination -Algorithm MD5).Hash.ToLowerInvariant()
        if ($ExistingMd5 -eq $ExpectedMd5.ToLowerInvariant()) {
            Write-Host "Verified existing download: $Destination" -ForegroundColor Green
            return
        }
        Write-Host "Removing incomplete or invalid archive: $Destination" -ForegroundColor Yellow
        Remove-Item -Force $Destination
    }

    Write-Host "`nDownloading $Url" -ForegroundColor Cyan
    if (Get-Command "Start-BitsTransfer" -ErrorAction SilentlyContinue) {
        Start-BitsTransfer -Source $Url -Destination $Destination -DisplayName "SPECTRA-INN dataset"
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    }

    $ActualMd5 = (Get-FileHash -Path $Destination -Algorithm MD5).Hash.ToLowerInvariant()
    if ($ActualMd5 -ne $ExpectedMd5.ToLowerInvariant()) {
        Remove-Item -Force $Destination
        throw "Checksum mismatch for $Url. Expected $ExpectedMd5, received $ActualMd5."
    }
    Write-Host "Checksum verified: $Destination" -ForegroundColor Green
}

function Expand-TarGzIfMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ArchivePath,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedRoot,
        [Parameter(Mandatory = $true)]
        [string]$DestinationRoot
    )

    if (Test-Path $ExpectedRoot -PathType Container) {
        Write-Host "Using extracted dataset: $ExpectedRoot" -ForegroundColor Green
        return
    }
    if (-not (Test-Path $ArchivePath -PathType Leaf)) {
        throw "Dataset archive is missing: $ArchivePath. Run without -SkipDownload."
    }
    Invoke-Checked -Executable "tar" -Arguments @(
        "-xzf", $ArchivePath, "-C", $DestinationRoot
    )
    if (-not (Test-Path $ExpectedRoot -PathType Container)) {
        throw "Extraction completed but the expected directory was not found: $ExpectedRoot"
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "SPECTRA-INN Windows bootstrap" -ForegroundColor Green
Write-Host "Project: $ProjectRoot"

Require-Command -Name "py" -InstallHint "Install 64-bit Python 3.11 from https://www.python.org/downloads/windows/ and enable the Python launcher."
Require-Command -Name "nvidia-smi" -InstallHint "Install or repair the NVIDIA Windows driver, then reboot."
Require-Command -Name "ffmpeg" -InstallHint "Install FFmpeg and add its bin directory to PATH."
Require-Command -Name "ffprobe" -InstallHint "Install FFmpeg and add its bin directory to PATH."
Require-Command -Name "tar" -InstallHint "Use a current Windows 10/11 installation or install bsdtar."

Invoke-Checked -Executable "nvidia-smi" -Arguments @()
Invoke-Checked -Executable "py" -Arguments @("-3.11", "-c", "import sys; print(sys.version); assert sys.maxsize > 2**32")

$VenvRoot = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"

if (-not (Test-Path $PythonExe -PathType Leaf)) {
    if (Test-Path $VenvRoot) {
        throw "A non-Windows or incomplete .venv exists at $VenvRoot. Rename or remove it, then run this script again."
    }
    Invoke-Checked -Executable "py" -Arguments @("-3.11", "-m", "venv", $VenvRoot)
}

if (-not $SkipInstall) {
    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
    )
    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "pip", "install", "--upgrade", "--force-reinstall",
        "torch", "torchaudio", "--index-url", $TorchIndexUrl
    )
    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "pip", "install", "-e", ".[dev,metrics]"
    )
}

$CudaProbe = @"
import torch
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable to PyTorch'
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name)
print('VRAM GiB:', round(p.total_memory / 1024**3, 2))
"@
Invoke-Checked -Executable $PythonExe -Arguments @("-c", $CudaProbe)
Invoke-Checked -Executable $PythonExe -Arguments @("scripts\check_env.py")

if ($SetupOnly) {
    Write-Host "`nSetup and GPU verification completed." -ForegroundColor Green
    exit 0
}

if (-not $SkipTests) {
    Invoke-Checked -Executable $PythonExe -Arguments @("-m", "pytest")
    Invoke-Checked -Executable $PythonExe -Arguments @("-m", "ruff", "check", ".")
}

if (-not $SkipSmoke) {
    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "spectra_inn.train",
        "--config", $ConfigPath,
        "--smoke",
        "--max-steps", $SmokeSteps.ToString(),
        "--device", "cuda"
    )
}

if ([System.IO.Path]::IsPathRooted($ManifestPath)) {
    $ResolvedManifest = $ManifestPath
} else {
    $ResolvedManifest = Join-Path $ProjectRoot $ManifestPath
}

$ManifestParent = Split-Path -Parent $ResolvedManifest
New-Item -ItemType Directory -Force -Path $ManifestParent | Out-Null

if ([string]::IsNullOrWhiteSpace($AudioRoot)) {
    if ([string]::IsNullOrWhiteSpace($DataDirectory)) {
        $ResolvedDataDirectory = Join-Path $ProjectRoot "datasets"
    } elseif ([System.IO.Path]::IsPathRooted($DataDirectory)) {
        $ResolvedDataDirectory = $DataDirectory
    } else {
        $ResolvedDataDirectory = Join-Path $ProjectRoot $DataDirectory
    }
    New-Item -ItemType Directory -Force -Path $ResolvedDataDirectory | Out-Null
    $DownloadDirectory = Join-Path $ResolvedDataDirectory "downloads"
    New-Item -ItemType Directory -Force -Path $DownloadDirectory | Out-Null

    $TrainArchive = Join-Path $DownloadDirectory "train-clean-100.tar.gz"
    $DevArchive = Join-Path $DownloadDirectory "dev-clean.tar.gz"
    $TestArchive = Join-Path $DownloadDirectory "test-clean.tar.gz"

    if (-not $SkipDownload) {
        Write-Host "`nUsing LibriSpeech under CC BY 4.0 from OpenSLR SLR12." -ForegroundColor Green
        Get-VerifiedDownload `
            -Url "https://www.openslr.org/resources/12/train-clean-100.tar.gz" `
            -Destination $TrainArchive `
            -ExpectedMd5 "2a93770f6d5c6c964bc36631d331a522"
        Get-VerifiedDownload `
            -Url "https://www.openslr.org/resources/12/dev-clean.tar.gz" `
            -Destination $DevArchive `
            -ExpectedMd5 "42e2234ba48799c1f50f24a7926300a1"
        Get-VerifiedDownload `
            -Url "https://www.openslr.org/resources/12/test-clean.tar.gz" `
            -Destination $TestArchive `
            -ExpectedMd5 "32fa31d27d2e1cad72775fee3f4849a9"
    }

    $LibriSpeechRoot = Join-Path $ResolvedDataDirectory "LibriSpeech"
    $TrainRoot = Join-Path $LibriSpeechRoot "train-clean-100"
    $DevRoot = Join-Path $LibriSpeechRoot "dev-clean"
    $TestRoot = Join-Path $LibriSpeechRoot "test-clean"

    Expand-TarGzIfMissing `
        -ArchivePath $TrainArchive `
        -ExpectedRoot $TrainRoot `
        -DestinationRoot $ResolvedDataDirectory
    Expand-TarGzIfMissing `
        -ArchivePath $DevArchive `
        -ExpectedRoot $DevRoot `
        -DestinationRoot $ResolvedDataDirectory
    Expand-TarGzIfMissing `
        -ArchivePath $TestArchive `
        -ExpectedRoot $TestRoot `
        -DestinationRoot $ResolvedDataDirectory

    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "spectra_inn.data.manifest",
        "--root", $TrainRoot,
        "--output", $ResolvedManifest,
        "--split", "train",
        "--source", "librispeech_train_clean_100"
    )
    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "spectra_inn.data.manifest",
        "--root", $DevRoot,
        "--output", $ResolvedManifest,
        "--split", "val",
        "--append",
        "--source", "librispeech_dev_clean"
    )
    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "spectra_inn.data.manifest",
        "--root", $TestRoot,
        "--output", $ResolvedManifest,
        "--split", "test",
        "--append",
        "--source", "librispeech_test_clean"
    )
} else {
    $ResolvedAudioRoot = (Resolve-Path $AudioRoot).Path
    if (-not (Test-Path $ResolvedAudioRoot -PathType Container)) {
        throw "Audio dataset directory does not exist: $AudioRoot"
    }
    Invoke-Checked -Executable $PythonExe -Arguments @(
        "-m", "spectra_inn.data.manifest",
        "--root", $ResolvedAudioRoot,
        "--output", $ResolvedManifest,
        "--train-ratio", "0.90",
        "--val-ratio", "0.05",
        "--source", "windows_local"
    )
}

$TrainArguments = @(
    "-m", "spectra_inn.train",
    "--config", $ConfigPath,
    "--device", "cuda",
    "--max-hours", $TrainingHours.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)

if ($MaxSteps -gt 0) {
    $TrainArguments += @("--max-steps", $MaxSteps.ToString())
}
if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $ResolvedResume = (Resolve-Path $Resume).Path
    $TrainArguments += @("--resume", $ResolvedResume)
}

Write-Host "`nStarting real training. Press Ctrl+C to stop safely after the current operation." -ForegroundColor Green
Invoke-Checked -Executable $PythonExe -Arguments $TrainArguments
