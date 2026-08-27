<#
.SYNOPSIS
One-command Windows evaluation runner for LF-VSN_V2.

.DESCRIPTION
Optionally prepares the Python environment and Vimeo-90K septuplet,
auto-selects a checkpoint (latest_G.pth or highest iter), and runs test.py.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_windows.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_windows.ps1 -Config options/train/train_LF-VSN_v2_4090_2day_scalable7.yml

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_windows.ps1 -Ckpt ..\experiments\train_LF-VSN_v2_4090_2day_1video\models\latest_G.pth
#>

param(
    [string]$Config = "options/train/train_LF-VSN_v2_4090_2day_1video.yml",
    [string]$Ckpt = "",
    [string]$PythonExe = "",
    [string]$PythonVersion = "3.10",
    [string]$VenvPath = ".venv",
    [ValidateSet("cu121", "cu124")]
    [string]$CudaWheel = "cu121",
    [switch]$SetupEnv,
    [switch]$SetupData,
    [string]$VimeoRoot = "",
    [int]$NumValPairs = 200
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $CodeDir "..")).Path

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if ([System.IO.Path]::IsPathRooted($VenvPath)) {
        $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
    } else {
        $PythonExe = Join-Path (Join-Path $CodeDir $VenvPath) "Scripts\python.exe"
    }
}

$NeedEnvSetup = $SetupEnv.IsPresent -or -not (Test-Path $PythonExe)
if ($NeedEnvSetup) {
    Write-Host "Preparing evaluation environment..."
    & (Join-Path $ScriptDir "setup_windows_4090_env.ps1") -PythonVersion $PythonVersion -VenvPath $VenvPath -CudaWheel $CudaWheel
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if ([System.IO.Path]::IsPathRooted($Config)) {
    $ConfigPath = $Config
} else {
    $ConfigPath = Join-Path $CodeDir $Config
}
$ConfigPath = (Resolve-Path $ConfigPath).Path

if ([string]::IsNullOrWhiteSpace($VimeoRoot)) {
    $VimeoRoot = Join-Path $CodeDir "vimeo90k"
}

$NeedDataSetup = $SetupData.IsPresent
$ExpectedData = @(
    (Join-Path $VimeoRoot "sequences"),
    (Join-Path $VimeoRoot "sep_trainlist.txt"),
    (Join-Path $VimeoRoot "sep_testlist.txt"),
    (Join-Path $VimeoRoot "sep_testlist_7video.txt")
)
foreach ($Path in $ExpectedData) {
    if (-not (Test-Path $Path)) {
        $NeedDataSetup = $true
    }
}

if ($NeedDataSetup) {
    Write-Host "Preparing Vimeo-90K septuplet dataset..."
    & (Join-Path $ScriptDir "setup_vimeo90k_septuplet_windows.ps1") -VimeoRoot $VimeoRoot -NumValPairs $NumValPairs -PythonExe $PythonExe
}

$ConfigName = (& $PythonExe -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8')); print(c['name'])" $ConfigPath).Trim()
if ([string]::IsNullOrWhiteSpace($ConfigName)) {
    throw "Could not read model name from config: $ConfigPath"
}

if ([string]::IsNullOrWhiteSpace($Ckpt)) {
    $ModelDir = Join-Path $RepoRoot ("experiments\{0}\models" -f $ConfigName)
    if (-not (Test-Path $ModelDir)) {
        throw "Model directory not found: $ModelDir. Train first or pass -Ckpt explicitly."
    }

    $LatestPath = Join-Path $ModelDir "latest_G.pth"
    if (Test-Path $LatestPath) {
        $CkptPath = $LatestPath
    } else {
        $Candidates = @()
        $Files = Get-ChildItem -Path $ModelDir -Filter "*_G.pth" -File
        foreach ($File in $Files) {
            if ($File.BaseName -match "^(\d+)_G$") {
                $Candidates += [PSCustomObject]@{ Iter = [int]$Matches[1]; Path = $File.FullName }
            }
        }
        if ($Candidates.Count -eq 0) {
            throw "No checkpoint found in $ModelDir. Train first or pass -Ckpt explicitly."
        }
        $CkptPath = ($Candidates | Sort-Object Iter -Descending | Select-Object -First 1).Path
    }
} else {
    if ([System.IO.Path]::IsPathRooted($Ckpt)) {
        $CkptPath = $Ckpt
    } else {
        $CkptPath = Join-Path $CodeDir $Ckpt
    }
}

if (-not (Test-Path $CkptPath)) {
    throw "Checkpoint file not found: $CkptPath"
}

$RuntimeDir = Join-Path $CodeDir "options\train\runtime_generated"
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
$RuntimeConfig = Join-Path $RuntimeDir ("{0}_test_{1}.yml" -f $BaseName, $Stamp)
$SequencesDir = Join-Path $VimeoRoot "sequences"
$ListsDir = $VimeoRoot

$PatchCode = @'
import sys
import os
import yaml

src, data_root, lists_dir, dst = sys.argv[1:5]
with open(src, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

datasets = cfg.setdefault('datasets', {})
for phase_name, phase_cfg in datasets.items():
    phase_cfg['data_path'] = data_root
    txt_name = os.path.basename(str(phase_cfg.get('txt_path', '')))
    if txt_name:
        phase_cfg['txt_path'] = os.path.join(lists_dir, txt_name)

with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
'@

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$RuntimeConfigOutput = & $PythonExe -c $PatchCode $ConfigPath $SequencesDir $ListsDir $RuntimeConfig 2>&1
$RuntimeConfigExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($RuntimeConfigExitCode -ne 0) {
    $Details = ($RuntimeConfigOutput -join [Environment]::NewLine).Trim()
    throw "Failed to generate test configuration with $PythonExe.`n$Details"
}
if (-not (Test-Path $RuntimeConfig)) {
    throw "Test configuration was not created: $RuntimeConfig"
}

Write-Host ""
Write-Host "Config: $RuntimeConfig"
Write-Host "Experiment: $ConfigName"
Write-Host "Checkpoint: $CkptPath"
Write-Host ""

Set-Location $CodeDir
& $PythonExe test.py -opt $RuntimeConfig --ckpt $CkptPath
