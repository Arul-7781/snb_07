<#
.SYNOPSIS
One-command Windows training runner for LF-VSN_V2.

.DESCRIPTION
Optionally prepares the Python environment and Vimeo-90K septuplet,
auto-resumes from latest training state, optionally caps total iters
for a 2-day budget, and launches train.py.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_windows.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_windows.ps1 -Strict2Day

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_windows.ps1 -Config options/train/train_LF-VSN_v2_4090_2day_scalable7.yml
#>

param(
    [string]$Config = "options/train/train_LF-VSN_v2_4090_2day_1video.yml",
    [string]$PythonExe = "",
    [string]$PythonVersion = "3.10",
    [string]$VenvPath = ".venv",
    [ValidateSet("cu121", "cu124")]
    [string]$CudaWheel = "cu121",
    [switch]$InstallOptional,
    [string]$VimeoRoot = "",
    [int]$NumValPairs = 200,
    [switch]$SetupEnv,
    [switch]$SetupData,
    [switch]$NoResume,
    [switch]$Strict2Day,
    [int]$NiterOverride = 0,
    [int]$TimeBudgetHours = 48,
    [double]$ConservativeItersPerSecond = 0.95
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
    Write-Host "Preparing training environment..."
    $EnvArgs = @{
        PythonVersion = $PythonVersion
        VenvPath = $VenvPath
        CudaWheel = $CudaWheel
    }
    if ($InstallOptional.IsPresent) {
        $EnvArgs["InstallOptional"] = $true
    }
    & (Join-Path $ScriptDir "setup_windows_4090_env.ps1") @EnvArgs
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$PythonExe = (Resolve-Path $PythonExe).Path

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

$ResumeStatePath = ""
$ResumeIter = 0
if (-not $NoResume.IsPresent) {
    $StateDir = Join-Path $RepoRoot ("experiments\{0}\training_state" -f $ConfigName)
    if (Test-Path $StateDir) {
        $StateFiles = Get-ChildItem -Path $StateDir -Filter "*.state" -File
        $Candidates = @()
        foreach ($File in $StateFiles) {
            $Iter = 0
            if ([int]::TryParse($File.BaseName, [ref]$Iter)) {
                $Candidates += [PSCustomObject]@{ Iter = $Iter; Path = $File.FullName }
            }
        }
        if ($Candidates.Count -gt 0) {
            $ResumeStatePath = ($Candidates | Sort-Object Iter -Descending | Select-Object -First 1).Path
            [void][int]::TryParse((Split-Path $ResumeStatePath -LeafBase), [ref]$ResumeIter)
        }
    }
}

$EffectiveNiterOverride = $NiterOverride
if ($Strict2Day.IsPresent -and $EffectiveNiterOverride -le 0) {
    $RawIters = [math]::Floor($TimeBudgetHours * 3600.0 * $ConservativeItersPerSecond)
    $RoundedIters = [math]::Floor($RawIters / 1000.0) * 1000
    if ($RoundedIters -lt 1000) {
        $RoundedIters = 1000
    }
    $EffectiveNiterOverride = [int]$RoundedIters
}

if ($EffectiveNiterOverride -gt 0 -and $ResumeIter -ge $EffectiveNiterOverride) {
    $EffectiveNiterOverride = $ResumeIter + 5000
    Write-Warning "Requested niter is not greater than resume iter. Bumping niter to $EffectiveNiterOverride"
}

$RuntimeDir = Join-Path $CodeDir "options\train\runtime_generated"
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
$RuntimeConfig = Join-Path $RuntimeDir ("{0}_run_{1}.yml" -f $BaseName, $Stamp)
$SequencesDir = Join-Path $VimeoRoot "sequences"
$TrainListPath = Join-Path $VimeoRoot "sep_trainlist.txt"
$ResumeStateArgument = if ([string]::IsNullOrWhiteSpace($ResumeStatePath)) { '__NO_RESUME__' } else { $ResumeStatePath }

$PatchCode = @'
import sys
import os
import yaml

src, resume_state, niter_override, data_root, train_list, dst = sys.argv[1:7]
with open(src, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('path', {})
cfg['path']['resume_state'] = None if resume_state == '__NO_RESUME__' else resume_state

datasets = cfg.setdefault('datasets', {})
for phase_name, phase_cfg in datasets.items():
    phase_cfg['data_path'] = data_root
    txt_name = os.path.basename(str(phase_cfg.get('txt_path', '')))
    if phase_name == 'train':
        phase_cfg['txt_path'] = train_list
    elif txt_name:
        phase_cfg['txt_path'] = os.path.join(os.path.dirname(train_list), txt_name)

niter = int(niter_override)
if niter > 0:
    cfg.setdefault('train', {})
    cfg['train']['niter'] = niter

with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
'@

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$RuntimeConfigOutput = & $PythonExe -c $PatchCode $ConfigPath $ResumeStateArgument $EffectiveNiterOverride $SequencesDir $TrainListPath $RuntimeConfig 2>&1
$RuntimeConfigExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($RuntimeConfigExitCode -ne 0) {
    $Details = ($RuntimeConfigOutput -join [Environment]::NewLine).Trim()
    throw "Failed to generate runtime configuration with $PythonExe.`n$Details"
}
if (-not (Test-Path $RuntimeConfig)) {
    throw "Runtime configuration was not created: $RuntimeConfig"
}

$FinalNiterOutput = & $PythonExe -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8')); print(c['train']['niter'])" $RuntimeConfig
if ($LASTEXITCODE -ne 0 -or $null -eq $FinalNiterOutput) {
    throw "Could not read train.niter from runtime configuration: $RuntimeConfig"
}
$FinalNiter = ([string]$FinalNiterOutput).Trim()

$PaperItersPerSecond = 250000.0 / (72.0 * 3600.0)
$PaperEtaHours = [math]::Round(([double]$FinalNiter / ($PaperItersPerSecond * 3600.0)), 2)
$ConservativeEtaHours = [math]::Round(([double]$FinalNiter / ($ConservativeItersPerSecond * 3600.0)), 2)

Write-Host ""
Write-Host "Run config: $RuntimeConfig"
Write-Host "Experiment name: $ConfigName"
Write-Host "Total iterations: $FinalNiter"
if ([string]::IsNullOrWhiteSpace($ResumeStatePath)) {
    Write-Host "Resume: disabled (no state found or -NoResume used)"
} else {
    Write-Host "Resume state: $ResumeStatePath"
}
Write-Host "ETA at paper speed (V100, 250K in 72h): ~$PaperEtaHours hours"
Write-Host "ETA at conservative speed ($ConservativeItersPerSecond it/s): ~$ConservativeEtaHours hours"
Write-Host ""

Set-Location $CodeDir
& $PythonExe train.py -opt $RuntimeConfig
