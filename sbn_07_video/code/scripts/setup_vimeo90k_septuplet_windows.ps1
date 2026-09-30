<#
.SYNOPSIS
Download Vimeo-90K septuplet and build LF-VSN list files on Windows.

.DESCRIPTION
Downloads vimeo_septuplet.zip, extracts sequences, and creates
sep_trainlist.txt, sep_testlist.txt, and sep_testlist_7video.txt.

.PARAMETER VimeoRoot
Dataset root directory. Defaults to code\vimeo90k.

.PARAMETER NumValPairs
Number of cover sequences in the generated test list (default: 200).

.PARAMETER PythonExe
Python executable used to run build_vimeo90k_lists.py (default: python).

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\setup_vimeo90k_septuplet_windows.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\setup_vimeo90k_septuplet_windows.ps1 -VimeoRoot D:\datasets\vimeo90k -NumValPairs 200 -PythonExe .\.venv\Scripts\python.exe
#>

param(
    [string]$VimeoRoot = "",
    [int]$NumValPairs = 200,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path

if ([string]::IsNullOrWhiteSpace($VimeoRoot)) {
    $VimeoRoot = Join-Path $CodeDir "vimeo90k"
}

$VimeoRoot = (New-Item -ItemType Directory -Path $VimeoRoot -Force).FullName
$DownloadsDir = Join-Path $VimeoRoot "downloads"
$ZipPath = Join-Path $DownloadsDir "vimeo_septuplet.zip"
$ExtractRoot = Join-Path $DownloadsDir "extract"
$DataUrl = "https://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"

New-Item -ItemType Directory -Path $DownloadsDir -Force | Out-Null

if (-not (Test-Path $ZipPath)) {
    Write-Host "Downloading Vimeo-90K septuplet..."
    Invoke-WebRequest -Uri $DataUrl -OutFile $ZipPath
} else {
    Write-Host "Using existing zip: $ZipPath"
}

$SequencesDir = Join-Path $VimeoRoot "sequences"
if (-not (Test-Path $SequencesDir)) {
    Write-Host "Extracting dataset..."
    New-Item -ItemType Directory -Path $ExtractRoot -Force | Out-Null
    $SevenZip = $null
    $SevenZipCommand = Get-Command 7z, 7zz -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($SevenZipCommand) {
        $SevenZip = $SevenZipCommand.Source
    } else {
        $SevenZipCandidates = @(
            (Join-Path $env:ProgramFiles '7-Zip\7z.exe'),
            (Join-Path ${env:ProgramFiles(x86)} '7-Zip\7z.exe')
        )
        $SevenZip = $SevenZipCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    }

    if (-not $SevenZip) {
        throw @"
7-Zip is required to extract Vimeo-90K on Windows because this archive is not supported by PowerShell Expand-Archive.
Install 7-Zip, reopen PowerShell, and run this script again:
  winget install --id 7zip.7zip -e
"@
    }

    & $SevenZip x $ZipPath "-o$ExtractRoot" -y
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip failed to extract $ZipPath (exit code $LASTEXITCODE)."
    }

    $CandidateA = Join-Path $ExtractRoot "vimeo_septuplet\sequences"
    $CandidateB = Join-Path $ExtractRoot "sequences"

    if (Test-Path $CandidateA) {
        Move-Item -Path $CandidateA -Destination $SequencesDir -Force
    } elseif (Test-Path $CandidateB) {
        Move-Item -Path $CandidateB -Destination $SequencesDir -Force
    } else {
        throw "Could not locate extracted sequences folder."
    }
} else {
    Write-Host "Using existing sequences: $SequencesDir"
}

Write-Host "Building LF-VSN list files..."
& $PythonExe (Join-Path $ScriptDir "build_vimeo90k_lists.py") --output-dir $VimeoRoot --num-test-pairs $NumValPairs --test-secrets 1 7 --data-root $SequencesDir --min-frames 7

Write-Host ""
Write-Host "Vimeo-90K septuplet setup complete."
Write-Host "Data root: $SequencesDir"
Write-Host "Train list: $(Join-Path $VimeoRoot 'sep_trainlist.txt')"
Write-Host "Test list (1 secret): $(Join-Path $VimeoRoot 'sep_testlist.txt')"
Write-Host "Test list (7 secret): $(Join-Path $VimeoRoot 'sep_testlist_7video.txt')"
