#!/usr/bin/env pwsh
<#
.SYNOPSIS
    RTX Hands-Free Assistant Launcher
    
.DESCRIPTION
    Activates conda environment va-clean and launches the voice assistant.
    Non-interactive; all settings auto-detect or use defaults.
    
.EXAMPLE
    .\scripts\launch.ps1                                # Auto-detect GPU, fast profile
    .\scripts\launch.ps1 --device cpu                   # Force CPU mode
    .\scripts\launch.ps1 --profile advanced             # Use advanced LLM/Whisper (needs > 12GB VRAM)
    .\scripts\launch.ps1 --keywords "okay luna,hey luna" # Custom wake words
#>

param(
    [string]$device = "",
    [string]$profile = "",
    [string]$keywords = "",
    [string]$provider = ""
)

$ErrorActionPreference = "Stop"

# Miniforge path
$miniformgePath = "C:\ProgramData\miniforge"
if (-not (Test-Path $miniformgePath)) {
    Write-Host "❌ Miniforge not found at $miniformgePath" -ForegroundColor Red
    Write-Host "Install Miniforge or update the path in this script."
    exit 1
}

# Activate conda
$condaInit = Join-Path $miniformgePath "shell\condabin\conda-hook.ps1"
if (Test-Path $condaInit) {
    & $condaInit
} else {
    Write-Host "❌ Conda initialization script not found" -ForegroundColor Red
    exit 1
}

# Activate va-clean environment
Write-Host "🔄 Activating conda environment 'va-clean'..." -ForegroundColor Cyan
conda activate va-clean
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate conda environment" -ForegroundColor Red
    exit 1
}

# Build CLI args
$pythonArgs = @("voice-assistant\main.py")
if ($device) { $pythonArgs += @("--device", $device) }
if ($profile) { $pythonArgs += @("--profile", $profile) }
if ($keywords) { $pythonArgs += @("--keywords", $keywords) }
if ($provider) { $pythonArgs += @("--provider", $provider) }

Write-Host "🚀 Launching RTX Hands-Free Assistant..." -ForegroundColor Green
Write-Host ""

# Launch assistant
python @pythonArgs
