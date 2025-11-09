#!/usr/bin/env powershell

# Ensure Ollama is running, cleans stale locks, starts the app
Write-Host "🎤 Starting Voice Assistant RTX..." -ForegroundColor Cyan
Write-Host ""

# Prefer Ollama Windows service if present
$ollamaService = Get-Service -Name "Ollama" -ErrorAction SilentlyContinue
if ($ollamaService) {
    if ($ollamaService.Status -ne 'Running') {
        Write-Host "🚀 Starting Ollama Windows service..." -ForegroundColor Yellow
        try { Start-Service -Name "Ollama"; Start-Sleep -Seconds 3; Write-Host "✅ Ollama service started" -ForegroundColor Green } catch { Write-Host "⚠️  Failed to start Ollama service" -ForegroundColor Yellow }
    } else {
        Write-Host "✅ Ollama service is running" -ForegroundColor Green
    }
} else {
    # Fallback to process check
    $ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaRunning) {
        Write-Host "🚀 Launching Ollama (no service detected)..." -ForegroundColor Yellow
        try { Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep -Seconds 3; Write-Host "✅ Ollama started" -ForegroundColor Green } catch { Write-Host "⚠️  Could not auto-start Ollama. Start manually: ollama serve" -ForegroundColor Yellow }
        } else {
            Write-Host "✅ Ollama is already running" -ForegroundColor Green
            }
    }

# Clean stale lock if PID not running
$lockPath = Join-Path $PSScriptRoot "voice_assistant.lock"
if (Test-Path $lockPath) {
    try {
        $pidText = Get-Content $lockPath -ErrorAction Stop | Select-Object -First 1
        $pidVal = [int]$pidText
        $proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
        if (-not $proc) { Remove-Item $lockPath -Force -ErrorAction SilentlyContinue; Write-Host "🧹 Removed stale lock" -ForegroundColor Yellow }
    } catch { Remove-Item $lockPath -Force -ErrorAction SilentlyContinue }
}

Write-Host ""

# Fix OpenMP duplicate library issue (common with conda)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Locate assistant entrypoint
$scriptPath = Join-Path $PSScriptRoot "voice-assistant\main.py"
if (-not (Test-Path $scriptPath)) {
    Write-Host "❌ Could not find main.py at $scriptPath" -ForegroundColor Red
    exit 1
}

# First-time setup wizard (saves user preferences under %APPDATA%\VoiceAssistant)
$settingsPath = Join-Path $env:APPDATA "VoiceAssistant\settings.yaml"
if (-not (Test-Path $settingsPath)) {
    Write-Host "🧪 Running first-time setup wizard..." -ForegroundColor Cyan
    try {
        & python $scriptPath --wizard
    } catch {
        Write-Host "⚠️  Wizard could not run; continuing with defaults." -ForegroundColor Yellow
    }
}

Write-Host "🚀 Launching assistant..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "`"$scriptPath`"" -WorkingDirectory $PSScriptRoot
