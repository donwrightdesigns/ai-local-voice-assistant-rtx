# Voice Assistant RTX Launcher
# Ensures Ollama is running, cleans stale locks, starts the app

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

# Start the app using the shared conda env (force correct Miniconda path)
$defaultConda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
$condaExe = if (Test-Path $defaultConda) { $defaultConda } else { "conda" }
# Force envs dir to Miniconda3 to avoid older paths
$env:CONDA_ENVS_PATH = Join-Path $env:USERPROFILE "miniconda3\envs"

# Allow overriding env name; default to new Kokoro env
$envName = if ($env:VOICE_ENV -and $env:VOICE_ENV.Trim().Length -gt 0) { $env:VOICE_ENV } else { "va-clean" }

# If env not found, print hint and exit cleanly
$envsList = & $condaExe env list 2>$null | Out-String
if ($envsList -notmatch "(?im)^$envName\s") {
    Write-Host "⚠️  Conda env '$envName' not found under $($env:CONDA_ENVS_PATH)." -ForegroundColor Yellow
    Write-Host "    Set VOICE_ENV to your existing env name or create 'voice-assistant'." -ForegroundColor Yellow
    Write-Host "    Example: `$env:VOICE_ENV='your-old-env'; .\\start_voice_assistant.ps1" -ForegroundColor Yellow
    exit 1
}

& $condaExe run -n $envName python src/ultimate_voice_assistant.py
