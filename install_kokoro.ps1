# Kokoro TTS Installation Script for Windows
# Run this from the voice-assistant-windows directory

Write-Host "🎤 Installing Kokoro TTS for Voice Assistant..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Install Python packages
Write-Host "📦 Installing Python packages..." -ForegroundColor Yellow
pip install kokoro>=0.9.2 soundfile

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install Python packages" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 2: Check for eSpeak-NG
Write-Host "🔍 Checking for eSpeak-NG..." -ForegroundColor Yellow

$espeakPath = Get-Command espeak-ng -ErrorAction SilentlyContinue

if ($espeakPath) {
    Write-Host "✅ eSpeak-NG is already installed at: $($espeakPath.Source)" -ForegroundColor Green
} else {
    Write-Host "⚠️  eSpeak-NG not found in PATH" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install eSpeak-NG using one of these methods:" -ForegroundColor Cyan
    Write-Host "  1. Chocolatey: choco install espeak-ng" -ForegroundColor White
    Write-Host "  2. Download: https://github.com/espeak-ng/espeak-ng/releases" -ForegroundColor White
    Write-Host ""
    
    # Ask if user wants to try Chocolatey installation
    $response = Read-Host "Do you want to try installing with Chocolatey now? (y/n)"
    
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Installing eSpeak-NG via Chocolatey..." -ForegroundColor Yellow
        choco install espeak-ng -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ eSpeak-NG installed!" -ForegroundColor Green
            Write-Host "   You may need to restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
        } else {
            Write-Host "❌ Chocolatey installation failed. Please install manually." -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "🧪 Testing Kokoro installation..." -ForegroundColor Cyan

# Test Kokoro
Set-Location src
python kokoro_tts.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 Kokoro TTS is ready to use!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To use your voice assistant with Kokoro:" -ForegroundColor Cyan
    Write-Host "  cd src" -ForegroundColor White
    Write-Host "  python ultimate_voice_assistant.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Available voices: af_heart, af_bella, am_adam, bf_emma, bm_george, and more!" -ForegroundColor Yellow
    Write-Host "Edit config/config.yaml to change voices." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "⚠️  Kokoro test failed. Check error messages above." -ForegroundColor Yellow
    Write-Host "See KOKORO_INSTALL.md for troubleshooting." -ForegroundColor Yellow
}

Set-Location ..
