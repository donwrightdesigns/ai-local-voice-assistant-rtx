# [Deprecated] Voice Assistant RTX TTS Installation Guide (Kokoro engine)

Note: This doc is superseded by SETUP_COMPLETE.md. Keeping for reference.

## 🎉 Kokoro-82M Integration Complete!

Your voice assistant now supports Kokoro-82M, a high-quality, lightweight TTS model with Apache license.

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install kokoro>=0.9.2 soundfile
```

### 2. Install eSpeak-NG (Required for Kokoro)

Kokoro uses eSpeak-NG for phonemization. Choose one:

**Option A: Using Chocolatey (Recommended)**
```powershell
choco install espeak-ng
```

**Option B: Manual Installation**
1. Download from: https://github.com/espeak-ng/espeak-ng/releases
2. Install the Windows `.msi` installer
3. Add to PATH: `C:\Program Files\eSpeak NG\`

### 3. Verify Installation

Test Kokoro directly:
```bash
cd voice-assistant-windows\src
python kokoro_tts.py
```

You should hear: "Hello! This is Kokoro, an open-weight text to speech model..."

### 4. Update Your Config (Already Done!)

The `config/config.yaml` is already configured to use Kokoro:

```yaml
tts:
  engine: "auto"  # Will automatically use Kokoro if available
  voice: "af_heart"  # Female American voice
  lang_code: "a"  # American English
```

## Available Voices

Kokoro comes with 54 voices! Here are some popular ones:

### Female American (af_*)
- `af_heart` - Warm, friendly (default)
- `af_bella` - Elegant
- `af_nicole` - Professional
- `af_sarah` - Clear
- `af_sky` - Bright

### Male American (am_*)
- `am_adam` - Deep, authoritative
- `am_michael` - Friendly

### Female British (bf_*)
- `bf_emma` - Classic British
- `bf_isabella` - Refined

### Male British (bm_*)
- `bm_george` - Distinguished
- `bm_lewis` - Modern British

**Full list**: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

## Usage

Just run your voice assistant as normal:

```bash
cd voice-assistant-windows\src
python ultimate_voice_assistant.py
```

Kokoro will automatically be used for conversation mode (Ctrl+F2).

## Troubleshooting

### "Kokoro not installed" error
```bash
pip install kokoro>=0.9.2 soundfile
```

### "espeak-ng not found" error
Install eSpeak-NG (see step 2 above)

### Want to switch back to Windows TTS?
Edit `config/config.yaml`:
```yaml
tts:
  engine: "windows"  # Force Windows SAPI
```

### Want to try different voices?
Edit `config/config.yaml`:
```yaml
tts:
  voice: "am_adam"  # Try male voice
```

## Performance Notes

- **First run**: Kokoro downloads the model (~330MB) - this only happens once
- **Latency**: ~1-2 seconds for short phrases (much faster than Bark, slower than Windows TTS)
- **Quality**: Significantly better than Windows voices, comparable to commercial TTS
- **VRAM usage**: ~500MB (runs alongside your LLM without issues)

## What Changed?

1. ✅ New `kokoro_tts.py` - Kokoro wrapper class
2. ✅ Updated `tts_service.py` - Added Kokoro as engine option
3. ✅ Updated `requirements.txt` - Added dependencies
4. ✅ Updated `config.yaml` - Configured for Kokoro
5. ✅ Updated `ultimate_voice_assistant.py` - Passes lang_code parameter

Kokoro is now the default when `engine: "auto"` is set! 🎊
