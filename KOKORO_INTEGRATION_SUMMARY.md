# 🎉 Kokoro-82M TTS Integration Complete!

## What is Kokoro?

Kokoro-82M is an open-source, Apache-licensed TTS model that delivers **high-quality, natural-sounding speech** with only 82 million parameters. Despite its small size, it rivals much larger models in quality while being significantly faster.

## Why Kokoro?

✅ **Perfect Balance**: High quality without massive VRAM requirements  
✅ **Fast**: Much faster than diffusion-based models like Bark  
✅ **Lightweight**: Only ~500MB VRAM (runs alongside your LLM)  
✅ **Natural Sound**: Way better than robotic Windows TTS  
✅ **54 Voices**: Multiple American & British English voices  
✅ **Apache License**: Use freely in any project  

## What Changed?

### New Files
1. **`src/kokoro_tts.py`** - Kokoro TTS wrapper class
2. **`KOKORO_INSTALL.md`** - Complete installation guide
3. **`install_kokoro.ps1`** - Automated installation script

### Modified Files
1. **`src/tts_service.py`** - Added Kokoro as a TTS engine option
2. **`src/ultimate_voice_assistant.py`** - Pass lang_code parameter
3. **`requirements.txt`** - Added kokoro and soundfile dependencies
4. **`config/config.yaml`** - Configured to use Kokoro by default

## Installation

### Quick Install

```powershell
cd voice-assistant-windows
.\install_kokoro.ps1
```

### Manual Install

```bash
# 1. Install Python packages
pip install kokoro>=0.9.2 soundfile

# 2. Install eSpeak-NG (required)
choco install espeak-ng

# 3. Test it
cd voice-assistant-windows\src
python kokoro_tts.py
```

## Configuration

Your `config/config.yaml` is already set up:

```yaml
tts:
  engine: "auto"        # Automatically uses Kokoro if available
  voice: "af_heart"     # Warm, friendly female voice
  lang_code: "a"        # American English
```

### Try Different Voices

Edit `config/config.yaml`:

```yaml
tts:
  voice: "am_adam"      # Deep male voice
  # or
  voice: "bf_emma"      # British female voice
  # or
  voice: "af_bella"     # Elegant female voice
```

## Available Voices

**Female American**: af_heart, af_bella, af_nicole, af_sarah, af_sky  
**Male American**: am_adam, am_michael  
**Female British**: bf_emma, bf_isabella  
**Male British**: bm_george, bm_lewis  

...and 43 more! See: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

## Usage

Just run your voice assistant as normal:

```bash
cd voice-assistant-windows\src
python ultimate_voice_assistant.py
```

Press **Ctrl+F2** for conversation mode with Kokoro voice responses!

## Performance

- **First Run**: Downloads model (~330MB) once
- **Latency**: 1-2 seconds (much faster than Bark)
- **Quality**: Comparable to commercial TTS
- **VRAM**: ~500MB (leaves room for your LLM)

## Fallback

If Kokoro isn't installed, the assistant automatically falls back to:
1. pyttsx3 (Windows voices)
2. Windows SAPI
3. Bark (if installed)

## Technical Details

- **Architecture**: StyleTTS 2 (no diffusion, decoder-only)
- **Sample Rate**: 24kHz (high quality)
- **License**: Apache 2.0
- **Model Size**: 82M parameters
- **Backend**: PyTorch
- **G2P**: eSpeak-NG for phonemization

## Next Steps

1. Run `.\install_kokoro.ps1` to get started
2. Test with `python src\kokoro_tts.py`
3. Try different voices by editing config.yaml
4. Enjoy your new high-quality voice assistant! 🎊

---

**Links:**
- Model: https://huggingface.co/hexgrad/Kokoro-82M
- Demo: https://hf.co/spaces/hexgrad/Kokoro-TTS
- GitHub: https://github.com/hexgrad/kokoro
