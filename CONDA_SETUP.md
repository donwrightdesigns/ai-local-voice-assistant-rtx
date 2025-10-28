# Voice Assistant RTX - Conda Environment Setup

## ✅ Setup Complete!

Your voice assistant now runs in a dedicated **Python 3.12 conda environment** with Kokoro TTS support.

## Environment Details

- **Environment Name**: `voice-assistant`
- **Location**: `C:\Users\don\miniconda3\envs\voice-assistant`
- **Python Version**: 3.12.12
- **PyTorch**: 2.5.1 with CUDA 12.4
- **Kokoro TTS**: 0.9.4 ✅

## Why Conda?

- **Shares packages** across environments (unlike venv)
- **Compatible with Pinokio's** miniconda installation
- **No redundancy** - reuses existing PyTorch/CUDA infrastructure
- **Python 3.12** - Required for Kokoro (your system has 3.13)

## Usage

### Quick Start

```powershell
.\start_voice_assistant.ps1
```

That's it! The script automatically activates the environment.

### Manual Start

```bash
# Activate environment
conda activate voice-assistant

# Run assistant
cd src
python ultimate_voice_assistant.py
```

### Test Kokoro

```bash
conda run -n voice-assistant python src/kokoro_tts.py
```

## Hotkeys

- **Ctrl+F2** - Conversation Mode (AI with Kokoro voice)
- **Ctrl+F1** - Dictation Mode (types what you say)
- **F15** - AI Typing Mode
- **F14** - Screen AI Mode
- **Menu** - Reset conversation
- **Escape** - Exit

## Configuration

Edit `config/config.yaml` to change voices:

```yaml
tts:
  engine: "auto"       # Uses Kokoro automatically
  voice: "af_heart"    # Try: af_bella, am_adam, bf_emma, etc.
  lang_code: "a"       # American English
```

## Available Voices

**Female American**: af_heart, af_bella, af_nicole, af_sarah, af_sky  
**Male American**: am_adam, am_michael  
**Female British**: bf_emma, bf_isabella  
**Male British**: bm_george, bm_lewis  

...and 43 more! See: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

## Installed Packages

Core dependencies installed in the environment:
- PyTorch 2.5.1 (CUDA 12.4)
- Kokoro 0.9.4
- faster-whisper 1.2.0
- langchain 1.0.2 + community
- transformers 4.57.1
- sounddevice, soundfile
- pynput, pyautogui, Pillow
- rich, httpx, pyyaml

## Managing the Environment

### Update packages
```bash
conda activate voice-assistant
pip install --upgrade kokoro
```

### List installed packages
```bash
conda run -n voice-assistant pip list
```

### Remove environment (if needed)
```bash
conda env remove -n voice-assistant
```

### Recreate from scratch
```bash
conda create -n voice-assistant python=3.12 -y
conda activate voice-assistant
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Shared Resources

The environment shares:
- **HuggingFace cache** - Downloaded models (if HF_HOME is set)
- **Conda packages** - PyTorch, CUDA libraries
- **System resources** - eSpeak-NG, Ollama, etc.

## Troubleshooting

### "conda: command not found"
Make sure conda is in your PATH or use the full path:
```powershell
C:\Users\don\miniconda3\Scripts\conda.exe activate voice-assistant
```

### Kokoro not working
```bash
conda run -n voice-assistant pip install --upgrade kokoro soundfile
```

### Want to use system Python instead?
This won't work - Kokoro requires Python 3.12, your system has 3.13.

## Performance

- **Model size**: ~330MB (downloads once)
- **VRAM usage**: ~500MB during synthesis
- **Latency**: 1-2 seconds for responses
- **Quality**: High - comparable to commercial TTS

---

**You've successfully avoided isolated environments while maintaining compatibility!** 🎉

The conda environment integrates seamlessly with your existing Pinokio/miniconda setup.
