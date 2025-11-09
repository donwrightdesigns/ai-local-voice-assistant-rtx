
### Wake‑word (Porcupine)

- Set env var: `$env:PICOVOICE_ACCESS_KEY = "<your_key>"`
- Optional keywords: set `PORCUPINE_KEYWORDS` (comma‑separated, e.g., `computer,jarvis`)
- Choose mode via `WAKE_MODE` (default is `porcupine`):
  - `porcupine` → wait for wake‑word then listen
  - `vad` → auto‑detect speech via VAD
- If `PICOVOICE_ACCESS_KEY` is missing or Porcupine isn't installed, the app will fall back to VAD automatically.

### First-time Setup Wizard

- Run once to select compute (GPU/CPU) and profiles:
  ```powershell
  python voice-assistant\main.py --wizard
  ```
- Choices are saved to `%APPDATA%\VoiceAssistant\settings.yaml` and used on subsequent runs.

### Hotkeys (optional mode)
- Set `mode: "hotkey"` in `voice-assistant/config.yaml` to enable.
- Defaults:
  - Ctrl+F2 → Conversation (AI speaks back)
  - Ctrl+F1 → Dictation (types what you say)
  - F15 → AI typing (LLM types a response)
  - F14 → Screen AI (captures a screenshot and uses it for context; text-only fallback)
  - Escape → Exit

# Voice Assistant – Local, GPU‑Powered, Windows 11

A minimal, fully‑offline voice assistant that uses:

- **Whisper** (Coqui) – ASR  
- **Kokoro‑TTS** – GPU‑accelerated voice synthesis  
- **OpenAI / Ollama / vLLM** – flexible LLM back‑ends  
- **Porcupine wake‑word (default)** – hands‑free voice trigger  
- **VAD‑based speech detection** – automatic fallback if Porcupine not configured  

## Install

```bash
pip install -r requirements.txt
# Optional: Porcupine wake‑word support
# You need a Picovoice Access Key (free for personal use)
# Set it via environment variable before running:
#   $env:PICOVOICE_ACCESS_KEY = "<your_key>"
