# Configuration – tweak these values to suit your environment
from pathlib import Path
from typing import Literal

# ------------------
# Core paths
# ------------------
BASE_DIR = Path(__file__).parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

# ------------------
# ASR
# ------------------
WHISPER_MODEL = "large-v3"          # Whisper model name (available from Coqui)
# ------------------
# TTS
# ------------------
KOKORO_MODEL = "kokoro-en"          # Pre‑trained Kokoro voice
# ------------------
# LLM
# ------------------
LLM_PROVIDER: Literal["openai", "ollama", "vllm"] = "openai"
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "llama3:instruct"
VLLM_ENDPOINT = "http://localhost:8000/v1"   # vLLM server

# ------------------
# Hot‑key / wake‑word
# ------------------
HOTKEY = "ctrl+shift+space"         # for backward compatibility (optional)
# ------------------
# VAD / energy threshold
# ------------------
VAD_MODE = 3
ENERGY_THRESHOLD = 2500            # adjust to your room
