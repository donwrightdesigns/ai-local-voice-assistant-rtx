import asyncio
import time
from typing import Literal, Dict, Any
from pathlib import Path
from .config_loader import (
    load_config,
    apply_computed_defaults,
    startup_wizard,
    save_user_settings,
    user_settings_path,
    is_tty,
)
from .audio_utils import AudioRecorder
from .llm_providers import (
    OpenAIProvider,
    OllamaProvider,
    VLLMProvider,
)
from .tts_service import TTSService
import coqui_whisper
import os
try:
    from .porcupine_wake import PorcupineWake
    PORCUPINE_AVAILABLE = True
except Exception:
    PORCUPINE_AVAILABLE = False

class Assistant:
    def __init__(self, provider: Literal["openai", "ollama", "vllm", "local"] = None, project_root: Path = None):
        # Load and apply configuration
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent  # Back to repo root
        
        self.config, settings_path = load_config(project_root)
        
        # First-time wizard if no user settings exist
        if not settings_path.exists() and is_tty():
            user_overrides = startup_wizard(self.config)
            save_user_settings(user_overrides)
            # Reload config with new settings
            self.config, _ = load_config(project_root)
        
        # Apply computed defaults (auto GPU detection, etc.)
        self.config = apply_computed_defaults(self.config)
        
        # Resolve active profiles to actual values
        stt_profile = self.config.get("stt", {}).get("profile", "medium")
        stt_model = self.config.get("stt", {}).get("profiles", {}).get(stt_profile, "base.en")
        stt_device = self.config.get("stt", {}).get("device", "cuda")
        
        # LLM provider resolution
        if provider is None:
            provider = self.config.get("llm", {}).get("provider", "ollama")
        
        # ------------------ ASR ------------------
        self.whisper = coqui_whisper.load(stt_model, device=stt_device)
        
        # ------------------ TTS ------------------
        tts_fallback = self.config.get("tts", {}).get("fallback", True)
        tts_device = self.config.get("tts", {}).get("device", "cuda")
        tts_model = self.config.get("tts", {}).get("model", "kokoro-en")
        self.tts_service = TTSService(fallback=tts_fallback, model_name=tts_model, device=tts_device)
        
        # ------------------ LLM provider ------------------
        if provider == "openai":
            openai_model = self.config.get("llm", {}).get("openai", {}).get("model", "gpt-4o-mini")
            self.llm = OpenAIProvider(openai_model)
        elif provider == "ollama":
            ollama_cfg = self.config.get("llm", {}).get("ollama", {})
            ollama_profile = ollama_cfg.get("active_profile", "fast")
            ollama_model = ollama_cfg.get("profiles", {}).get(ollama_profile, ollama_cfg.get("model", "llama3.2:3b"))
            base_url = ollama_cfg.get("base_url", "http://localhost:11434")
            self.llm = OllamaProvider(ollama_model, base_url=base_url)
        elif provider in ("vllm", "local"):
            vllm_endpoint = self.config.get("llm", {}).get("vllm", {}).get("endpoint", "http://localhost:8000/v1")
            self.llm = VLLMProvider(vllm_endpoint)
        else:
            # Fallback to ollama
            ollama_profile = self.config.get("llm", {}).get("ollama", {}).get("active_profile", "fast")
            ollama_model = self.config.get("llm", {}).get("ollama", {}).get("profiles", {}).get(ollama_profile, "llama3.2:3b")
            self.llm = OllamaProvider(ollama_model)
        
        vad_mode = self.config.get("vad", {}).get("mode", 3)
        self.recorder = AudioRecorder(vad_mode=vad_mode)
        self.state = "idle"

    async def _process_query(self, text: str):
        self.state = "processing"
        async for token in self.llm.stream(text):
            # streaming → incremental speaking
            self.tts_service.speak_tokens([token])
        self.state = "idle"

    async def listen(self):
        """
        Wait for a speech segment (energy+VAD or Porcupine wake-word) and trigger the assistant.
        Select behavior via env WAKE_MODE = "vad" (default) or "porcupine".
        """
        self.state = "listening"
        wake_mode = os.environ.get("WAKE_MODE", str(self.config.get("wake_mode", "porcupine"))).lower()

        # Porcupine (default) with graceful fallback
        if wake_mode == "porcupine":
            if not PORCUPINE_AVAILABLE:
                print("[wake] Porcupine not available – falling back to VAD. Install pvporcupine and set PICOVOICE_ACCESS_KEY.")
            else:
                keywords = os.environ.get("PORCUPINE_KEYWORDS", "computer").split(",")
                access_key = os.environ.get("PICOVOICE_ACCESS_KEY", "").strip()
                if not access_key:
                    print("[wake] Missing PICOVOICE_ACCESS_KEY. Get one free at https://console.picovoice.ai and set it before running. Falling back to VAD.")
                else:
                    # Block until wake word detected, then record a short utterance
                    with PorcupineWake(keywords=[k.strip() for k in keywords if k.strip()], access_key=access_key) as wake:
                        detected = wake.wait()
                        if not detected:
                            return
                # after wake, capture speech for a few seconds
                frames = []
                start_ts = time.time()
                capture_sec = int(self.config.get("vad", {}).get("capture_duration", 5))
                while time.time() - start_ts < capture_sec:
                            f = next(self.recorder)
                            frames.append(f)
                        audio_bytes = b"".join(frames)
                        transcript = self.whisper.transcribe(audio_bytes)["text"]
                        await self._process_query(transcript)
                        return

        # Default VAD path
        energy_threshold = self.config.get("vad", {}).get("energy_threshold", 2500)
        for frame in self.recorder:
            if self.recorder.energy(frame) > energy_threshold and self.recorder.is_speech(frame):
                # collect configured seconds of audio
                frames = []
                start_ts = time.time()
                capture_sec = int(self.config.get("vad", {}).get("capture_duration", 5))
                while time.time() - start_ts < capture_sec:
                    f = next(self.recorder)
                    frames.append(f)
                audio_bytes = b"".join(frames)
                transcript = self.whisper.transcribe(audio_bytes)["text"]
                await self._process_query(transcript)

    async def run(self):
        """
        Main loop – keeps listening forever.
        """
        while True:
            await self.listen()

    def stop(self):
        self.recorder.stop()
