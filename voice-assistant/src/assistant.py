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

# Optional hotkey and typing support
try:
    from pynput.keyboard import Key, Listener, Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False

# Optional screenshot support
SCREENSHOT_AVAILABLE = False
try:
    import pyautogui
    from PIL import Image
    SCREENSHOT_AVAILABLE = True
except Exception:
    SCREENSHOT_AVAILABLE = False

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

        # Hotkey state
        self._keyboard = KeyboardController() if PYNPUT_AVAILABLE else None
        self._record_flag = False
        self._mode_active: str | None = None
        self._frame_buffer: list[bytes] = []

    async def _process_query(self, text: str):
        self.state = "processing"
        # Simple buffered streaming for smoother TTS
        buf = []
        last_flush = time.time()
        async for token in self.llm.stream(text):
            buf.append(token)
            should_flush = False
            if any(token.endswith(p) for p in (".", "!", "?")):
                should_flush = True
            if len(" ".join(buf)) >= 80:
                should_flush = True
            if time.time() - last_flush > 0.6:
                should_flush = True
            if should_flush:
                chunk = " ".join(buf).strip()
                if chunk:
                    self.tts_service.speak_tokens([chunk])
                buf.clear()
                last_flush = time.time()
        # Flush any remainder
        if buf:
            chunk = " ".join(buf).strip()
            if chunk:
                self.tts_service.speak_tokens([chunk])
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

    def _type_text(self, text: str):
        if not self._keyboard:
            return
        try:
            # small wake-up
            self._keyboard.type(" ")
            import time as _t
            _t.sleep(0.02)
            # backspace the wake space
            from pynput.keyboard import Key as _Key
            self._keyboard.press(_Key.backspace)
            self._keyboard.release(_Key.backspace)
            _t.sleep(0.02)
            self._keyboard.type(text)
        except Exception:
            pass

    def _capture_screenshot_b64(self, max_width: int = 1024) -> str | None:
        if not SCREENSHOT_AVAILABLE:
            return None
        try:
            img = pyautogui.screenshot()
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)))
            import base64, io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None

    def _frame_reader(self):
        """Background reader that appends frames while recording flag is set."""
        for frame in self.recorder:
            if self._record_flag:
                self._frame_buffer.append(frame)

    def _consume_audio_and_process(self, mode: str):
        # Join frames and transcribe
        audio_bytes = b"".join(self._frame_buffer)
        self._frame_buffer.clear()
        if not audio_bytes:
            return
        transcript = self.whisper.transcribe(audio_bytes)["text"]
        if not transcript.strip():
            return
        if mode == "dictation":
            self._type_text(transcript + " ")
            return
        if mode == "ai_typing":
            # get llm response then type
            async def _go():
                async for tok in self.llm.stream(transcript):
                    pass
            # simpler: one-shot using buffered _process_query then type final
            # Here we just reuse streaming into a buffer
            async def _respond_and_type():
                out = []
                async for tok in self.llm.stream(transcript):
                    out.append(tok)
                txt = " ".join(out).strip()
                if txt:
                    self._type_text(txt + " ")
            asyncio.run(_respond_and_type())
            return
        if mode == "screen_ai":
            b64 = self._capture_screenshot_b64()
            prompt = transcript
            if b64:
                prompt = (
                    "You have access only to text. The following is a base64-encoded screenshot provided for context. "
                    "Describe what to do succinctly based on the user's request.\nSCREENSHOT_BASE64: "
                    + b64[:2048]
                    + "...\nUser: "
                    + transcript
                )
            asyncio.run(self._process_query(prompt))
            # Also type a short response for convenience handled by TTS already
            return
        # conversation (default)
        asyncio.run(self._process_query(transcript))

    def _run_hotkeys(self):
        if not PYNPUT_AVAILABLE:
            print("[hotkeys] pynput not available; cannot start hotkey mode.")
            return
        # Start frame reader thread
        import threading
        t = threading.Thread(target=self._frame_reader, daemon=True)
        t.start()

        # Key bindings
        # Defaults: Ctrl+F2 conversation, Ctrl+F1 dictation, F15 ai_typing, F14 screen_ai
        ctrl_down = {"state": False}

        def on_press(key):
            try:
                from pynput.keyboard import Key as _K
                if key in (_K.ctrl_l, _K.ctrl_r):
                    ctrl_down["state"] = True
                    return
                # Ctrl+F2
                if ctrl_down["state"] and key == _K.f2 and not self._record_flag:
                    self._record_flag = True
                    self._mode_active = "conversation"
                    self._frame_buffer.clear()
                # Ctrl+F1
                elif ctrl_down["state"] and key == _K.f1 and not self._record_flag:
                    self._record_flag = True
                    self._mode_active = "dictation"
                    self._frame_buffer.clear()
                # F15
                elif key == _K.f15 and not self._record_flag:
                    self._record_flag = True
                    self._mode_active = "ai_typing"
                    self._frame_buffer.clear()
                # F14
                elif key == _K.f14 and not self._record_flag:
                    self._record_flag = True
                    self._mode_active = "screen_ai"
                    self._frame_buffer.clear()
            except Exception:
                pass

        def on_release(key):
            try:
                from pynput.keyboard import Key as _K
                if key in (_K.ctrl_l, _K.ctrl_r):
                    ctrl_down["state"] = False
                # Stop when corresponding key releases
                if self._record_flag and (
                    key in (_K.f2, _K.f1, _K.f14, _K.f15)
                ):
                    self._record_flag = False
                    mode = self._mode_active or "conversation"
                    self._mode_active = None
                    # process in thread to not block listener
                    import threading
                    threading.Thread(target=self._consume_audio_and_process, args=(mode,), daemon=True).start()
                if key == _K.esc:
                    return False
            except Exception:
                pass

        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    async def run(self):
        """
        Main entry – hotkey mode or wake/VAD mode based on config.
        """
        mode = str(self.config.get("mode", "hotkey")).lower()
        if mode == "hotkey":
            self._run_hotkeys()
            return
        while True:
            await self.listen()

    def stop(self):
        self.recorder.stop()
