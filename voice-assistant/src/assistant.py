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
from .agent_setup import create_agent, build_langchain_llm
from faster_whisper import WhisperModel
import os

# LangChain imports
try:
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    ConversationBufferMemory = None
    ConversationChain = None
    PromptTemplate = None

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
        self.whisper = WhisperModel(stt_model, device=stt_device, compute_type=self.config.get("stt", {}).get("compute_type", "int8"))
        
        # ------------------ TTS ------------------
        tts_fallback = self.config.get("tts", {}).get("fallback", True)
        tts_device = self.config.get("tts", {}).get("device", "cuda")
        tts_model = self.config.get("tts", {}).get("model", "kokoro-en")
        self.tts_service = TTSService(fallback=tts_fallback, model_name=tts_model, device=tts_device)
        
        # ------------------ LLM + Agent ------------------
        # Build LangChain LLM instance
        llm_config = self.config.get("llm", {})
        llm_provider_config = llm_config.get(provider, {})
        self.llm_instance = build_langchain_llm(provider, llm_provider_config)
        
        # Create ReAct agent with memory
        system_prompt = self.config.get("prompts", {}).get("system_prompt", None)
        self.agent = create_agent(self.llm_instance, system_prompt)
        
        vad_mode = self.config.get("vad", {}).get("mode", 3)
        self.recorder = AudioRecorder(vad_mode=vad_mode)
        self.state = "idle"
        
        # Store wake words for display
        self.wake_words = os.environ.get("PORCUPINE_KEYWORDS", "okay luna,hey luna")

        # Hotkey state
        self._keyboard = KeyboardController() if PYNPUT_AVAILABLE else None
        self._record_flag = False
        self._mode_active: str | None = None
        self._frame_buffer: list[bytes] = []
        
        # Display initialization banner
        mode = self.config.get("mode", "hotkey").lower()
        print("\n" + "="*70)
        print("🎤 VOICE ASSISTANT - INITIALIZING")
        print("="*70)
        print(f"Mode: {mode}")
        if mode == "vad" or mode != "hotkey":
            print(f"Wake Words: {self.wake_words}")
        print(f"LLM: {self.config.get('llm', {}).get('provider', 'ollama')}")
        print(f"STT: {stt_model}")
        print(f"TTS: {self.config.get('tts', {}).get('engine', 'auto')}")
        print("="*70 + "\n")

    async def _process_query(self, text: str):
        self.state = "processing"
        try:
            # Use LangChain agent with memory and reasoning
            response = self.agent.invoke({"input": text})
            output = response.get("output", "Sorry, I couldn't process that.")
            
            # Speak the response in chunks
            if output:
                sentences = [s.strip() + "." for s in output.split(".") if s.strip()]
                self.tts_service.speak_tokens(sentences)
        except Exception as e:
            print(f"[error] Query processing failed: {e}")
            self.tts_service.speak_tokens(["Sorry, something went wrong."])
        finally:
            self.state = "idle"

    async def listen(self):
        """
        Wait for a speech segment (energy+VAD or Porcupine wake-word) and trigger the assistant.
        Select behavior via env WAKE_MODE = "vad" (default) or "porcupine".
        """
        self.state = "listening"
        wake_mode = os.environ.get("WAKE_MODE", str(self.config.get("wake_mode", "porcupine"))).lower()
        
        # Display welcome banner on first listen
        if not hasattr(self, '_listen_banner_shown'):
            print("\n" + "="*70)
            print("🎤 VOICE ASSISTANT - LISTENING MODE")
            print("="*70)
            print("✅ Ready! Waiting for voice...")
            print("")
            
            if wake_mode == "porcupine":
                porcupine_key = os.environ.get("PICOVOICE_ACCESS_KEY", "").strip()
                keywords = os.environ.get("PORCUPINE_KEYWORDS", "okay luna,hey luna").split(",")
                if not porcupine_key:
                    print("Wake Mode: VAD (Porcupine not configured - falling back)")
                    print("Just speak and the assistant will listen.")
                else:
                    keywords_str = ", ".join([k.strip() for k in keywords if k.strip()])
                    print(f"Wake Word(s): {keywords_str}")
                    print("Say the wake word, then your question.")
            else:
                print("Wake Mode: VAD (automatic speech detection)")
                print("Just speak and the assistant will listen.")
            
            print("")
            print("Hotkeys:")
            print("• Press Ctrl+C to exit")
            print("="*70 + "\n")
            self._listen_banner_shown = True

        # Porcupine (default) with graceful fallback
        if wake_mode == "porcupine":
            if not PORCUPINE_AVAILABLE:
                print("[wake] Porcupine not available – falling back to VAD. Install pvporcupine and set PICOVOICE_ACCESS_KEY.")
            else:
                keywords = os.environ.get("PORCUPINE_KEYWORDS", "okay luna,hey luna").split(",")
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
        try:
            import numpy as np
            # Convert raw PCM bytes to numpy float32 array (16-bit signed @ 16kHz)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = self.whisper.transcribe(audio_np, beam_size=5)
            transcript = " ".join([seg.text for seg in segments])
        except Exception as e:
            print(f"[error] Transcription failed: {e}")
            return
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
        
        # Display welcome banner with hotkey instructions
        print("\n" + "="*70)
        print("🎤 VOICE ASSISTANT - HOTKEY MODE")
        print("="*70)
        print("✅ Ready! Use these hotkeys:")
        print("")
        print("• Ctrl+F2  - Conversation (AI responds with voice)")
        print("• Ctrl+F1  - Dictation (types what you say)")
        print("• F15      - AI Typing (AI types a response)")
        print("• F14      - Screen AI (AI sees screen + responds with voice)")
        print("• Escape   - Exit")
        print("")
        print("Hold the key while speaking, release when done.")
        print("="*70 + "\n")
        
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

        try:
            with Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
        finally:
            # Ensure audio resources are released when hotkey loop exits (e.g. ESC)
            try:
                self.stop()
            except Exception:
                pass

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
