import numpy as np
import sounddevice as sd
from typing import Iterable
from .kokoro_tts import KokoroTTS
from .windows_tts import WindowsTTS
from .config import KOKORO_MODEL

class TTSService:
    """
    High‑level interface that handles
    – Kokoro TTS synthesis
    – optional Windows fallback
    – synchronous playback
    """
    def __init__(self, fallback: bool = False, model_name: str = None, device: str = None):
        name = model_name or KOKORO_MODEL
        dev = device or "cuda"
        self.tts = KokoroTTS(model_name=name, device=dev)
        self.fallback = fallback
        if fallback:
            self.win_tts = WindowsTTS()

    def speak(self, text: str):
        wav = self.tts.synthesize(text)          # NumPy float32 [-1, 1]
        self._play(wav, self.tts.sample_rate)

    def speak_tokens(self, tokens: Iterable[str]):
        """Incrementally speak a list of tokens (used for streaming)."""
        buffer = []
        for tok in tokens:
            buffer.append(tok)
            wav = self.tts.synthesize_tokens(buffer)
            self._play(wav, self.tts.sample_rate)

    def _play(self, wav: np.ndarray, fs: int):
        try:
            sd.play(wav, fs)
            sd.wait()
        except Exception:
            if self.fallback:
                # fallback to SAPI5
                self.win_tts.speak(" ".join(wav.tolist()))  # crude, but works
