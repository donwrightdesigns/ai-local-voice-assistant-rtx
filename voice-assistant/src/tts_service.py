import numpy as np
import sounddevice as sd
from typing import Iterable
from .kokoro_tts import KokoroTTS

# Optional Windows fallback import
WindowsTTS = None
try:
    from .windows_tts import WindowsTTS
except ImportError:
    pass

class TTSService:
    """
    Kokoro TTS primary with silent Windows SAPI fallback.
    No engine switching - Kokoro only.
    """
    def __init__(self, fallback: bool = True, model_name: str = "kokoro-en", device: str = "cuda"):
        self.tts = KokoroTTS(model_name=model_name, device=device)
        self.win_tts = WindowsTTS() if fallback and WindowsTTS else None

    def speak(self, text: str):
        wav = self.tts.synthesize(text)          # NumPy float32 [-1, 1]
        self._play(wav, self.tts.sample_rate, text)

    def speak_tokens(self, tokens: Iterable[str]):
        """Incrementally speak a list of tokens (used for streaming)."""
        text = " ".join(tokens)
        wav = self.tts.synthesize(text)  # simplified: full synthesis for each buffer
        self._play(wav, self.tts.sample_rate, text)

    def _play(self, wav: np.ndarray, fs: int, original_text: str = ""):
        try:
            if len(wav) > 0:
                sd.play(wav, fs)
                sd.wait()
        except Exception:
            # Silent fallback to Windows SAPI if available
            if self.win_tts and original_text:
                try:
                    self.win_tts.speak(original_text)
                except Exception:
                    pass  # Silent failure - no console spam
