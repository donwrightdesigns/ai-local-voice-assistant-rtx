import torch
import numpy as np
from typing import List

class KokoroTTS:
    """
    Lightweight wrapper around the Kokoro TTS library.
    """
    def __init__(self, model_name: str = "kokoro-en", device: str = "cuda"):
        # Use direct kokoro package import
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code="a" if "en" in model_name else "a")
        self.voice = "af_heart"  # default voice
        self.sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize full text → NumPy array of float32 samples in [-1, 1].
        """
        audio_segments = []
        for _, _, audio in self.pipeline(text, voice=self.voice):
            audio_segments.append(audio)
        if audio_segments:
            return np.concatenate(audio_segments)
        return np.array([], dtype=np.float32)

    def synthesize_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Synthesize token‑by‑token (for streaming LLMs).
        """
        return self.synthesize(" ".join(tokens))
