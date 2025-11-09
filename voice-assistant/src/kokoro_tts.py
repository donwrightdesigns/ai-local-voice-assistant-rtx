import torch
import numpy as np
from typing import List

class KokoroTTS:
    """
    Lightweight wrapper around the Kokoro TTS library.
    """
    def __init__(self, model_name: str = "kokoro-en", device: str = "cuda"):
        # The public repo exposes `torch.hub.load` interface
        self.tts = torch.hub.load("kokoro/tts", "tts", model_name=model_name, device=device)
        self.sample_rate = getattr(self.tts, "sample_rate", 22050)   # default if not exposed

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize full text → NumPy array of float32 samples in [-1, 1].
        """
        wav = self.tts.synthesize(text)          # returns torch.Tensor
        return wav.cpu().numpy()

    def synthesize_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Synthesize token‑by‑token (for streaming LLMs).
        """
        return self.synthesize(" ".join(tokens))
