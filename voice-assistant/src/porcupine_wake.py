import pvporcupine
import pyaudio
import numpy as np
from typing import List, Optional

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_LENGTH = 512  # 32 ms at 16kHz for Porcupine

class PorcupineWake:
    """
    Context manager for Porcupine wake-word detection.

    Usage:
      with PorcupineWake(["computer", "jarvis"], access_key=os.environ.get("PICOVOICE_ACCESS_KEY")) as wake:
          if wake.wait():
              ...
    """
    def __init__(self, keywords: List[str], access_key: Optional[str] = None, sensitivities: Optional[List[float]] = None):
        if sensitivities is None:
            sensitivities = [0.6] * len(keywords)
        self.porcupine = pvporcupine.create(access_key=access_key, keywords=keywords, sensitivities=sensitivities)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            rate=RATE,
            channels=CHANNELS,
            format=FORMAT,
            input=True,
            frames_per_buffer=FRAME_LENGTH,
        )

    def wait(self) -> bool:
        """Blocks until a keyword is detected. Returns True if detected, False otherwise."""
        try:
            while True:
                pcm = self.stream.read(FRAME_LENGTH, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                result = self.porcupine.process(pcm)
                if result >= 0:
                    return True
        except KeyboardInterrupt:
            return False

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.p is not None:
            self.p.terminate()
            self.p = None
        if self.porcupine is not None:
            self.porcupine.delete()
            self.porcupine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
