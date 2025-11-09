import pyaudio
import webrtcvad
import numpy as np
from typing import Generator

CHUNK = 512                      # 10 ms at 16 kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioRecorder:
    """
    Non‑blocking audio recorder that yields raw PCM frames.
    """
    def __init__(self, vad_mode: int = 3):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self.vad = webrtcvad.Vad(int(vad_mode))

    def __iter__(self) -> Generator[bytes, None, None]:
        for _ in iter(None, None):
            frame = self.stream.read(CHUNK, exception_on_overflow=False)
            yield frame

    def is_speech(self, frame: bytes) -> bool:
        return self.vad.is_speech(frame, RATE)

    def energy(self, frame: bytes) -> float:
        return np.linalg.norm(np.frombuffer(frame, dtype=np.int16))

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
