import pyaudio
import webrtcvad
import numpy as np
from typing import Generator

# WebRTC VAD requires frame sizes of 10, 20, or 30 ms.
# At 16 kHz, that's 160, 320, or 480 samples respectively.
CHUNK = 480                      # 30 ms at 16 kHz (compatible with webrtcvad)
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
        self._running = True

    def __iter__(self) -> Generator[bytes, None, None]:
        while self._running:
            try:
                frame = self.stream.read(CHUNK, exception_on_overflow=False)
            except OSError:
                if not self._running:
                    break
                continue
            yield frame

    def is_speech(self, frame: bytes) -> bool:
        return self.vad.is_speech(frame, RATE)

    def energy(self, frame: bytes) -> float:
        return np.linalg.norm(np.frombuffer(frame, dtype=np.int16))

    def stop(self):
        self._running = False
        try:
            try:
                if self.stream and hasattr(self.stream, 'is_active'):
                    if self.stream.is_active():
                        self.stream.stop_stream()
            except OSError:
                # Stream already closed
                pass
            if self.stream:
                self.stream.close()
        finally:
            if self.p:
                self.p.terminate()
