import win32com.client
from typing import Optional

class WindowsTTS:
    """
    Simple SAPI5 wrapper – useful on machines that lack a GPU.
    """
    def __init__(self):
        self.synth = win32com.client.Dispatch("SAPI.SpVoice")

    def speak(self, text: str, blocking: bool = True) -> Optional[int]:
        """
        Speak text synchronously.
        Returns the number of chars spoken or None if an error occurs.
        """
        try:
            self.synth.Speak(text, 0 if blocking else 1)
            return len(text)
        except Exception:
            return None
