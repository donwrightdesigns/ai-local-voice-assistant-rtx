#!/usr/bin/env python3
"""
Kokoro TTS - High-quality, fast TTS using Kokoro-82M model
"""

import numpy as np
import torch
from typing import Optional, Tuple


class KokoroTTS:
    def __init__(self, voice: str = "af_heart", lang_code: str = "a"):
        """
        Initialize Voice Assistant RTX TTS (Kokoro engine)
        
        Args:
            voice: Voice preset (e.g., "af_heart", "am_adam", "bf_emma", etc.)
                   See https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
            lang_code: Language code ('a' for American English, 'b' for British, etc.)
        """
        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=lang_code)
            self.voice = voice
            self.sample_rate = 24000
            self.available = True
            print(f"✅ Voice Assistant RTX TTS ready (voice: {voice})")
        except ImportError:
            print("❌ Voice Assistant RTX TTS engine not installed. Install with: pip install kokoro>=0.9.2 soundfile")
            print("   Also requires espeak-ng: choco install espeak-ng (or download from GitHub)")
            self.pipeline = None
            self.available = False
        except Exception as e:
            print(f"❌ Voice Assistant RTX TTS init failed: {e}")
            self.pipeline = None
            self.available = False
    
    def is_available(self) -> bool:
        """Check if Kokoro TTS is ready"""
        return self.available and self.pipeline is not None
    
    def synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (sample_rate, audio_array)
        """
        if not self.is_available():
            return self.sample_rate, np.array([])
        
        try:
            # Generate audio using Kokoro pipeline
            generator = self.pipeline(text, voice=self.voice)
            
            # Kokoro returns a generator - we'll concatenate all segments
            audio_segments = []
            for _, _, audio in generator:
                audio_segments.append(audio)
            
            # Concatenate all audio segments
            if audio_segments:
                full_audio = np.concatenate(audio_segments)
                return self.sample_rate, full_audio
            else:
                return self.sample_rate, np.array([])
                
        except Exception as e:
            print(f"❌ Voice Assistant RTX TTS synthesis error: {e}")
            return self.sample_rate, np.array([])
    
    def set_voice(self, voice: str):
        """Change the voice preset"""
        self.voice = voice
        print(f"Voice Assistant RTX voice changed to: {voice}")
    
    def get_current_voice(self) -> str:
        """Get current voice preset"""
        return self.voice
    
    def list_voices(self):
        """Print available voice presets"""
        voices = [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]
        print("\nAvailable voices (Voice Assistant RTX):")
        for voice in voices:
            marker = "→" if voice == self.voice else " "
            print(f"  {marker} {voice}")
        print("\nSee more at: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md")


def test_kokoro():
    """Test Kokoro TTS with sample text"""
    print("Testing Kokoro TTS...")
    
    kokoro = KokoroTTS(voice="af_heart")
    
    if not kokoro.is_available():
        print("Kokoro is not available. Please install dependencies.")
        return
    
    test_text = "Hello! This is Kokoro, an open-weight text to speech model with 82 million parameters."
    
    print(f"Generating speech for: {test_text}")
    sample_rate, audio = kokoro.synthesize(test_text)
    
    if len(audio) > 0:
        print(f"✅ Generated {len(audio)} samples at {sample_rate}Hz")
        print(f"   Duration: {len(audio) / sample_rate:.2f} seconds")
        
        # Play the audio
        try:
            import sounddevice as sd
            print("Playing audio...")
            sd.play(audio, sample_rate)
            sd.wait()
            print("✅ Playback complete!")
        except ImportError:
            print("sounddevice not available for playback")
    else:
        print("❌ No audio generated")


if __name__ == "__main__":
    test_kokoro()
