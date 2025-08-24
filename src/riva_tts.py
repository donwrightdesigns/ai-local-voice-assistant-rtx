#!/usr/bin/env python3
"""
NVIDIA Riva TTS Service
High-quality, fast local TTS using NVIDIA Riva
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Tuple
import tempfile
import os

# Riva client imports
RIVA_AVAILABLE = False
try:
    import riva.client
    RIVA_AVAILABLE = True
    print("✅ NVIDIA Riva client available")
except ImportError:
    print("❌ NVIDIA Riva client not available - install with: pip install nvidia-riva-client")

class RivaTTS:
    def __init__(self, server_url: str = "localhost:50051"):
        """Initialize Riva TTS client"""
        self.server_url = server_url
        self.client = None
        self.sample_rate = 22050
        self.voices = []
        
        if RIVA_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Riva client connection"""
        try:
            print(f"🎤 Connecting to Riva server at {self.server_url}...")
            auth = riva.client.Auth(uri=self.server_url, use_ssl=False)
            self.client = riva.client.SpeechSynthesisService(auth)
            
            # Get available voices
            self.voices = self.client.get_voices().voices
            if self.voices:
                print(f"✅ Riva TTS initialized with {len(self.voices)} voices:")
                for i, voice in enumerate(self.voices):
                    print(f"  {i}: {voice.name} ({voice.language_code})")
            else:
                print("⚠️  No voices found on Riva server")
                
        except Exception as e:
            print(f"❌ Failed to connect to Riva server: {e}")
            print("Make sure Riva server is running on localhost:50051")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Riva TTS is available"""
        return RIVA_AVAILABLE and self.client is not None and len(self.voices) > 0
    
    def get_voice_list(self):
        """Get list of available voices"""
        if not self.is_available():
            return []
        return [(i, voice.name, voice.language_code) for i, voice in enumerate(self.voices)]
    
    def synthesize_speech(self, text: str, voice_index: int = 0) -> Tuple[int, np.ndarray]:
        """
        Synthesize speech using Riva
        Returns: (sample_rate, audio_array)
        """
        if not self.is_available():
            return 22050, np.array([])
        
        if not text or not text.strip():
            return 22050, np.array([])
        
        if voice_index >= len(self.voices):
            voice_index = 0
        
        try:
            voice = self.voices[voice_index]
            print(f"🎵 Synthesizing with voice: {voice.name}")
            
            # Create synthesis request
            req = riva.client.SynthesizeSpeechRequest(
                text=text.strip(),
                language_code=voice.language_code,
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=self.sample_rate,
                voice_name=voice.name
            )
            
            # Get response from Riva
            response = self.client.synthesize(req)
            
            # Convert audio bytes to numpy array
            audio_data = np.frombuffer(response.audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            print(f"✅ Generated {len(audio_data)} samples at {self.sample_rate} Hz")
            return self.sample_rate, audio_data
            
        except Exception as e:
            print(f"❌ Riva synthesis error: {e}")
            return 22050, np.array([])
    
    def play_speech(self, text: str, voice_index: int = 0):
        """Synthesize and play speech"""
        sample_rate, audio_data = self.synthesize_speech(text, voice_index)
        
        if len(audio_data) > 0:
            print("🔊 Playing audio...")
            sd.play(audio_data, sample_rate)
            sd.wait()
            return True
        return False
    
    def get_current_voice_name(self, voice_index: int = 0) -> str:
        """Get name of current voice"""
        if self.is_available() and voice_index < len(self.voices):
            return f"{self.voices[voice_index].name} ({self.voices[voice_index].language_code})"
        return "No voice available"

def test_riva_tts():
    """Test function for Riva TTS"""
    print("🎤 Testing NVIDIA Riva TTS")
    print("=" * 50)
    
    riva_tts = RivaTTS()
    
    if not riva_tts.is_available():
        print("❌ Riva TTS not available")
        print("Make sure Riva server is running:")
        print("  docker run --gpus all -it --rm -p 50051:50051 nvcr.io/nvidia/riva/riva-speech:2.14.0")
        return
    
    test_text = "Hello! This is NVIDIA Riva text to speech running on your RTX 4070. I should sound natural and be very fast to generate!"
    
    # Test available voices
    voices = riva_tts.get_voice_list()
    for i, (idx, name, lang) in enumerate(voices):
        print(f"\n🎵 Testing voice {i}: {name} ({lang})")
        success = riva_tts.play_speech(test_text, idx)
        
        if success:
            print("✅ Voice played successfully!")
        else:
            print("❌ Voice playback failed")
        
        if i < len(voices) - 1:  # Don't prompt after last voice
            input("Press Enter to try next voice...")
    
    print("\n✅ Riva TTS test complete!")

if __name__ == "__main__":
    test_riva_tts()
