#!/usr/bin/env python3
"""
Improved pyttsx3 TTS Service
Better control over Windows SAPI voices with rate/pitch adjustments
"""

import pyttsx3
import numpy as np
from typing import Tuple

class PyTTSX3Service:
    def __init__(self):
        """Initialize pyttsx3 TTS engine"""
        self.engine = None
        self.voices = []
        self.current_voice_index = 0
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the pyttsx3 engine"""
        try:
            print("🎤 Initializing pyttsx3 TTS...")
            self.engine = pyttsx3.init()
            self.voices = self.engine.getProperty('voices')
            
            if self.voices:
                print(f"✅ pyttsx3 initialized with {len(self.voices)} voices")
                self._optimize_voice_settings()
            else:
                print("❌ No voices found")
                self.engine = None
                
        except Exception as e:
            print(f"❌ pyttsx3 initialization failed: {e}")
            self.engine = None
    
    def _optimize_voice_settings(self):
        """Optimize voice settings for better sound"""
        if not self.engine:
            return
            
        # Try to use Zira (female) as default if available, as it often sounds better
        zira_index = None
        for i, voice in enumerate(self.voices):
            if 'zira' in voice.name.lower():
                zira_index = i
                break
        
        if zira_index is not None:
            self.current_voice_index = zira_index
            self.engine.setProperty('voice', self.voices[zira_index].id)
            print(f"🎵 Using voice: {self.voices[zira_index].name}")
        
        # Optimize settings
        self.engine.setProperty('rate', 190)    # Slightly faster than default (200)
        self.engine.setProperty('volume', 0.9)  # Slightly quieter for better quality
    
    def is_available(self) -> bool:
        """Check if TTS is available"""
        return self.engine is not None and len(self.voices) > 0
    
    def get_voice_list(self):
        """Get list of available voices"""
        if not self.is_available():
            return []
        return [(i, voice.name) for i, voice in enumerate(self.voices)]
    
    def set_voice(self, voice_index: int):
        """Set voice by index"""
        if not self.is_available() or voice_index >= len(self.voices):
            return False
        
        self.current_voice_index = voice_index
        self.engine.setProperty('voice', self.voices[voice_index].id)
        print(f"🎵 Voice changed to: {self.voices[voice_index].name}")
        return True
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        if self.engine:
            rate = max(50, min(400, rate))  # Clamp to reasonable range
            self.engine.setProperty('rate', rate)
    
    def speak(self, text: str) -> bool:
        """Speak text"""
        if not self.is_available() or not text:
            return False
        
        try:
            # Clean text for better speech
            clean_text = text.replace("🎤", "").replace("🤖", "").replace("👤", "")
            clean_text = clean_text.replace("*", "").replace("_", "")
            
            self.engine.say(clean_text)
            self.engine.runAndWait()
            return True
            
        except Exception as e:
            print(f"❌ pyttsx3 speech error: {e}")
            return False
    
    def get_current_voice_name(self) -> str:
        """Get current voice name"""
        if self.is_available():
            return self.voices[self.current_voice_index].name
        return "No voice available"
    
    def next_voice(self):
        """Switch to next voice"""
        if self.is_available():
            next_index = (self.current_voice_index + 1) % len(self.voices)
            self.set_voice(next_index)

def test_pyttsx3_service():
    """Test the pyttsx3 service"""
    print("🎤 Testing Improved pyttsx3 TTS")
    print("=" * 50)
    
    tts = PyTTSX3Service()
    
    if not tts.is_available():
        print("❌ pyttsx3 TTS not available")
        return
    
    # Test different rates and voices
    test_text = "Hello! This is an improved pyttsx3 text to speech test. I'm trying to sound better than the default Windows voice."
    
    voices = tts.get_voice_list()
    for i, (idx, name) in enumerate(voices):
        print(f"\n🎵 Testing voice {i}: {name}")
        tts.set_voice(idx)
        
        # Test different rates
        for rate in [160, 180, 200]:
            print(f"  Rate: {rate} WPM")
            tts.set_rate(rate)
            tts.speak(test_text)
            
            if i < len(voices) - 1 or rate < 200:
                input("Press Enter for next test...")
    
    print("\n✅ pyttsx3 test complete!")

if __name__ == "__main__":
    test_pyttsx3_service()
