#!/usr/bin/env python3
"""
Windows SAPI Text-to-Speech Service
Alternative to Bark for testing Windows built-in voices
"""

import win32com.client
import os
from typing import List, Optional

class WindowsTTS:
    def __init__(self):
        """Initialize Windows SAPI TTS"""
        try:
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.voices = self.speaker.GetVoices()
            self.current_voice_index = 0
            
            # Set default voice to first available
            if self.voices.Count > 0:
                self.speaker.Voice = self.voices.Item(self.current_voice_index)
                
            print(f"✅ Windows SAPI TTS initialized with {self.voices.Count} voices")
            self.list_voices()
            
        except Exception as e:
            print(f"❌ Windows SAPI TTS initialization failed: {e}")
            self.speaker = None

    def list_voices(self):
        """List all available Windows voices"""
        if not self.speaker:
            return
            
        print("\n🎤 Available Windows Voices:")
        for i in range(self.voices.Count):
            voice = self.voices.Item(i)
            name = voice.GetDescription()
            current = "👉 " if i == self.current_voice_index else "   "
            print(f"{current}{i}: {name}")
        print()

    def get_voice_list(self) -> List[str]:
        """Get list of voice names"""
        if not self.speaker:
            return []
        
        voice_list = []
        for i in range(self.voices.Count):
            voice = self.voices.Item(i)
            voice_list.append(voice.GetDescription())
        return voice_list

    def set_voice(self, voice_identifier: any) -> bool:
        """Set voice by index or name"""
        if not self.speaker:
            return False

        if isinstance(voice_identifier, int):
            if voice_identifier < 0 or voice_identifier >= self.voices.Count:
                return False
            self.speaker.Voice = self.voices.Item(voice_identifier)
            self.current_voice_index = voice_identifier
            return True
        elif isinstance(voice_identifier, str):
            for i in range(self.voices.Count):
                voice = self.voices.Item(i)
                if voice.GetDescription().lower() == voice_identifier.lower():
                    self.speaker.Voice = voice
                    self.current_voice_index = i
                    return True
            return False
        else:
            return False

    def next_voice(self):
        """Switch to next available voice"""
        if not self.speaker or self.voices.Count <= 1:
            return
            
        next_index = (self.current_voice_index + 1) % self.voices.Count
        self.set_voice(next_index)

    def set_rate(self, rate: int):
        """Set speech rate (-10 to 10, 0 is default)"""
        if self.speaker:
            self.speaker.Rate = max(-10, min(10, rate))

    def set_volume(self, volume: int):
        """Set volume (0 to 100)"""
        if self.speaker:
            self.speaker.Volume = max(0, min(100, volume))

    def speak(self, text: str, async_speech: bool = True):
        """Speak text using Windows SAPI"""
        if not self.speaker or not text:
            return False
            
        try:
            # Clean up text for better speech
            clean_text = text.replace("🎤", "").replace("🤖", "").replace("👤", "")
            clean_text = clean_text.replace("*", "").replace("_", "")
            
            if async_speech:
                # Non-blocking speech
                self.speaker.Speak(clean_text, 1)  # 1 = SVSFlagsAsync
            else:
                # Blocking speech  
                self.speaker.Speak(clean_text, 0)  # 0 = SVSFDefault (synchronous)
            return True
            
        except Exception as e:
            print(f"❌ Windows TTS error: {e}")
            return False

    def stop_speaking(self):
        """Stop current speech"""
        if self.speaker:
            self.speaker.Speak("", 2)  # 2 = SVSFPurgeBeforeSpeak

    def get_current_rate(self) -> int:
        """Get current speech rate"""
        if self.speaker:
            return self.speaker.Rate
        return 0

    def get_current_volume(self) -> int:
        """Get current volume"""
        if self.speaker:
            return self.speaker.Volume
        return 100
        """Get current voice name"""
        if self.speaker and self.voices.Count > 0:
            return self.voices.Item(self.current_voice_index).GetDescription()
        return "No voice available"

def test_windows_tts():
    """Test function to try out Windows voices"""
    tts = WindowsTTS()
    if not tts.speaker:
        print("Windows TTS not available")
        return
    
    test_text = "Hello! This is a test of the Windows text to speech system. How do I sound?"
    
    print("Testing all available voices...")
    for i in range(tts.voices.Count):
        tts.set_voice(i)
        voice_name = tts.get_current_voice()
        print(f"\n🎤 Testing voice {i}: {voice_name}")
        print("Speaking test text...")
        tts.speak(test_text, async_speech=False)  # Wait for completion
        
        input("Press Enter to try next voice (or Ctrl+C to stop)...")

if __name__ == "__main__":
    test_windows_tts()
