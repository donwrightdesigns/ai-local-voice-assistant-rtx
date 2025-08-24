#!/usr/bin/env python3
"""
Quick test of automatic TTS selection
"""

from src.tts_service import TextToSpeechService

def test_auto_tts():
    print("🎤 Testing Automatic TTS Selection")
    print("=" * 50)
    
    tts = TextToSpeechService("auto")  # Should pick pyttsx3 automatically
    
    if tts.is_available():
        print(f"✅ Selected engine: {tts.get_current_engine()}")
        print(f"✅ Voice info: {tts.get_voice_info()}")
        
        test_text = "Hello! I'm using the optimized text to speech system. Zira should sound much better now with the improved settings."
        print("🔊 Speaking test message...")
        tts.speak_direct(test_text)
        
    else:
        print("❌ No TTS available")

if __name__ == "__main__":
    test_auto_tts()
