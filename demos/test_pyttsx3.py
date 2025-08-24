#!/usr/bin/env python3
"""
Test pyttsx3 voices - might find more voices than basic SAPI
"""

import pyttsx3

def test_pyttsx3_voices():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        print("🎤 Available pyttsx3 Voices:")
        print("=" * 50)
        
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name}")
            print(f"      ID: {voice.id}")
            print(f"      Languages: {getattr(voice, 'languages', 'Unknown')}")
            print()
        
        print(f"Total voices found: {len(voices)}")
        
        # Test the first voice
        if voices:
            print("Testing first voice...")
            engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', 180)  # Adjust speed
            engine.say("Hello! This is a test of pyttsx3 text to speech. How do I sound?")
            engine.runAndWait()
        
    except Exception as e:
        print(f"Error with pyttsx3: {e}")

if __name__ == "__main__":
    test_pyttsx3_voices()
