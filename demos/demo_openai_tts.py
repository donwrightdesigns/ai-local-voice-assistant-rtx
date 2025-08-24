#!/usr/bin/env python3
"""
OpenAI TTS Demo
Requires: pip install openai
And OpenAI API key in environment variable OPENAI_API_KEY
"""

import os
from openai import OpenAI
import pygame
import io

def demo_openai_tts():
    """Demo OpenAI TTS voices"""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("Get your key from: https://platform.openai.com/api-keys")
        return
    
    try:
        client = OpenAI()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        test_text = "Hello! This is a demonstration of OpenAI's text to speech system. I sound quite natural, don't I?"
        
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        print("🎤 OpenAI TTS Voice Demo")
        print("=" * 40)
        
        for voice in voices:
            print(f"\n🎵 Testing voice: {voice}")
            print("Generating speech...")
            
            try:
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=test_text
                )
                
                # Play audio
                audio_data = io.BytesIO(response.content)
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                input(f"Press Enter to try next voice...")
                
            except Exception as e:
                print(f"❌ Error with voice {voice}: {e}")
        
        print("✅ OpenAI TTS demo complete!")
        
    except Exception as e:
        print(f"❌ OpenAI TTS error: {e}")

if __name__ == "__main__":
    demo_openai_tts()
