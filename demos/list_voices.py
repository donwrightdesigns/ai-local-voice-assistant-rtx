#!/usr/bin/env python3
"""
Quick script to list current Windows SAPI voices
"""

import win32com.client

def list_windows_voices():
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        
        print("🎤 Current Windows SAPI Voices:")
        print("=" * 50)
        
        for i in range(voices.Count):
            voice = voices.Item(i)
            name = voice.GetDescription()
            print(f"  {i}: {name}")
            
        print(f"\nTotal voices: {voices.Count}")
        return voices.Count
        
    except Exception as e:
        print(f"Error listing voices: {e}")
        return 0

if __name__ == "__main__":
    list_windows_voices()
