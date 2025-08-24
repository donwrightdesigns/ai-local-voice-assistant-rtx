#!/usr/bin/env python3
"""
Azure Cognitive Services TTS Demo
Requires: pip install azure-cognitiveservices-speech
And Azure Speech API key + region in environment variables
"""

import os
import azure.cognitiveservices.speech as speechsdk

def demo_azure_tts():
    """Demo Azure TTS voices"""
    
    # Check for API key and region
    api_key = os.getenv('AZURE_SPEECH_KEY')
    region = os.getenv('AZURE_SPEECH_REGION', 'eastus')  # Default to eastus
    
    if not api_key:
        print("❌ Please set AZURE_SPEECH_KEY environment variable")
        print("Get your key from: https://portal.azure.com")
        print("Also set AZURE_SPEECH_REGION (e.g., 'eastus')")
        return
    
    try:
        # Set up the speech config
        speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
        
        test_text = "Hello! This is a demonstration of Azure's neural text to speech system. I'm one of the most natural sounding voices available."
        
        # Some good English voices to demo
        voices = [
            ("en-US-JennyNeural", "Jenny (US Female - Conversational)"),
            ("en-US-GuyNeural", "Guy (US Male - Natural)"),
            ("en-US-AriaNeural", "Aria (US Female - Cheerful)"),
            ("en-US-DavisNeural", "Davis (US Male - Professional)"),
            ("en-US-AmberNeural", "Amber (US Female - Young)"),
            ("en-US-BrandonNeural", "Brandon (US Male - Young)"),
        ]
        
        print("🎤 Azure TTS Voice Demo")
        print("=" * 40)
        
        for voice_name, description in voices:
            print(f"\n🎵 Testing: {description}")
            print("Generating speech...")
            
            try:
                speech_config.speech_synthesis_voice_name = voice_name
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
                
                result = synthesizer.speak_text_async(test_text).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print("✅ Speech generated successfully")
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = result.cancellation_details
                    print(f"❌ Speech synthesis canceled: {cancellation_details.reason}")
                    if cancellation_details.error_details:
                        print(f"Error details: {cancellation_details.error_details}")
                
                input("Press Enter to try next voice...")
                
            except Exception as e:
                print(f"❌ Error with voice {voice_name}: {e}")
        
        print("✅ Azure TTS demo complete!")
        
    except Exception as e:
        print(f"❌ Azure TTS error: {e}")

if __name__ == "__main__":
    demo_azure_tts()
