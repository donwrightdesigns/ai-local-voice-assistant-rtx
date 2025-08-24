#!/usr/bin/env python3
"""
Simple Piper TTS Demo
Uses direct piper CLI to avoid format issues
"""

import os
import subprocess
import sounddevice as sd
import soundfile as sf
import tempfile
import urllib.request

def download_voice_if_needed(voice_name):
    """Download Piper voice model if not exists"""
    voice_dir = "piper_voices"
    os.makedirs(voice_dir, exist_ok=True)
    
    onnx_file = os.path.join(voice_dir, f"{voice_name}.onnx")
    json_file = os.path.join(voice_dir, f"{voice_name}.onnx.json")
    
    if os.path.exists(onnx_file) and os.path.exists(json_file):
        return onnx_file
    
    # Use the correct Hugging Face URLs for Piper voices
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    
    voice_urls = {
        "en_US-amy-medium": f"{base_url}/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "en_US-amy-medium.json": f"{base_url}/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "en_US-lessac-high": f"{base_url}/en/en_US/lessac/high/en_US-lessac-high.onnx", 
        "en_US-lessac-high.json": f"{base_url}/en/en_US/lessac/high/en_US-lessac-high.onnx.json",
        "en_US-ryan-high": f"{base_url}/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "en_US-ryan-high.json": f"{base_url}/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
    }
    
    try:
        print(f"Downloading {voice_name}...")
        if voice_name in voice_urls:
            urllib.request.urlretrieve(voice_urls[voice_name], onnx_file)
            urllib.request.urlretrieve(voice_urls[f"{voice_name}.json"], json_file)
            return onnx_file
    except Exception as e:
        print(f"Download failed: {e}")
    
    return None

def test_piper_voice_direct(text, voice_file):
    """Test Piper voice using direct synthesis"""
    try:
        from piper import PiperVoice
        
        voice = PiperVoice.load(voice_file)
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = temp_wav.name
        
        # Synthesize to WAV file
        with open(wav_path, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        
        # Play the WAV file
        try:
            data, samplerate = sf.read(wav_path)
            print(f"Playing audio: {len(data)} samples at {samplerate} Hz")
            sd.play(data, samplerate)
            sd.wait()
            success = True
        except Exception as play_error:
            print(f"Playback error: {play_error}")
            success = False
        
        # Cleanup
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        
        return success
        
    except Exception as e:
        print(f"Voice synthesis error: {e}")
        return False

def main():
    test_text = "Hello! This is Piper text to speech. I'm running offline on your Windows computer and should sound much better than the robotic Windows voices."
    
    voices = [
        ("en_US-amy-medium", "Amy - US Female (Medium Quality)"),
        ("en_US-lessac-high", "Lessac - US Female (High Quality)"),  
        ("en_US-ryan-high", "Ryan - US Male (High Quality)")
    ]
    
    print("🎤 Simple Piper TTS Demo")
    print("=" * 40)
    
    for voice_name, description in voices:
        print(f"\n🎵 Testing: {description}")
        
        voice_file = download_voice_if_needed(voice_name)
        if not voice_file:
            print("❌ Could not download voice model")
            input("Press Enter for next voice...")
            continue
        
        print("Generating speech...")
        success = test_piper_voice_direct(test_text, voice_file)
        
        if success:
            print("✅ Voice played successfully!")
        else:
            print("❌ Voice playback failed")
        
        input("Press Enter for next voice...")
    
    print("\n✅ Piper demo complete!")

if __name__ == "__main__":
    main()
