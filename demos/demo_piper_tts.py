#!/usr/bin/env python3
"""
Piper TTS Demo
Requires: pip install piper-tts onnxruntime soundfile sounddevice
"""

import os
import subprocess
import tempfile
import soundfile as sf
import sounddevice as sd

def download_piper_voice(voice_name, base_url):
    """Download a Piper voice model if it doesn't exist"""
    voice_dir = f"piper_voices"
    os.makedirs(voice_dir, exist_ok=True)
    
    onnx_file = os.path.join(voice_dir, f"{voice_name}.onnx")
    json_file = os.path.join(voice_dir, f"{voice_name}.onnx.json")
    
    if not os.path.exists(onnx_file):
        print(f"Downloading {voice_name} model...")
        import urllib.request
        urllib.request.urlretrieve(f"{base_url}/{voice_name}.onnx", onnx_file)
        urllib.request.urlretrieve(f"{base_url}/{voice_name}.onnx.json", json_file)
    
    return onnx_file

def test_piper_voice(text, voice_file):
    """Test a Piper voice by generating and playing audio"""
    try:
        # Import piper after potential installation
        from piper import PiperVoice
        
        voice = PiperVoice.load(voice_file)
        
        # Generate audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
            
        with open(wav_path, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        
        # Play audio
        data, samplerate = sf.read(wav_path)
        sd.play(data, samplerate)
        sd.wait()
        
        # Cleanup
        os.unlink(wav_path)
        
        return True
        
    except Exception as e:
        print(f"Error with Piper voice: {e}")
        return False

def demo_piper_voices():
    """Demo different Piper voices"""
    
    test_text = "Hello! This is a demonstration of Piper text to speech. I'm a neural voice that runs completely offline on your computer."
    
    # Define voices to test
    voices = [
        {
            "name": "en_US-amy-medium",
            "desc": "Amy (US Female - Medium Quality)",
            "base": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
        },
        {
            "name": "en_US-kristin-high", 
            "desc": "Kristin (US Female - High Quality)",
            "base": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kristin/high"
        },
        {
            "name": "en_US-ryan-high",
            "desc": "Ryan (US Male - High Quality)", 
            "base": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
        }
    ]
    
    print("🎤 Piper TTS Voice Demo")
    print("=" * 50)
    print("Note: First run will download voice models (~50MB each)")
    print()
    
    for voice in voices:
        print(f"🎵 Testing: {voice['desc']}")
        
        try:
            # Download voice if needed
            voice_file = download_piper_voice(voice['name'], voice['base'])
            
            print("Generating speech...")
            success = test_piper_voice(test_text, voice_file)
            
            if success:
                print("✅ Voice played successfully")
            else:
                print("❌ Voice failed to play")
                
        except Exception as e:
            print(f"❌ Error with {voice['name']}: {e}")
        
        input("Press Enter to try next voice...")
    
    print("✅ Piper TTS demo complete!")

if __name__ == "__main__":
    try:
        demo_piper_voices()
    except ImportError as e:
        print("❌ Missing dependencies. Please install:")
        print("pip install piper-tts onnxruntime soundfile sounddevice")
        print(f"Error: {e}")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
