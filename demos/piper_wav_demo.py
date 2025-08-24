#!/usr/bin/env python3
"""
Piper TTS WAV File Demo
Generates WAV files you can play manually
"""

import os
import urllib.request

def download_voice_if_needed(voice_name):
    """Download Piper voice model if not exists"""
    voice_dir = "piper_voices"
    os.makedirs(voice_dir, exist_ok=True)
    
    onnx_file = os.path.join(voice_dir, f"{voice_name}.onnx")
    json_file = os.path.join(voice_dir, f"{voice_name}.onnx.json")
    
    if os.path.exists(onnx_file) and os.path.exists(json_file):
        return onnx_file
    
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

def generate_piper_wav(text, voice_file, output_path):
    """Generate WAV file using Piper"""
    try:
        from piper import PiperVoice
        
        voice = PiperVoice.load(voice_file)
        
        with open(output_path, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        
        return True
        
    except Exception as e:
        print(f"Error generating WAV: {e}")
        return False

def main():
    test_text = "Hello! This is Piper text to speech. I'm running offline on your Windows computer and should sound much better than the robotic Windows voices."
    
    voices = [
        ("en_US-amy-medium", "Amy - US Female (Medium Quality)"),
        ("en_US-lessac-high", "Lessac - US Female (High Quality)"),  
        ("en_US-ryan-high", "Ryan - US Male (High Quality)")
    ]
    
    print("🎤 Piper TTS WAV Generator")
    print("=" * 40)
    
    generated_files = []
    
    for voice_name, description in voices:
        print(f"\n🎵 Generating: {description}")
        
        voice_file = download_voice_if_needed(voice_name)
        if not voice_file:
            print("❌ Could not download voice model")
            continue
        
        output_wav = f"{voice_name}_demo.wav"
        print(f"Creating WAV file: {output_wav}")
        
        success = generate_piper_wav(test_text, voice_file, output_wav)
        
        if success and os.path.exists(output_wav):
            file_size = os.path.getsize(output_wav)
            print(f"✅ Generated {output_wav} ({file_size} bytes)")
            generated_files.append((output_wav, description))
        else:
            print("❌ WAV generation failed")
    
    if generated_files:
        print("\n" + "=" * 50)
        print("🎧 GENERATED WAV FILES - Play them manually:")
        print("=" * 50)
        for wav_file, desc in generated_files:
            print(f"• {wav_file} - {desc}")
        
        print("\nDouble-click each WAV file to hear the voice quality!")
        print("Or use: start filename.wav")
    else:
        print("\n❌ No WAV files were generated successfully")

if __name__ == "__main__":
    main()
