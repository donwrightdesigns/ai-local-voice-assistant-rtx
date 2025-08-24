#!/usr/bin/env python3
"""
Attempt to fix Bark TTS by setting PyTorch to allow unsafe loading
WARNING: This bypasses PyTorch security features!
"""

import torch
import warnings

def fix_bark_loading():
    """Try to fix Bark by allowing unsafe PyTorch loading"""
    
    print("🔧 Attempting to fix Bark loading issue...")
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Set torch to allow unsafe loading (security risk!)
    torch.serialization.add_safe_globals([
        'numpy.core.multiarray.scalar',
        'numpy.dtype',
        'numpy.ndarray',
        'collections.OrderedDict'
    ])
    
    try:
        print("📦 Importing Bark...")
        from bark import generate_audio, preload_models
        
        print("⏳ Loading Bark models (this may take a while)...")
        preload_models()
        
        print("🎤 Testing Bark generation...")
        test_text = "Hello! This is a test of Bark text to speech. If you can hear this, the fix worked!"
        
        audio_array = generate_audio(test_text, history_prompt="v2/en_speaker_6")
        
        print("✅ Bark is working! Playing test audio...")
        
        # Play the audio
        import sounddevice as sd
        from bark import SAMPLE_RATE
        
        sd.play(audio_array, SAMPLE_RATE)
        sd.wait()
        
        print("🎉 Bark TTS fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Bark fix failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_bark_loading()
    if success:
        print("✅ You can now use Bark in your voice assistant!")
    else:
        print("❌ Bark fix didn't work. Consider using Azure or OpenAI TTS instead.")
