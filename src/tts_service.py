#!/usr/bin/env python3
"""
Dual TTS Service - Supports both Bark and Windows SAPI
Allows easy switching between TTS engines
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
from typing import Optional, Tuple

# Try to import Bark
BARK_AVAILABLE = False
SAMPLE_RATE = 22050  # Default sample rate
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    BARK_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError) as e:
    # Catch import errors, missing dependencies, or broken installations
    BARK_AVAILABLE = False
    generate_audio = None
    preload_models = None

# Try to import pyttsx3
PYTTSX3_AVAILABLE = False
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    print("pyttsx3 not available - install with: pip install pyttsx3")

# Try to import Windows SAPI
WINDOWS_TTS_AVAILABLE = False
try:
    from windows_tts import WindowsTTS
    WINDOWS_TTS_AVAILABLE = True
except ImportError:
    print("Windows TTS not available")

# Try to import Kokoro TTS
KOKORO_AVAILABLE = False
try:
    from kokoro_tts import KokoroTTS
    KOKORO_AVAILABLE = True
except ImportError:
    print("Voice Assistant RTX TTS not available - install with: pip install kokoro>=0.9.2 soundfile")



class TextToSpeechService:
    def __init__(self, engine="auto", voice="", lang_code="a"):
        """
        Initialize TTS service
        engine options: "auto", "pyttsx3", "bark", "windows", "kokoro", "none"
        voice: specific voice name to use (e.g., "zira", "david", "af_heart")
        lang_code: language code for Kokoro ('a' for American English)
        """
        self.engine = None
        self.bark_ready = False
        self.windows_tts = None
        self.pyttsx3_tts = None
        self.kokoro_tts = None
        self.preferred_voice = voice.lower().strip() if voice else ""
        self.lang_code = lang_code
        
        if engine == "auto":
            # Priority: Kokoro (best quality), pyttsx3, Windows, Bark
            if KOKORO_AVAILABLE:
                engine = "kokoro"
            elif PYTTSX3_AVAILABLE:
                engine = "pyttsx3"
            elif WINDOWS_TTS_AVAILABLE:
                engine = "windows"
            elif BARK_AVAILABLE:
                engine = "bark"
            else:
                engine = "none"
        
        self.current_engine = engine
        self._initialize_engine(engine)
        
    def _initialize_engine(self, engine: str):
        """Initialize the specified TTS engine"""
        self.engine = None

        if engine == "kokoro" and KOKORO_AVAILABLE:
            self._init_kokoro()
        elif engine == "bark" and BARK_AVAILABLE:
            self._init_bark()
        elif engine == "pyttsx3" and PYTTSX3_AVAILABLE:
            self._init_pyttsx3()
        elif engine == "windows" and WINDOWS_TTS_AVAILABLE:
            self._init_windows_tts()
        elif engine == "none":
            self.engine = "none"
            print("🔇 TTS disabled")
        else:
            print("❌ No TTS engine available")

    def _init_bark(self):
        try:
            print("🎤 Initializing Bark TTS...")
            preload_models()
            self.bark_ready = True
            self.engine = "bark"
            print("✅ Bark TTS initialized")
        except Exception as e:
            print(f"❌ Bark initialization failed: {e}")

    def _init_pyttsx3(self):
        try:
            print("🎤 Initializing pyttsx3 TTS...")
            import pyttsx3
            self.pyttsx3_tts = pyttsx3.init()
            if self.pyttsx3_tts:
                # Configure voice settings
                voices = self.pyttsx3_tts.getProperty('voices')
                voice_set = False
                
                if voices:
                    # First try to use the preferred voice from config
                    if self.preferred_voice:
                        for voice in voices:
                            if self.preferred_voice in voice.name.lower():
                                self.pyttsx3_tts.setProperty('voice', voice.id)
                                print(f"Selected voice: {voice.name}")
                                voice_set = True
                                break
                    
                    # If preferred voice not found or not specified, try female/zira
                    if not voice_set:
                        for voice in voices:
                            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                self.pyttsx3_tts.setProperty('voice', voice.id)
                                print(f"Selected voice: {voice.name}")
                                voice_set = True
                                break
                    
                    # Show current voice if no specific selection was made
                    if not voice_set and voices:
                        current_voice = self.pyttsx3_tts.getProperty('voice')
                        for voice in voices:
                            if voice.id == current_voice:
                                print(f"Using default voice: {voice.name}")
                                break
                                
                # Set speech rate and volume
                self.pyttsx3_tts.setProperty('rate', 200)  # Speed of speech
                self.pyttsx3_tts.setProperty('volume', 0.9)  # Volume level
                self.engine = "pyttsx3"
                print("✅ pyttsx3 TTS initialized")
            else:
                print("❌ pyttsx3 engine failed to initialize")
        except Exception as e:
            print(f"❌ pyttsx3 initialization failed: {e}")


    def _init_windows_tts(self):
        try:
            print("🎤 Initializing Windows SAPI TTS...")
            self.windows_tts = WindowsTTS()
            self.engine = "windows"
            print("✅ Windows SAPI TTS initialized")
        except Exception as e:
            print(f"❌ Windows TTS initialization failed: {e}")

    def _init_kokoro(self):
        try:
            print("🎤 Initializing Voice Assistant RTX TTS...")
            # Use preferred voice if specified, otherwise default to af_heart
            voice = self.preferred_voice if self.preferred_voice else "af_heart"
            self.kokoro_tts = KokoroTTS(voice=voice, lang_code=self.lang_code)
            if self.kokoro_tts.is_available():
                self.engine = "kokoro"
                print("✅ Voice Assistant RTX TTS initialized")
            else:
                print("❌ Voice Assistant RTX TTS failed to initialize")
        except Exception as e:
            print(f"❌ Voice Assistant RTX TTS initialization failed: {e}")

    def switch_engine(self, new_engine: str) -> bool:
        if new_engine == self.engine:
            print(f"Already using {new_engine} TTS")
            return True
            
        old_engine = self.engine
        self._initialize_engine(new_engine)
        
        if self.engine == new_engine:
            print(f"🔄 Switched from {old_engine} to {new_engine} TTS")
            return True
        else:
            print(f"❌ Failed to switch to {new_engine} TTS")
            return False

    def get_current_engine(self) -> str:
        """Get current TTS engine name"""
        return self.engine if self.engine else "none"

    def is_available(self) -> bool:
        """Check if TTS is available and working"""
        return self.engine is not None and self.engine != "none"


    def _bark_synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """Synthesize using Bark"""
        try:
            # Use a simple preset for faster generation
            audio_array = generate_audio(text, history_prompt="v2/en_speaker_6")
            return SAMPLE_RATE, audio_array
        except Exception as e:
            print(f"❌ Bark synthesis error: {e}")
            return 22050, np.array([])

    def _pyttsx3_synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """Synthesize using pyttsx3 (plays directly)"""
        try:
            if self.pyttsx3_tts:
                # Clear any pending speech
                self.pyttsx3_tts.stop()
                self.pyttsx3_tts.say(text)
                # Use a safer approach to avoid run loop conflicts
                try:
                    self.pyttsx3_tts.runAndWait()
                except RuntimeError as e:
                    if "run loop already started" in str(e):
                        print("⚠️  TTS run loop conflict - speech may be queued")
                    else:
                        raise e
                # pyttsx3 plays directly, return empty array
                return 22050, np.array([])
            return 22050, np.array([])
        except Exception as e:
            print(f"❌ pyttsx3 synthesis error: {e}")
            return 22050, np.array([])
    
    
    def _windows_synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Synthesize using Windows SAPI
        Note: Windows SAPI plays directly, so we return empty array
        """
        try:
            if self.windows_tts:
                success = self.windows_tts.speak(text, async_speech=False)
                if success:
                    # Windows TTS plays directly, return empty array
                    return 22050, np.array([])
            return 22050, np.array([])
        except Exception as e:
            print(f"❌ Windows TTS synthesis error: {e}")
            return 22050, np.array([])

    def _kokoro_synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """Synthesize using Kokoro TTS"""
        try:
            if self.kokoro_tts:
                return self.kokoro_tts.synthesize(text)
            return 24000, np.array([])
        except Exception as e:
            print(f"❌ Kokoro synthesis error: {e}")
            return 24000, np.array([])

    def _threaded_pyttsx3_speak(self, text: str):
        """Thread-safe pyttsx3 speech function"""
        try:
            import pyttsx3
            # Create a new engine instance for this thread
            engine = pyttsx3.init()
            if engine:
                # Configure voice settings (copy from main instance)
                if self.pyttsx3_tts:
                    try:
                        voice = self.pyttsx3_tts.getProperty('voice')
                        rate = self.pyttsx3_tts.getProperty('rate')
                        volume = self.pyttsx3_tts.getProperty('volume')
                        
                        engine.setProperty('voice', voice)
                        engine.setProperty('rate', rate)
                        engine.setProperty('volume', volume)
                    except:
                        pass  # Use defaults if can't copy settings
                
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print(f"❌ pyttsx3 threaded speak error: {e}")

    def speak_direct(self, text: str) -> bool:
        """
        Speak text directly (for Windows TTS which plays immediately)
        """
        if not text or not self.is_available():
            return False

        if self.engine == "kokoro" and self.kokoro_tts:
            # For Kokoro, synthesize and play
            sample_rate, audio_array = self._kokoro_synthesize(text)
            if len(audio_array) > 0:
                sd.play(audio_array, sample_rate)
                sd.wait()
                return True
            return False

        elif self.engine == "pyttsx3" and self.pyttsx3_tts:
            # Use threaded approach to avoid run loop conflicts
            try:
                thread = threading.Thread(target=self._threaded_pyttsx3_speak, args=(text,), daemon=True)
                thread.start()
                # Don't wait for thread to complete - return immediately
                return True
            except Exception as e:
                print(f"❌ pyttsx3 speak error: {e}")
                return False
            
        elif self.engine == "bark":
            # For Bark, synthesize and play
            sample_rate, audio_array = self._bark_synthesize(text)
            if len(audio_array) > 0:
                sd.play(audio_array, sample_rate)
                sd.wait()
                return True
            return False
            
        elif self.engine == "windows" and self.windows_tts:
            return self.windows_tts.speak(text, async_speech=False)
        
        return False

    def get_voice_info(self) -> str:
        """Get information about current voice"""
        if self.engine == "kokoro" and self.kokoro_tts:
            return f"Voice Assistant RTX TTS - {self.kokoro_tts.get_current_voice()}"
        elif self.engine == "bark":
            return "Bark TTS (v2/en_speaker_6)"
        elif self.engine == "pyttsx3" and self.pyttsx3_tts:
            try:
                voice = self.pyttsx3_tts.getProperty('voice')
                voices = self.pyttsx3_tts.getProperty('voices')
                current_voice_name = "Unknown"
                for v in voices:
                    if v.id == voice:
                        current_voice_name = v.name
                        break
                return f"pyttsx3 - {current_voice_name}"
            except:
                return "pyttsx3 TTS"
        elif self.engine == "windows" and self.windows_tts:
            return f"Windows SAPI - {self.windows_tts.get_current_voice()}"
        elif self.engine == "none":
            return "TTS Disabled"
        else:
            return "No TTS Available"

    def next_voice(self):
        """Switch to next available voice"""
        if self.engine == "kokoro" and self.kokoro_tts:
            # Kokoro voice cycling
            voices = [
                "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
                "am_adam", "am_michael",
                "bf_emma", "bf_isabella",
                "bm_george", "bm_lewis"
            ]
            current = self.kokoro_tts.get_current_voice()
            try:
                current_idx = voices.index(current)
                next_idx = (current_idx + 1) % len(voices)
                self.kokoro_tts.set_voice(voices[next_idx])
            except ValueError:
                self.kokoro_tts.set_voice(voices[0])
        elif self.engine == "pyttsx3" and self.pyttsx3_tts:
            try:
                voices = self.pyttsx3_tts.getProperty('voices')
                current_voice = self.pyttsx3_tts.getProperty('voice')
                if voices and len(voices) > 1:
                    current_index = 0
                    for i, voice in enumerate(voices):
                        if voice.id == current_voice:
                            current_index = i
                            break
                    next_index = (current_index + 1) % len(voices)
                    self.pyttsx3_tts.setProperty('voice', voices[next_index].id)
                    print(f"Switched to voice: {voices[next_index].name}")
                else:
                    print("No other voices available")
            except Exception as e:
                print(f"Error switching voice: {e}")
        elif self.engine == "windows" and self.windows_tts:
            self.windows_tts.next_voice()
        elif self.engine == "bark":
            # Bark has a fixed voice for now
            print("Bark TTS only has one voice.")
        else:
            print(f"Voice switching not supported for {self.engine}")

def test_tts_engines():
    """Test both TTS engines"""
    test_text = "This is a test of the text to speech system. How do I sound?"
    
    print("=" * 60)
    print("🎤 Testing TTS Engines")
    print("=" * 60)
    
    # Test Bark
    if BARK_AVAILABLE:
        print("\n🎤 Testing Bark TTS...")
        bark_tts = TextToSpeechService("bark")
        if bark_tts.is_available():
            bark_tts.speak_direct(test_text)
        input("Press Enter to continue to Windows TTS...")
    
    # Test Windows
    if WINDOWS_TTS_AVAILABLE:
        print("\n🎤 Testing Windows TTS...")
        win_tts = TextToSpeechService("windows")
        if win_tts.is_available():
            win_tts.speak_direct(test_text)
    
    print("✅ TTS testing complete!")

if __name__ == "__main__":
    test_tts_engines()
