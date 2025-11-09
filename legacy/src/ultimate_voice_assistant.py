#!/usr/bin/env python3
"""
Ultimate Unified Voice Assistant for Windows
Combines the best of both voice-assistant and vibevoice projects:
- Ctrl+f2: Conversation Mode (AI chat with TTS)
- Ctrl+F1: Dictation Mode (system-wide text injection)
- Ctrl+f4: AI Typing Mode (AI responses typed at cursor)
- Ctrl+f5: Screen Analysis Mode (AI sees screen + types response)
- Menu: Reset conversation
- Escape: Exit
"""

import sys

try:
    # Python 3.7+ supports reconfigure on text streams
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    # Fallback: set PYTHONUNBUFFERED for subprocesses, and we'll flush explicitly where needed.
    import os
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Now it's safe to import modules that may print during import
import time
import threading
import numpy as np
from faster_whisper import WhisperModel
import sounddevice as sd
from queue import Queue
from rich.console import Console

# LANGCHAIN MEMORY IMPORT - compatibility across versions
ConversationBufferMemory = None
try:
    # Newest, definitive import path for the currently installed Conda version (0.3.27)
    from langchain.memory.buffer import ConversationBufferMemory  # type: ignore
    ConversationBufferMemory = ConversationBufferMemory
except Exception:
    try:
        # Fallback 1 (New split packaging)
        from langchain_core.memory import ConversationBufferMemory  # type: ignore
        ConversationBufferMemory = ConversationBufferMemory
    except Exception:
        try:
            # Fallback 2 (Community package location)
            from langchain_community.memory import ConversationBufferMemory  # type: ignore
            ConversationBufferMemory = ConversationBufferMemory
        except Exception:
            # If all imports fail, keep ConversationBufferMemory as None
            ConversationBufferMemory = None

from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import yaml
import os
import sys
import base64
import subprocess
import requests
import json

# Phase 1: provider abstraction
from llm_providers import make_provider, OllamaProvider, ProviderResult

# Import pynput for hotkey detection and text injection
try:
    from pynput.keyboard import Key, Listener, KeyCode, Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except ImportError:
    print("pynput not available. Install with: pip install pynput")
    PYNPUT_AVAILABLE = False

# Screenshot support
SCREENSHOT_AVAILABLE = False
try:
    import pyautogui
    from PIL import Image
    SCREENSHOT_AVAILABLE = True
except ImportError:
    print("Screenshot functionality not available. Install with: pip install Pillow pyautogui")

# TTS support
try:
    from tts_service import TextToSpeechService
except ImportError:
    print("TTS service not available - will skip TTS functionality")
    TextToSpeechService = None

class UltimateVoiceAssistant:
    def __init__(self, config_path=None):
        self.console = Console()

        # Singleton pattern to ensure only one instance runs
        self.lock_file = os.path.abspath('voice_assistant.lock')
        if os.path.exists(self.lock_file):
            self.console.print("[red]Another instance is already running. Exiting.[/red]")
            sys.exit(1)
        
        # Create lock file
        with open(self.lock_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Smart config path resolution
        if config_path is None:
            config_path = self._find_config_file()
        
        self.console.print(f"[blue]📁 Using config file: {os.path.abspath(config_path)}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize Speech-to-Text
        stt_config = self.config['stt']
        self.stt = WhisperModel(
            stt_config['model'], 
            device=stt_config['device'], 
            compute_type=stt_config['compute_type']
        )
        
        # Initialize Text-to-Speech if available
        self.tts = None
        if TextToSpeechService:
            try:
                tts_config = self.config.get('tts', {})
                self.tts = TextToSpeechService(
                    engine=tts_config.get('engine', 'auto'),
                    voice=tts_config.get('voice', ''),
                    lang_code=tts_config.get('lang_code', 'a')
                )
                self.console.print(f"[green]✅ TTS initialized: {self.tts.get_voice_info()}")
            except Exception as e:
                self.console.print(f"[yellow]⚠️  TTS initialization failed: {e}")

        # Phase 1: backend selection and preflight
        backend = os.environ.get("VOICE_BACKEND", "ollama").strip().lower()
        mode = os.environ.get("VOICE_MODE", "faster").strip().lower()

        # Optional interactive selection if env not set
        if backend not in ("ollama", "local", "transformers") and sys.stdin.isatty():
            self.console.print("\n[bold]Select LLM backend[/bold]")
            self.console.print("  1) Ollama (default)")
            self.console.print("  2) Local Transformers (coming in Phase 2)")
            choice = input("Enter 1 or 2 (default 1): ").strip()
            backend = "ollama" if choice in ("", "1") else "local"

        # Build provider
        # Choose local model path based on VOICE_MODE if provided in config
        local_cfg = self.config.get('local', {})
        local_model_path = local_cfg.get('faster_path') if mode == 'faster' else local_cfg.get('advanced_path', local_cfg.get('faster_path'))

        provider = make_provider(
            backend,
            base_url=self.config['ollama']['base_url'],
            model=self.config['ollama']['model'],
            model_path=local_model_path or self.config.get('local_model_path', ''),
            mode=mode,
        )

        # Preflight
        pr: ProviderResult = provider.preflight()
        if not pr.ok:
            if isinstance(provider, OllamaProvider):
                self.console.print("[red]❌ Ollama is not reachable.[/red]")
                self.console.print(
                    f"[yellow]- Tried: {self.config['ollama']['base_url']}\n"
                    "- Make sure Ollama is running. On Windows: start the Ollama service (or run 'ollama serve').\n"
                    "- After starting, try again."
                )
                # Fail fast to avoid confusing runtime errors
                sys.exit(2)
            else:
                self.console.print(f"[yellow]⚠️  {pr.message}")
                self.console.print("[yellow]Falling back to Ollama backend for now.")
                provider = make_provider(
                    "ollama",
                    base_url=self.config['ollama']['base_url'],
                    model=self.config['ollama']['model']
                )

        # Initialize LLM chain with conversation memory
        prompt_template = self.config['prompts']['system_prompt']
        prompt = PromptTemplate(input_variables=["history", "input"], template=prompt_template)

        llm = provider.build_langchain_llm()
        # Use ConversationBufferMemory only if it was successfully imported
        memory_obj = None
        if ConversationBufferMemory is not None:
            try:
                memory_obj = ConversationBufferMemory(ai_prefix="Assistant:")
            except Exception as e:
                self.console.print(f"[yellow]⚠️ Failed to instantiate ConversationBufferMemory: {e}")
                memory_obj = None
        else:
            self.console.print("[yellow]⚠️ langchain ConversationBufferMemory import failed - conversation history disabled")

        self.chain = ConversationChain(
            prompt=prompt,
            verbose=False,
            memory=memory_obj,
            llm=llm,
        )

        # Initialize keyboard controller for text injection
        if PYNPUT_AVAILABLE:
            self.keyboard_controller = KeyboardController()
        else:
            self.keyboard_controller = None

        # Recording state
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.current_mode = None
        self._ctrl_pressed = False
        
        # Hotkey configuration - optimized for practical use
        self.CONVERSATION_KEY = Key.f2          # Ctrl+F2: AI conversation with TTS
        self.DICTATION_KEY = Key.f1             # Ctrl+F1: Simple dictation (text injection)
        self.AI_TYPING_KEY = Key.f15            # F15: AI response typed at cursor (rarely used)
        self.SCREEN_AI_KEY = Key.f14            # F14: AI with screen context
        self.RESET_KEY = Key.menu               # Menu key: Reset conversation
        
        # Support for custom keys (FLIRC, etc.) - can be configured via environment
        self.EXTRA_DICTATION_KEYS = []
        extra_keys_str = os.environ.get("VOICE_EXTRA_KEYS", "")
        if extra_keys_str:
            for key_str in extra_keys_str.split(","):
                key_str = key_str.strip()
                if key_str and key_str.startswith("vk_"):
                    try:
                        vk_code = int(key_str.replace("vk_", ""))
                        self.EXTRA_DICTATION_KEYS.append(KeyCode(vk=vk_code))
                    except ValueError:
                        self.console.print(f"[yellow]Warning: Invalid VK code '{key_str}'")
        
        # Display initialization info (This is the full welcome banner)
        self.console.print(f"[green]✅ Ultimate Voice Assistant initialized")
        self.console.print(f"[green]✅ Model: {self.config['ollama']['model']}")
        self.console.print(f"[green]✅ Whisper: {self.config['stt']['model']}")
        self.console.print("\n" + "="*70)
        self.console.print("[bold cyan]🎤 ULTIMATE VOICE ASSISTANT STARTED!")
        self.console.print("="*70)
        self.console.print(f"[green]• [bold]Ctrl+F2[/bold] - Conversation Mode (AI chat with voice response)")
        self.console.print(f"[green]• [bold]Ctrl+F1[/bold] - Dictation Mode (types what you say)")
        self.console.print(f"[yellow]• [bold]F15[/bold] - AI Typing Mode (AI response typed at cursor)")
        self.console.print(f"[yellow]• [bold]F14[/bold] - Screen AI Mode (AI sees screen + types response)")
        if self.EXTRA_DICTATION_KEYS:
            self.console.print(f"[green]• [bold]Custom Keys[/bold] - Extra dictation keys configured")
        self.console.print(f"[blue]• [bold]Menu[/bold] - Reset conversation memory")
        self.console.print(f"[red]• [bold]Escape[/bold] - Exit")
        self.console.print(f"[blue]• TTS: {'Enabled' if self.tts else 'Disabled'}")
        self.console.print(f"[blue]• Screenshots: {'Enabled' if SCREENSHOT_AVAILABLE else 'Disabled'}")
        self.console.print("="*70 + "\n")

    def _find_config_file(self):
        """Intelligently find the config file in various locations"""
        possible_paths = [
            # Current directory
            'config.yaml',
            # Config subdirectory in current directory
            'config/config.yaml',
            # Parent directory config
            '../config/config.yaml',
            # Two levels up config (in case we're in a subdirectory of src)
            '../../config/config.yaml',
            # Absolute path relative to this script's location
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.yaml'),
            # Environment variable override
            os.environ.get('VOICE_ASSISTANT_CONFIG', '')
        ]
        
        # Remove empty paths
        possible_paths = [p for p in possible_paths if p]
        
        for config_path in possible_paths:
            if os.path.exists(config_path):
                return config_path
        
        # If no config found, create a helpful error message
        self.console.print("[red]❌ Could not find config.yaml in any of these locations:")
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            exists = "✅" if os.path.exists(path) else "❌"
            self.console.print(f"[yellow]  {exists} {abs_path}")
        
        self.console.print("\n[blue]💡 Solutions:")
        self.console.print("[blue]  1. Run from the project root directory")
        self.console.print("[blue]  2. Set VOICE_ASSISTANT_CONFIG environment variable")
        self.console.print("[blue]  3. Create config.yaml in current directory")
        
        raise FileNotFoundError("config.yaml not found in any expected location")

    def record_audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            self.console.print(f"[red]Audio status: {status}")
        if self.recording:
            self.audio_data.extend(indata.flatten())

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper"""
        if not audio_data:
            return ""
            
        audio_np = np.array(audio_data, dtype=np.float32)
        if len(audio_np) == 0:
            return ""
            
        try:
            segments, info = self.stt.transcribe(audio_np, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()
            return text
        except Exception as e:
            self.console.print(f"[red]Transcription error: {e}")
            return ""

    def get_llm_response(self, text, include_screenshot=False):
        """Get response from LLM with optional screenshot context"""
        try:
            if include_screenshot and SCREENSHOT_AVAILABLE:
                screenshot_path, screenshot_base64 = self.capture_screenshot()
                if screenshot_base64:
                    # The vision model is now handled by the ConversationChain
                    response = self.chain.predict(input=text, images=[screenshot_base64])
                else:
                    response = self.chain.predict(input=text)
            else:
                response = self.chain.predict(input=text)

            if response.startswith("Assistant:"):
                response = response[len("Assistant:"):].strip()
            return response

        except Exception as e:
            self.console.print(f"[red]LLM error: {e}")
            return "Sorry, I couldn't process that request."

    def capture_screenshot(self):
        """Capture screenshot and return path + base64 data"""
        if not SCREENSHOT_AVAILABLE:
            return None, None
            
        try:
            screenshot = pyautogui.screenshot()
            if not screenshot:
                self.console.print("[red]Screenshot failed.[/red]")
                return None, None

            screenshot_path = os.path.abspath('screenshot.png')
            
            # Resize if too large
            max_width = int(os.environ.get('SCREENSHOT_MAX_WIDTH', '1024'))
            width, height = screenshot.size
            if width > max_width:
                ratio = max_width / width
                new_width = max_width
                new_height = int(height * ratio)
                screenshot = screenshot.resize((new_width, new_height))
            
            screenshot.save(screenshot_path)
            
            with open(screenshot_path, "rb") as image_file:
                base64_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            return screenshot_path, base64_data
        except Exception as e:
            self.console.print(f"[red]Screenshot error: {e}")
            return None, None


    def type_text(self, text):
        """Type text at current cursor position"""
        if self.keyboard_controller and text:
            try:
                # Replace smart quotes
                normalized_text = text.replace('\u2019', "'").replace('\u2018', "'")
                
                # More aggressive approach to prevent first character drop
                time.sleep(0.1)  # Longer initial delay
                
                # Send a dummy space and backspace to "wake up" the input system
                self.keyboard_controller.type(' ')
                time.sleep(0.02)
                self.keyboard_controller.press(Key.backspace)
                self.keyboard_controller.release(Key.backspace)
                time.sleep(0.02)
                
                # Now type the actual text
                self.keyboard_controller.type(normalized_text)
                    
            except Exception as e:
                self.console.print(f"[red]Typing error: {e}")

    def process_voice_command(self, mode):
        """Process a voice command based on the mode"""
        if not self.audio_data:
            self.console.print("[yellow]No audio data captured")
            return
            
        # Transcribe
        self.console.print(f"[cyan]🎧 Transcribing... (Mode: {mode})")
        transcript = self.transcribe_audio(self.audio_data)
        
        if not transcript.strip():
            self.console.print("[yellow]No speech detected")
            return
            
        self.console.print(f"[yellow]👤 You: {transcript}")
        
        if mode == "conversation":
            # Right Shift: Full conversation with TTS
            self.console.print("[cyan]🧠 Thinking...")
            response = self.get_llm_response(transcript)
            self.console.print(f"[green]🤖 Assistant: {response}")
            
            if self.tts:
                self.console.print("[cyan]🔊 Speaking...")
                self.tts.speak_direct(response)
                
        elif mode == "dictation":
            # Right Ctrl: Simple dictation - just type what was said
            processed_transcript = transcript + " "
            self.console.print(f"[green]📝 Typing: '{processed_transcript}'")
            self.type_text(processed_transcript)
            
        elif mode == "ai_typing":
            # Right Alt: AI response typed at cursor
            self.console.print("[cyan]🧠 Thinking...")
            response = self.get_llm_response(transcript)
            self.console.print(f"[green]🤖 Assistant: {response}")
            self.console.print("[cyan]📝 Typing response...")
            self.type_text(response + " ")
            
        elif mode == "screen_ai":
            # Right Win: AI with screen context
            self.console.print("[cyan]📸 Capturing screen...")
            self.console.print("[cyan]🧠 Analyzing...")
            response = self.get_llm_response(transcript, include_screenshot=True)
            self.console.print(f"[green]🤖 Assistant: {response}")
            self.console.print("[cyan]📝 Typing response...")
            self.type_text(response + " ")

    def reset_conversation(self):
        """Reset conversation memory"""
        self.chain.memory.clear()
        self.console.print("[blue]🔄 Conversation reset")

    def on_press(self, key):
        """Handle key press events"""
        if self.recording:
            return  # Already recording
            
        # Check for Ctrl modifier combinations
        ctrl_pressed = hasattr(key, 'vk') or key in [Key.ctrl_l, Key.ctrl_r]
        
        try:
            # Check for Ctrl+F2 (conversation mode)
            if hasattr(self, '_ctrl_pressed') and self._ctrl_pressed and key == self.CONVERSATION_KEY:
                self.recording = True
                self.current_mode = "conversation"
                self.audio_data = []
                self.console.print("[green]🎤 [Ctrl+F2] Conversation Mode - Listening... (release to stop)")
                
            # Check for Ctrl+F1 (dictation mode)  
            elif hasattr(self, '_ctrl_pressed') and self._ctrl_pressed and (key == self.DICTATION_KEY or key in self.EXTRA_DICTATION_KEYS):
                self.recording = True
                self.current_mode = "dictation"
                self.audio_data = []
                self.console.print("[green]🎤 [Ctrl+F1] Dictation Mode - Listening... (release to stop)")
                
            # Track Ctrl key state
            elif key in [Key.ctrl_l, Key.ctrl_r]:
                self._ctrl_pressed = True
                
            elif key == self.AI_TYPING_KEY:
                self.recording = True
                self.current_mode = "ai_typing"
                self.audio_data = []
                self.console.print("[green]🎤 [F15] AI Typing Mode - Listening... (release to stop)")
                
            elif key == self.SCREEN_AI_KEY:
                self.recording = True
                self.current_mode = "screen_ai"
                self.audio_data = []
                self.console.print("[green]🎤 [F14] Screen AI Mode - Listening... (release to stop)")
                
            elif key == self.RESET_KEY:
                self.reset_conversation()
                
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events"""
        try:
            # Handle Ctrl key release
            if key in [Key.ctrl_l, Key.ctrl_r]:
                self._ctrl_pressed = False
                
            # Handle recording stop when keys are released (only if recording)
            if self.recording and (
                key == self.CONVERSATION_KEY or 
                key == self.DICTATION_KEY or 
                key == self.AI_TYPING_KEY or 
                key == self.SCREEN_AI_KEY or
                key in self.EXTRA_DICTATION_KEYS
            ):
                self.recording = False
                mode = self.current_mode
                self.console.print("[yellow]🎤 Processing...")
                # Process in separate thread to avoid blocking
                threading.Thread(target=self.process_voice_command, args=(mode,), daemon=True).start()
                
        except AttributeError:
            pass
            
        # Exit on Escape
        if key == Key.esc:
            self.console.print("[red]Exiting...")
            return False


    def run(self):
        """Main run loop with hotkey listening"""
        if not PYNPUT_AVAILABLE:
            self.console.print("[red]❌ pynput not available. Cannot start hotkey mode.")
            return

        try:
            # Start audio input stream
            with sd.InputStream(
                callback=self.record_audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.float32
            ):
                # Start keyboard listener
                with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                    listener.join()
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user")
        except Exception as e:
            self.console.print(f"[red]Error: {e}")
        finally:
            self.console.print("[blue]Ultimate Voice Assistant stopped[/blue]")
            # Remove lock file on exit
            if os.path.exists(self.lock_file):
                os.unlink(self.lock_file)

def main():
    assistant = UltimateVoiceAssistant()
    sys.stdout.flush()
    assistant.run()

if __name__ == "__main__":
    main()
