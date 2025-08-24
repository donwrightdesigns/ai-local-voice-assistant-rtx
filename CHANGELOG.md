# Changelog

All notable changes to the Voice Assistant for Windows project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub repository setup with proper documentation

## [1.0.0] - 2024-08-24

### Added
- 🎤 **Ultimate Voice Assistant for Windows** with system-wide hotkeys
- 🗨️ **Conversation Mode** (`Ctrl+F2`) - AI chat with text-to-speech responses
- 📝 **Dictation Mode** (`Ctrl+F1`) - Direct speech-to-text transcription
- 🤖 **AI Typing Mode** (`F15`) - AI processes speech and types response at cursor
- 📸 **Screen AI Mode** (`F14`) - AI analyzes screen content and provides contextual responses
- 🧠 **AI Integration** with Ollama for local LLM processing
- 🎧 **Advanced Speech Recognition** using OpenAI's Whisper model
- 🔊 **Multiple TTS Engines** support (Windows SAPI, Piper TTS, pyttsx3)
- 📷 **Screenshot Analysis** capability for visual context
- 💭 **Conversation Memory** with reset functionality (`Menu` key)
- ⚡ **Custom Hotkey Support** via environment variables
- 🎛️ **Configurable Settings** through YAML configuration
- 🔒 **Singleton Pattern** to prevent multiple instances
- ⌨️ **Text Injection** with smart character handling and normalization

### Features
- System-wide hotkey detection that works in any application
- Real-time audio recording and processing
- Conversation state management with LangChain
- Rich console output with colored status messages
- Intelligent text typing with first-character drop prevention
- Support for custom hardware keys (FLIRC remotes, etc.)
- Configurable Whisper models (tiny.en to large-v3)
- GPU acceleration support for Whisper transcription
- Multiple compute types (int8, float16, int8_float16)
- Automatic screenshot resizing for optimal processing
- Base64 image encoding for AI vision models

### Technical Details
- **Core Framework**: Python 3.8+ with modern async/await patterns
- **Speech-to-Text**: faster-whisper for optimized Whisper inference
- **AI/LLM**: Ollama integration with conversation chains
- **Audio**: sounddevice for cross-platform audio input
- **UI**: Rich library for beautiful console output
- **System Integration**: pynput for global hotkeys and text injection
- **Configuration**: YAML-based settings management
- **Image Processing**: Pillow and pyautogui for screenshot capabilities

### Configuration Options
- Ollama server URL and model selection
- Whisper model size and compute settings
- TTS engine selection and voice configuration
- Custom hotkey combinations
- AI prompt customization
- Performance tuning parameters

### Demo Scripts
- `demo_piper_tts.py` - Test Piper text-to-speech
- `demo_azure_tts.py` - Test Azure Cognitive Services TTS
- `demo_openai_tts.py` - Test OpenAI TTS API
- `test_pyttsx3.py` - Test pyttsx3 TTS engine
- `simple_piper_demo.py` - Simple Piper TTS demonstration
- `test_auto_tts.py` - Automated TTS testing

### Dependencies
- torch>=2.0.0 (PyTorch for AI models)
- transformers>=4.35.0 (Hugging Face transformers)
- langchain>=0.3.0 (LLM conversation chains)
- faster-whisper>=1.1.0 (Optimized Whisper inference)
- sounddevice>=0.5.0 (Audio input/output)
- rich>=13.0.0 (Rich text and beautiful formatting)
- pynput==1.7.6 (Global hotkeys and text injection)
- pyautogui>=0.9.54 (Screenshot capabilities)
- Pillow>=10.0.0 (Image processing)
- pyyaml>=6.0 (YAML configuration parsing)

### Known Limitations
- Windows-specific implementation (Windows 10/11 required)
- Requires Ollama server for AI functionality
- Some hotkeys may conflict with system or application shortcuts
- GPU acceleration requires compatible CUDA installation
- Large Whisper models require significant memory

### Performance Notes
- Default configuration optimized for CPU usage (int8 quantization)
- GPU acceleration available for faster transcription
- Smaller Whisper models (tiny.en, base.en) for real-time performance
- Conversation memory can be reset to manage memory usage
- Screenshot analysis may introduce latency for large displays

---

## Version History Notes

### Version Numbering
- **MAJOR**: Breaking changes, new architecture, or major feature overhauls
- **MINOR**: New features, significant improvements, or API additions
- **PATCH**: Bug fixes, minor improvements, or documentation updates

### Release Types
- **Alpha**: Early development versions with experimental features
- **Beta**: Feature-complete versions undergoing testing and refinement
- **Release Candidate (RC)**: Near-final versions with minimal remaining issues
- **Stable**: Production-ready releases with full documentation and support

### Changelog Categories
- **Added**: New features and capabilities
- **Changed**: Modifications to existing functionality
- **Deprecated**: Features being phased out (still available but discouraged)
- **Removed**: Features that have been completely removed
- **Fixed**: Bug fixes and corrections
- **Security**: Vulnerability patches and security improvements

---

*For detailed technical documentation, see [README.md](README.md)*
