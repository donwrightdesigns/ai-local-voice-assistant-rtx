# Ultimate Voice Assistant for Windows

A powerful, AI-driven voice assistant designed specifically for Windows with system-wide hotkeys, speech-to-text transcription, AI conversation capabilities, and text-to-speech responses.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows/)

## 🌟 Features

### Voice Interaction Modes
- **🗨️ Conversation Mode** (`Ctrl+F2`) - AI chat with text-to-speech responses
- **📝 Dictation Mode** (`Ctrl+F1`) - Direct speech-to-text transcription 
- **🤖 AI Typing Mode** (`F15`) - AI processes your speech and types the response
- **📸 Screen AI Mode** (`F14`) - AI analyzes your screen and responds contextually

### Core Capabilities
- **System-wide hotkeys** - Works in any application
- **Advanced speech recognition** using OpenAI's Whisper model
- **AI conversation** powered by Ollama (local LLM)
- **Multiple TTS engines** (Windows SAPI, Piper TTS)
- **Screenshot analysis** for visual context
- **Conversation memory** with reset capability
- **Custom hotkey support** via environment variables

## 🚀 Quick Start

### Prerequisites
- Windows 10/11
- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed and running locally

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voice-assistant-windows.git
   cd voice-assistant-windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama**
   - Download from [ollama.com](https://ollama.com/)
   - Install a model: `ollama pull llama3.2:3b`
   - Ensure Ollama is running on `http://127.0.0.1:11434`

4. **Configure the assistant**
   - Edit `config/config.yaml` to customize settings
   - Adjust model, voice, and hotkey preferences

5. **Run the assistant**
   ```bash
   python src/ultimate_voice_assistant.py
   ```

## 🎮 Usage

### Hotkeys
- **`Ctrl+F2`** - Start conversation mode (AI chat with voice response)
- **`Ctrl+F1`** - Start dictation mode (types what you say)
- **`F15`** - Start AI typing mode (AI response typed at cursor)
- **`F14`** - Start screen AI mode (AI sees screen + types response)
- **`Menu`** - Reset conversation memory
- **`Escape`** - Exit the application

### How to Use
1. Press and hold the desired hotkey combination
2. Speak your message
3. Release the key to process
4. The assistant will respond according to the selected mode

### Custom Hotkeys
You can configure additional dictation keys using environment variables:
```bash
set VOICE_EXTRA_KEYS=vk_123,vk_124
```

## ⚙️ Configuration

The main configuration file is `config/config.yaml`:

```yaml
# Ollama Configuration
ollama:
  base_url: "http://127.0.0.1:11434"
  model: "llama3.2:3b"
  parameters:
    temperature: 0.7

# Text-to-Speech Configuration
tts:
  engine: "auto"  # auto, piper, windows
  voice: ""

# Speech-to-Text Configuration
stt:
  model: "base.en"
  device: "cpu"  # Use "cuda" for GPU acceleration
  compute_type: "int8"

# Prompts
prompts:
  system_prompt: |
    You are a helpful and friendly AI assistant. You are polite, 
    respectful, and aim to provide concise responses.
```

### Available Models
- **Whisper STT**: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3`
- **Ollama LLM**: Any model supported by Ollama (llama3.2, mistral, etc.)

## 🔧 Development

### Project Structure
```
voice-assistant-windows/
├── src/
│   ├── ultimate_voice_assistant.py  # Main application
│   ├── tts_service.py              # TTS service wrapper
│   ├── pyttsx3_tts.py              # pyttsx3 TTS implementation
│   ├── windows_tts.py              # Windows SAPI TTS
│   └── riva_tts.py                 # NVIDIA Riva TTS
├── config/
│   └── config.yaml                 # Configuration file
├── piper_voices/                   # Piper TTS voice models
├── demo_*.py                       # Demo scripts
└── requirements.txt                # Python dependencies
```

### Running Demos
The project includes several demo scripts to test individual components:
- `demo_piper_tts.py` - Test Piper TTS
- `demo_azure_tts.py` - Test Azure TTS
- `demo_openai_tts.py` - Test OpenAI TTS
- `test_pyttsx3.py` - Test pyttsx3 TTS

## 🐛 Troubleshooting

### Common Issues

**"No speech detected"**
- Check your microphone permissions
- Ensure your microphone is working and set as default
- Try speaking closer to the microphone

**"Ollama connection failed"**
- Verify Ollama is installed and running
- Check if the service is accessible at `http://127.0.0.1:11434`
- Ensure the specified model is installed: `ollama list`

**"TTS not working"**
- Windows SAPI should work by default
- For Piper TTS, ensure voice models are in `piper_voices/` directory
- Check TTS engine configuration in `config.yaml`

**"Hotkeys not responding"**
- Run as Administrator if needed
- Check if another application is capturing the same hotkeys
- Verify pynput is installed correctly

**"Import errors"**
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

### Performance Tips
- Use GPU acceleration for Whisper if available (`device: "cuda"`)
- Choose smaller models for faster response times
- Adjust `beam_size` in STT configuration for speed vs accuracy trade-off

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Ollama](https://ollama.com/) for local LLM support
- [Piper TTS](https://github.com/rhasspy/piper) for high-quality text-to-speech
- [LangChain](https://langchain.com/) for LLM conversation management
- [Rich](https://github.com/Textualize/rich) for beautiful console output

## 📞 Support

If you encounter any problems or have questions, please:
1. Check the [troubleshooting section](#🐛-troubleshooting)
2. Search existing [GitHub issues](https://github.com/yourusername/voice-assistant-windows/issues)
3. Create a new issue with detailed information about your problem

---

**Made with ❤️ for the Windows community**

## 🧭 Setup Wizard, Profiles, and User Settings

- On first run (or via `python voice-assistant/main.py --wizard`), a TTY-only wizard lets you choose:
  - Compute: GPU or CPU (auto-detected default)
  - Local LLM profile: Fast / Medium / Advanced (maps to configured models)
  - Whisper size: Fast / Medium / Advanced (tiny/base/large-v3)
- Your choices are saved to `%APPDATA%\VoiceAssistant\settings.yaml` and override `voice-assistant/config.yaml`.
- Desktop icon/PowerShell launcher runs the wizard automatically on first run when possible. Otherwise, defaults apply and you can customize later.

### Wake-word (default) and Fallback
- Default wake mode: Porcupine (hands-free). Set `PICOVOICE_ACCESS_KEY` to enable.
- If Porcupine isn’t configured or installed, the app falls back to VAD automatically.

### Quick commands
```powershell
# Run wizard explicitly
python voice-assistant\main.py --wizard

# Normal launch (Ollama by default)
.\start_voice_assistant.ps1
```
