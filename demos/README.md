# Voice Assistant Demos

This directory contains demonstration scripts and test utilities for the Voice Assistant project.

## Demo Scripts

### Text-to-Speech (TTS) Demos
- **`demo_piper_tts.py`** - Demonstrates Piper TTS functionality with different voices
- **`demo_azure_tts.py`** - Shows Azure Cognitive Services TTS integration
- **`demo_openai_tts.py`** - Tests OpenAI TTS API capabilities
- **`simple_piper_demo.py`** - Simple Piper TTS demonstration
- **`piper_wav_demo.py`** - Piper TTS with WAV file output

### Testing Utilities
- **`test_pyttsx3.py`** - Tests the pyttsx3 TTS engine
- **`test_auto_tts.py`** - Automated TTS testing across different engines
- **`list_voices.py`** - Lists available system voices

### Other Utilities
- **`try_bark_fix.py`** - Experimental fixes for Bark TTS issues

## Usage

To run any demo script, navigate to the project root directory and run:

```bash
python demos/script_name.py
```

For example:
```bash
python demos/demo_piper_tts.py
python demos/test_pyttsx3.py
```

## Requirements

Most demo scripts require the same dependencies as the main application. Some may have additional requirements:

- **Azure demos**: Require Azure Cognitive Services credentials
- **OpenAI demos**: Require OpenAI API key
- **Piper demos**: Require Piper voice models in the `piper_voices/` directory

## Configuration

Some demos may require configuration files or environment variables. Check individual scripts for specific requirements.

## Notes

- These scripts are primarily for testing and demonstration purposes
- Not all features may be available depending on your system configuration
- Some demos may require specific hardware (microphone, speakers) or API keys
- Demo scripts are not intended for production use

## Contributing

When adding new demo scripts:
1. Follow the naming convention: `demo_*.py` for demonstrations, `test_*.py` for test utilities
2. Include docstrings explaining the purpose and usage
3. Add error handling for missing dependencies or configuration
4. Update this README with a description of the new demo
