import argparse
import asyncio
import sys
from pathlib import Path
from src.assistant import Assistant
from src.config_loader import load_config, apply_computed_defaults, startup_wizard, save_user_settings, is_tty


def main():
    parser = argparse.ArgumentParser(description="Local Windows Voice Assistant")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama", "vllm", "local"],
        default=None,
        help="Which LLM backend to use (default: from config)",
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Run first-time setup wizard and save preferences",
    )
    parser.add_argument(
        "--clone",
        type=str,
        metavar="VOICE_WAV",
        help="Path to a 30‑second WAV file – fine‑tunes a Kokoro voice.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    cfg, settings_path = load_config(project_root)
    cfg = apply_computed_defaults(cfg)

    if args.wizard and is_tty():
        user_overrides = startup_wizard(cfg)
        save_user_settings(user_overrides)

    # Provider selection: CLI > config
    provider = args.provider or cfg.get("llm", {}).get("provider", "ollama")
    if provider == "local":
        provider = "vllm"  # map 'local' alias to vLLM provider

    if args.clone:
        from src.kokoro_tts import KokoroTTS
        tts = KokoroTTS()
        print("Voice cloning is a work in progress – currently a placeholder.")
        return

    assistant = Assistant(provider=provider, project_root=project_root)
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        assistant.stop()


if __name__ == "__main__":
    main()
