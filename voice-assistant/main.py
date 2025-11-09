import argparse
import asyncio
from pathlib import Path
from src.assistant import Assistant
from src.config_loader import load_config, apply_computed_defaults, deep_merge


def parse_cli_overrides(args) -> dict:
    """Parse CLI args into config overrides."""
    overrides = {}
    
    if args.device:
        if args.device not in ("cpu", "gpu"):
            print(f"Invalid device '{args.device}'. Use 'cpu' or 'gpu'.")
            exit(1)
        dev_value = "cuda" if args.device == "gpu" else "cpu"
        overrides["stt"] = {"device": dev_value}
        overrides["tts"] = {"device": dev_value}
    
    if args.profile:
        if args.profile not in ("fast", "advanced"):
            print(f"Invalid profile '{args.profile}'. Use 'fast' or 'advanced'.")
            exit(1)
        overrides["stt"] = overrides.get("stt", {}) | {"profile": args.profile}
        overrides["llm"] = {"ollama": {"active_profile": args.profile}}
    
    if args.keywords:
        overrides["porcupine"] = {"keywords": args.keywords.split(",")}
    
    if args.provider:
        overrides["llm"] = overrides.get("llm", {}) | {"provider": args.provider}
    
    return overrides


def main():
    parser = argparse.ArgumentParser(description="RTX Hands-Free Assistant - Local, open-source voice assistant")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default=None,
        help="Compute device (default: auto-detect GPU if available)",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "advanced"],
        default=None,
        help="LLM/Whisper size (default: fast; advanced available if VRAM > 12GB)",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="Wake words comma-separated, e.g., 'computer,jarvis' (Porcupine only)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama", "vllm"],
        default=None,
        help="LLM backend (default: ollama)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    cfg, _ = load_config(project_root)
    
    # Parse CLI overrides
    cli_overrides = parse_cli_overrides(args)
    
    # Apply auto-detect + overrides
    cfg = apply_computed_defaults(cfg, cli_overrides)

    # Provider selection: CLI > config
    provider = cfg.get("llm", {}).get("provider", "ollama")
    if provider == "local":
        provider = "vllm"

    assistant = Assistant(provider=provider, project_root=project_root, config=cfg)
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        assistant.stop()


if __name__ == "__main__":
    main()
