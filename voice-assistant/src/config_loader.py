import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    import yaml
except Exception:
    yaml = None

def appdata_dir() -> Path:
    # Prefer %APPDATA% on Windows; fallback to user home
    base = os.environ.get("APPDATA") or str(Path.home() / ".voice_assistant")
    p = Path(base) / "VoiceAssistant"
    p.mkdir(parents=True, exist_ok=True)
    return p

def base_config_path(project_root: Path) -> Path:
    return project_root / "voice-assistant" / "config.yaml"

def user_settings_path() -> Path:
    return appdata_dir() / "settings.yaml"

def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(path: Path, data: Dict[str, Any]):
    if yaml is None:
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(project_root: Path) -> Tuple[Dict[str, Any], Path]:
    base = load_yaml(base_config_path(project_root))
    user = load_yaml(user_settings_path())
    merged = deep_merge(base, user)
    return merged, user_settings_path()

def has_gpu() -> bool:
    try:
        import torch
        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False

def is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

DEFAULTS = {
    "llm": {"ollama": {"active_profile": "fast"}},
    "stt": {"profile": "medium", "device": "auto"},
    "tts": {"device": "auto"},
    "wake_mode": "porcupine",
}

def apply_computed_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Fill minimal defaults
    merged = deep_merge(DEFAULTS, cfg)
    # Resolve device auto -> gpu/cpu
    compute = "gpu" if has_gpu() else "cpu"
    for key in ("stt", "tts"):
        section = merged.get(key, {})
        dev = str(section.get("device", "auto")).lower()
        if dev == "auto":
            section["device"] = "cuda" if compute == "gpu" else "cpu"
            merged[key] = section
    return merged

def save_user_settings(settings: Dict[str, Any]):
    path = user_settings_path()
    dump_yaml(path, settings)

def startup_wizard(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Only interactive choices; return minimal user overrides
    user_overrides: Dict[str, Any] = {}
    print("\n=== Voice Assistant – First-time setup ===")
    # Compute target
    compute_default = "gpu" if has_gpu() else "cpu"
    compute = input(f"Compute target [gpu/cpu] (default: {compute_default}): ").strip().lower() if is_tty() else compute_default
    if compute not in ("gpu", "cpu"):
        compute = compute_default
    # Map to device values
    dev_value = "cuda" if compute == "gpu" else "cpu"
    user_overrides.setdefault("stt", {})["device"] = dev_value
    user_overrides.setdefault("tts", {})["device"] = dev_value

    # LLM profile (ollama)
    llm_profiles = cfg.get("llm", {}).get("ollama", {}).get("profiles", {})
    llm_default = cfg.get("llm", {}).get("ollama", {}).get("active_profile", "fast")
    print(f"LLM profile options: {', '.join(llm_profiles.keys()) or 'fast, medium, advanced'}")
    llm_choice = input(f"Local LLM profile [fast/medium/advanced] (default: {llm_default}): ").strip().lower() if is_tty() else llm_default
    if llm_choice not in ("fast", "medium", "advanced"):
        llm_choice = llm_default
    user_overrides.setdefault("llm", {}).setdefault("ollama", {})["active_profile"] = llm_choice

    # STT profile (whisper)
    stt_default = cfg.get("stt", {}).get("profile", "medium")
    stt_choice = input(f"Whisper size [fast/medium/advanced] (default: {stt_default}): ").strip().lower() if is_tty() else stt_default
    if stt_choice not in ("fast", "medium", "advanced"):
        stt_choice = stt_default
    user_overrides.setdefault("stt", {})["profile"] = stt_choice

    print("\nSettings saved. You can change these later in the settings file.")
    return user_overrides
