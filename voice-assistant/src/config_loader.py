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

def get_gpu_info() -> tuple[bool, str, int]:
    """Returns (has_gpu, model_name, vram_mb)"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "", 0
        device = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  # MB
        return True, device, vram
    except Exception:
        return False, "", 0

def apply_computed_defaults(cfg: Dict[str, Any], cli_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Fill defaults, auto-detect device/profile, apply CLI overrides.
    Merge chain: repo defaults → user settings → CLI overrides
    """
    merged = deep_merge(DEFAULTS, cfg)
    
    # Auto-detect GPU
    has_gpu_val, gpu_model, vram_mb = get_gpu_info()
    device_value = "cuda" if has_gpu_val else "cpu"
    
    # Resolve device auto -> gpu/cpu
    for key in ("stt", "tts"):
        section = merged.get(key, {})
        dev = str(section.get("device", "auto")).lower()
        if dev == "auto":
            section["device"] = device_value
            merged[key] = section
    
    # Profile constraints based on VRAM
    if has_gpu_val and vram_mb < 6000:  # < 6 GB
        # Force fast profile
        merged.setdefault("stt", {})["profile"] = "fast"
        merged.setdefault("llm", {}).setdefault("ollama", {})["active_profile"] = "fast"
    
    # Apply CLI overrides (highest precedence)
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)
    
    # Store computed values for banner
    merged["_system"] = {
        "has_gpu": has_gpu_val,
        "gpu_model": gpu_model,
        "vram_mb": vram_mb,
        "device": device_value,
    }
    
    return merged

def save_user_settings(settings: Dict[str, Any]):
    path = user_settings_path()
    # Don't save _system computed values
    clean = {k: v for k, v in settings.items() if not k.startswith("_")}
    dump_yaml(path, clean)
