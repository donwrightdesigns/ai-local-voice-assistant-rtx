# Voice Assistant: Local/Windows-Native Upgrade Plan

Last updated: 2025-10-13

Goals
- Keep the assistant Windows-native and reliable (no WSL required)
- Offer a simple startup selection: backend (Ollama vs Local) and mode (Faster vs Advanced)
- Improve TTS quality with practical, cost-aware options
- Preserve the existing UX: multiple input/output modes, works in any text field

A. LLM Backends: Options and Tradeoffs
1) Ollama (HTTP server)
- Pros: Simple, stable, low memory, streaming
- Cons: Requires external service, performance < CUDA transformers
- Use case: Safe fallback

2) Local Transformers (CUDA, quantized AWQ/BNB)
- Pros: Best latency/throughput on RTX 4070 Ti, uses your local models
- Cons: Requires PyTorch/CUDA stack
- Use case: Daily driver

3) llama.cpp (GGUF) via llama-cpp-python (optional later)
- Pros: Minimal footprint, very stable
- Cons: Slower than CUDA for same params
- Use case: Ultra portability

Recommended startup mappings
- Local + Faster: Mistral 7B AWQ @ J:\\models\\mistral_7b_AWQ_int4_chat
- Local + Advanced: Llama2 13B AWQ @ J:\\models\\llama2_13b_AWQ_INT4_chat
- Ollama + Faster: llama3.2:3b (existing)
- Optional: llama.cpp + GGUF later

B. TTS Options (quality vs cost)
Free/local
- Windows SAPI: zero setup, lowest quality
- Piper TTS: good quality, fast CPU, small voice downloads (RECOMMENDED next)
- Coqui XTTS v2: higher quality, GPU ~3–5GB, more setup
Paid/hosted
- ElevenLabs (best quality, trivial), Azure Neural (enterprise-grade)
Plan
- Step 1: Add Piper as selectable provider; keep SAPI fallback
- Step 2: Optionally add XTTS v2; Step 3: Optionally add ElevenLabs

C. UX Design
- Startup menu (or --select flag):
  Backend: [1] Ollama, [2] Local (Transformers), [3] Local (llama.cpp optional)
  Mode: [A] Faster (Mistral 7B), [B] Advanced (Llama2 13B)
  Accept: 1A, 2B, etc. Persist choice in config.config.yaml under runtime.*
- CLI overrides: --backend=ollama|local|llamacpp --mode=faster|advanced
- Preflight checks: GPU/VRAM, model path, TTS test beep
- Power-user: hotkey/tray action to switch backend (re-init safely)

D. Architecture Changes
- Introduce provider interfaces:
  - LLMProvider: chat(prompt, stream=True), maps common params (temperature/top_p/top_k)
  - TTSProvider: speak(text), get_voices(), set_voice()
- Concrete providers:
  - LLM: OllamaProvider, TransformersProvider, LlamaCppProvider (later)
  - TTS: WindowsSAPIProvider, PiperProvider, XTTSProvider, ElevenLabsProvider
- Keep current config keys; internal mapping per provider for portability

E. Latency & Responsiveness
- STT: current faster-whisper base.en (CPU int8). Options: small.en or GPU fp16 for more speed
- Endpointing: add optional VAD for quicker turn-around
- TTS streaming: sentence chunking for local engines; real streaming for hosted engines

F. Stability Choices
- Entire stack Windows-native (no WSL)
- Use J:\\models without duplicating caches
- Only start Ollama when chosen
- Clear logs and preflight at startup

G. Implementation Plan
Phase 1 (providers + startup menu) [1–2h]
- Add LLMProvider abstraction; wire into ConversationChain
- Implement TransformersProvider for Mistral 7B and Llama2 13B (AWQ on CUDA)
- Keep OllamaProvider as-is; add startup menu + persistence + CLI overrides

Phase 2 (TTS providers) [1–2h]
- Add TTSProvider abstraction
- Implement WindowsSAPIProvider and PiperProvider (+ voice list/test)

Phase 3 (optional) [2–3h]
- Add XTTSProvider or ElevenLabsProvider
- Add preflight checks + better errors; optional tray switcher

Defaults
- First run: Local + Faster (Transformers + Mistral 7B AWQ)
- Default TTS: Piper voice curated for clarity; SAPI fallback available
- Panic fallback: Ollama + SAPI

What you’ll notice
- Faster responses (especially with Mistral 7B on CUDA)
- Better voices with Piper/XTTS/ElevenLabs
- Simple selection at start; robust preflight; same powerful hotkeys
