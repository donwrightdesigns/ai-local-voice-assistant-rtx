#!/usr/bin/env python3
"""
LLM provider abstractions for the Windows voice assistant.
Phase 1 goals:
- Keep current Ollama integration but add a small preflight check and a clean factory
- Prepare a placeholder for local Transformers provider (to be implemented in Phase 2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import requests

try:
    from langchain_community.llms import Ollama as LangchainOllama
except Exception:
    # LangChain is expected to be installed already per project requirements
    LangchainOllama = None  # type: ignore


@dataclass
class ProviderResult:
    ok: bool
    message: str


class LLMProvider:
    """Abstract provider interface."""

    def preflight(self) -> ProviderResult:
        raise NotImplementedError

    def build_langchain_llm(self):
        """Return a LangChain-compatible LLM if applicable."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Ollama provider using the existing LangChain integration."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def preflight(self) -> ProviderResult:
        """Lightweight connectivity test and friendly guidance if not running."""
        try:
            # /api/tags is a cheap endpoint; if it 200s, the daemon is up
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return ProviderResult(True, "Ollama is reachable")
            return ProviderResult(False, f"Ollama responded with HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            return ProviderResult(False, f"Cannot reach Ollama at {self.base_url}: {e}")

    def build_langchain_llm(self):
        if LangchainOllama is None:
            raise RuntimeError("LangChain Ollama integration not available")
        return LangchainOllama(model=self.model, base_url=self.base_url)


class TransformersProvider(LLMProvider):
    """
    Local Transformers CUDA provider for quantized (AWQ) models on Windows.
    - Faster: Mistral 7B AWQ
    - Advanced: Llama2 13B AWQ
    """

    def __init__(self, model_path: str, mode: str = "faster"):
        self.model_path = (model_path or "").strip()
        self.mode = mode

    def preflight(self) -> ProviderResult:
        import os
        try:
            import torch  # noqa: F401
        except Exception as e:
            return ProviderResult(False, f"PyTorch not available: {e}")
        if not self.model_path or not os.path.exists(self.model_path):
            return ProviderResult(False, f"Local model path not found: {self.model_path}")
        # Prefer CUDA but do not hard fail if missing
        try:
            import torch
            if not torch.cuda.is_available():
                return ProviderResult(True, "CUDA not available - will run on CPU (slow)")
        except Exception:
            pass
        # Check autoawq presence
        try:
            import autoawq  # noqa: F401
        except Exception:
            return ProviderResult(False, "autoawq not installed. Install with: pip install autoawq")
        return ProviderResult(True, "Local Transformers preflight OK")

    def build_langchain_llm(self):
        import torch
        from transformers import AutoTokenizer, pipeline
        from langchain_community.llms import HuggingFacePipeline
        # Load AWQ quantized model if available
        try:
            from autoawq import AutoAWQForCausalLM
        except Exception as e:
            raise RuntimeError(
                f"autoawq is required for local quantized models: {e}. Install with 'pip install autoawq'"
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        # Use fused layers for speed where supported
        model = AutoAWQForCausalLM.from_quantized(
            self.model_path,
            fuse_layers=True,
            trust_remote_code=True,
            device_map="auto",
        )
        gen_pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_new_tokens=512,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        return HuggingFacePipeline(pipeline=gen_pipe)


def make_provider(backend: str, *, base_url: Optional[str] = None, model: Optional[str] = None,
                  model_path: Optional[str] = None, mode: str = "faster") -> LLMProvider:
    backend = (backend or "").strip().lower()
    if backend in ("ollama", "1", "1a", "1b"):
        if not base_url or not model:
            raise ValueError("Ollama provider requires base_url and model")
        return OllamaProvider(base_url=base_url, model=model)
    elif backend in ("local", "transformers", "2", "2a", "2b"):
        return TransformersProvider(model_path=model_path or "", mode=mode)
    else:
        # Default to Ollama if unknown
        if not base_url or not model:
            raise ValueError("Unknown backend and missing Ollama config; cannot default")
        return OllamaProvider(base_url=base_url, model=model)