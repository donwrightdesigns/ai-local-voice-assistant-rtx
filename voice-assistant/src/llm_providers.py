import asyncio
import aiohttp
from typing import AsyncGenerator, List, Literal, Protocol, Any
import os

# ------------------
# Base provider interface
# ------------------
class LLMProvider(Protocol):
    async def stream(self, prompt: str) -> AsyncGenerator[str, None]: ...

# ------------------
# OpenAI provider
# ------------------
class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.AsyncClient()  # OpenAI 1.0.0+ (async)
        self.model = model

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        async for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            content = chunk.choices[0].delta.content or ""
            yield content

# ------------------
# Ollama provider
# ------------------
class OllamaProvider:
    def __init__(self, model: str = "llama3:instruct", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": True},
            ) as resp:
                async for line in resp.content:
                    if line:
                        obj = line.decode()
                        # Ollama streams JSONL lines; each line contains {"role":"assistant","content":"..."}
                        data = obj.strip().split("\n")
                        for d in data:
                            try:
                                import json
                                part = json.loads(d)
                                yield part.get("content", "")
                            except Exception:
                                pass

# ------------------
# vLLM provider
# ------------------
class VLLMProvider:
    def __init__(self, endpoint: str = "http://localhost:8000/v1"):
        self.endpoint = endpoint

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{self.endpoint}/chat/completions",
                json={"model": "llama3", "messages": [{"role": "user", "content": prompt}], "stream": True},
            ) as resp:
                async for line in resp.content:
                    if line:
                        data = line.decode()
                        # vLLM streams raw text chunks
                        yield data.strip()
