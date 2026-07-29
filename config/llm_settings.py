"""Shared, optional configuration for the LLM laboratory."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
except ImportError:
    genai = None


def env(name: str, default=None):
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def _load_environment():
    configured = env("LLM_ENV_FILE")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home() / "work" / ".config" / "llm" / "apis.env",
        Path("/workspace/.env"),
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if path and path.is_file():
            load_dotenv(path, override=False)
            return path
    return None


ENV_PATH = _load_environment()


def required_env(name: str) -> str:
    value = env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _client(base_url, api_key, headers=None):
    if OpenAI is None:
        raise ImportError("Install the dependency: pip install openai")
    kwargs = {"base_url": base_url, "api_key": api_key}
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def _ollama_headers():
    username, password = env("OLLAMA_USERNAME"), env("OLLAMA_PASSWORD")
    if not username or not password:
        return {}
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# Ollama: local or Ollama Cloud, through the OpenAI-compatible endpoint.
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_API_KEY = env("OLLAMA_API_KEY", "ollama")
OLLAMA_USERNAME = env("OLLAMA_USERNAME")
OLLAMA_PASSWORD = env("OLLAMA_PASSWORD")
OLLAMA_MODEL_GEMMA = env("OLLAMA_MODEL_GEMMA", "gemma4:12b")
OLLAMA_MODEL_LLAMA = env("OLLAMA_MODEL_LLAMA", "qwen3.5:9b")
OLLAMA_MODEL = env("OLLAMA_MODEL", OLLAMA_MODEL_GEMMA)
OLLAMA_HEADERS = _ollama_headers()
ollama_client = _client(f"{OLLAMA_BASE_URL}/v1", OLLAMA_API_KEY, OLLAMA_HEADERS)


# NVIDIA NIM: OpenAI-compatible endpoint. It is optional in the lab.
NVIDIA_BASE_URL = env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NVIDIA_API_KEY = env("NVIDIA_API_KEY")
NVIDIA_MODEL = env("NVIDIA_MODEL", "z-ai/glm-5.2")
nvidia_client = _client(NVIDIA_BASE_URL, NVIDIA_API_KEY) if NVIDIA_API_KEY else None


# OpenAI.
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4o-mini")
openai_client = _client("https://api.openai.com/v1", OPENAI_API_KEY) if OPENAI_API_KEY else None


# Anthropic Claude.
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY")
CLAUDE_MODEL = env("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
claude_client = (
    anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    if ANTHROPIC_API_KEY and anthropic is not None
    else None
)


# Google Gemini using the current google-genai SDK.
GOOGLE_API_KEY = env("GOOGLE_API_KEY")
GOOGLE_MODEL = env("GOOGLE_MODEL", "gemini-2.0-flash")
google_client = (
    genai.Client(api_key=GOOGLE_API_KEY)
    if GOOGLE_API_KEY and genai is not None
    else None
)
