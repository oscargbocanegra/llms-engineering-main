"""Central, backward-compatible configuration for the LLM laboratory.

The module exposes:
1. A structured ``settings`` object for the new architecture.
2. Legacy module-level variables and clients used by existing notebooks.

Secrets are read from environment variables. They are never printed.
Clients are created lazily to avoid import-time network/client failures.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read and trim an environment variable."""
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = env(name)
    if raw is None:
        return default

    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name} must be boolean; received {raw!r}."
    )


def env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    raw = env(name)
    if raw is None:
        return default

    if isinstance(raw, bool):
        raise TypeError(f"{name} must be an integer, not bool.")

    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error

    if value < 1:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def _load_environment() -> Optional[Path]:
    """Load the first existing environment file without overriding OS values."""
    configured = env("LLM_ENV_FILE")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home() / "work" / ".config" / "llm" / "apis.env",
        Path("/workspace/.env"),
        Path.cwd() / ".env",
    ]

    for path in candidates:
        if path is not None and path.is_file():
            load_dotenv(path, override=False)
            return path

    return None


ENV_PATH = _load_environment()


def required_env(name: str) -> str:
    """Return a required variable or raise an explicit error."""
    value = env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration required to create and invoke one provider."""

    name: str
    model: Optional[str]
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: bool = True

    @property
    def configured(self) -> bool:
        """Whether the provider has enough configuration to be attempted."""
        if not self.enabled or not self.model:
            return False

        if self.name == "ollama":
            return bool(self.base_url)

        return bool(self.api_key)


@dataclass(frozen=True)
class LLMSettings:
    """Immutable application settings."""

    ollama: ProviderConfig
    nvidia: ProviderConfig
    openai: ProviderConfig
    anthropic: ProviderConfig
    google: ProviderConfig
    huggingface: ProviderConfig
    default_provider: str
    default_max_tokens: int
    request_timeout_seconds: int


def _build_settings() -> LLMSettings:
    """Build settings once after loading the environment."""
    ollama_base = (env("OLLAMA_BASE_URL", "http://localhost:11434") or "").rstrip("/")

    return LLMSettings(
        ollama=ProviderConfig(
            name="ollama",
            base_url=ollama_base,
            api_key=env("OLLAMA_API_KEY", "ollama"),
            username=env("OLLAMA_USERNAME"),
            password=env("OLLAMA_PASSWORD"),
            # Existing defaults are preserved for notebook compatibility.
            model=env("OLLAMA_MODEL", env("OLLAMA_MODEL_GEMMA", "gemma4:12b")),
            enabled=env_bool("OLLAMA_ENABLED", True),
        ),
        nvidia=ProviderConfig(
            name="nvidia",
            base_url=(env(
                "NVIDIA_BASE_URL",
                "https://integrate.api.nvidia.com/v1",
            ) or "").rstrip("/"),
            api_key=env("NVIDIA_API_KEY"),
            model=env("NVIDIA_MODEL", "z-ai/glm-5.2"),
            enabled=env_bool("NVIDIA_ENABLED", True),
        ),
        openai=ProviderConfig(
            name="openai",
            base_url=(env(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            ) or "").rstrip("/"),
            api_key=env("OPENAI_API_KEY"),
            model=env("OPENAI_MODEL", "gpt-4o-mini"),
            enabled=env_bool("OPENAI_ENABLED", True),
        ),
        anthropic=ProviderConfig(
            name="anthropic",
            api_key=env("ANTHROPIC_API_KEY"),
            model=env("CLAUDE_MODEL", "claude-opus-4-1-20250805"),
            enabled=env_bool("ANTHROPIC_ENABLED", True),
        ),
        google=ProviderConfig(
            name="google",
            api_key=env("GOOGLE_API_KEY"),
            # API model identifiers should be supplied through GOOGLE_MODEL.
            model=env("GOOGLE_MODEL", "gemini-2.5-pro"),
            enabled=env_bool("GOOGLE_ENABLED", True),
        ),
        huggingface=ProviderConfig(
            name="huggingface",
            api_key=env("HF_TOKEN"),
            model=env("HF_MODEL","black-forest-labs/FLUX.1-schnell",),
            enabled=env_bool("HUGGINGFACE_ENABLED",True,),
        ),
        default_provider=(env("LLM_DEFAULT_PROVIDER", "ollama") or "ollama").lower(),
        default_max_tokens=env_int("LLM_DEFAULT_MAX_TOKENS", 1000),
        request_timeout_seconds=env_int("LLM_REQUEST_TIMEOUT_SECONDS", 120),
    )

settings = _build_settings()


_PROVIDER_ALIASES = {
    "ollama": "ollama",
    "nvidia": "nvidia",
    "nim": "nvidia",
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "google",
    "gemini": "google",
    "huggingface": "huggingface",
    "hf": "huggingface",
}


def normalize_provider(provider: str) -> str:
    """Normalize aliases to the canonical provider name."""
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string.")

    key = provider.lower().replace("-", "").replace("_", "").strip()
    try:
        return _PROVIDER_ALIASES[key]
    except KeyError as error:
        raise ValueError(f"Unsupported provider: {provider}") from error


def get_provider_config(provider: str) -> ProviderConfig:
    """Return structured configuration for one provider."""
    canonical = normalize_provider(provider)
    return getattr(settings, canonical)


def get_model(provider: str) -> Optional[str]:
    """Return the configured default model for a provider."""
    return get_provider_config(provider).model


def _basic_auth_headers(
    username: Optional[str],
    password: Optional[str],
) -> dict[str, str]:
    if not username or not password:
        return {}

    token = base64.b64encode(
        f"{username}:{password}".encode("utf-8")
    ).decode("ascii")
    return {"Authorization": f"Basic {token}"}


@lru_cache(maxsize=None)
def get_client(provider: str) -> Any:
    """Create and cache a provider client lazily."""
    canonical = normalize_provider(provider)
    config = get_provider_config(canonical)

    if not config.enabled:
        raise RuntimeError(f"Provider '{canonical}' is disabled.")

    if canonical in {"ollama", "nvidia", "openai"}:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError(
                "OpenAI-compatible providers require: pip install openai"
            ) from error

        if canonical != "ollama" and not config.api_key:
            return None

        kwargs: dict[str, Any] = {
            "base_url": (
                f"{config.base_url}/v1"
                if canonical == "ollama"
                and config.base_url
                and not config.base_url.endswith("/v1")
                else config.base_url
            ),
            "api_key": config.api_key or "ollama",
            "timeout": settings.request_timeout_seconds,
        }

        if canonical == "ollama":
            headers = _basic_auth_headers(
                config.username,
                config.password,
            )
            if headers:
                kwargs["default_headers"] = headers

        return OpenAI(**kwargs)

    if canonical == "anthropic":
        if not config.api_key:
            return None
        try:
            import anthropic
        except ImportError as error:
            raise ImportError(
                "Anthropic requires: pip install anthropic"
            ) from error
        return anthropic.Anthropic(
            api_key=config.api_key,
            timeout=settings.request_timeout_seconds,
        )

    if canonical == "google":
        if not config.api_key:
            return None
        try:
            from google import genai
        except ImportError as error:
            raise ImportError(
                "Gemini requires: pip install google-genai"
            ) from error
        return genai.Client(api_key=config.api_key)

    if canonical == "huggingface":
        if not config.api_key:
            return None
    
        try:
            from huggingface_hub import InferenceClient
        except ImportError as error:
            raise ImportError(
                "Hugging Face requires: "
                "pip install huggingface_hub"
            ) from error
    
        return InferenceClient(
            api_key=config.api_key,
            timeout=settings.request_timeout_seconds,
        )

    raise AssertionError(f"Unhandled provider: {canonical}")


def clear_client_cache() -> None:
    """Clear cached clients after changing environment variables."""
    get_client.cache_clear()


# ---------------------------------------------------------------------------
# Legacy compatibility exports.
# Existing notebooks can continue importing these names.
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = settings.ollama.base_url
OLLAMA_API_KEY = settings.ollama.api_key
OLLAMA_USERNAME = settings.ollama.username
OLLAMA_PASSWORD = settings.ollama.password
OLLAMA_MODEL_GEMMA = env("OLLAMA_MODEL_GEMMA", "gemma4:12b")
OLLAMA_MODEL_LLAMA = env("OLLAMA_MODEL_LLAMA", "qwen3.5:9b")
OLLAMA_MODEL = settings.ollama.model
OLLAMA_HEADERS = _basic_auth_headers(OLLAMA_USERNAME, OLLAMA_PASSWORD)

# Audio local
BASE_URL_AUDIO = env("BASE_URL_AUDIO","http://192.168.80.14:8000",)
MODEL_KOKORO = env("MODEL_KOKORO","speaches-ai/Kokoro-82M-v1.0-ONNX-int8",)
MODEL_PIPER_CLAUDE = env("MODEL_PIPER_CLAUDE","speaches-ai/piper-es_MX-claude-high",)
MODEL_PIPEL_DANIELA = env("MODEL_PIPEL_DANIELA","it-lab/piper-es_AR-daniela-high",)


NVIDIA_BASE_URL = settings.nvidia.base_url
NVIDIA_API_KEY = settings.nvidia.api_key
NVIDIA_MODEL = settings.nvidia.model
NVIDIA_MODEL_NEMOTRON = env("NVIDIA_MODEL_NEMOTRON","nvidia/nemotron-3.5-lightning-30b-a3b")

OPENAI_API_KEY = settings.openai.api_key
OPENAI_MODEL = settings.openai.model

ANTHROPIC_API_KEY = settings.anthropic.api_key
CLAUDE_MODEL = settings.anthropic.model

GOOGLE_API_KEY = settings.google.api_key
GOOGLE_MODEL = settings.google.model

HF_TOKEN = settings.huggingface.api_key
HF_MODEL = settings.huggingface.model

def _safe_legacy_client(provider: str,) -> Any:
    """
    Return a lazily created client or None when
    optional configuration is absent.
    """
    config = get_provider_config(provider)

    if provider != "ollama" and not config.api_key:
        return None

    try:
        return get_client(provider)

    except (
        ImportError,
        RuntimeError,
        ValueError,
    ):
        return None


ollama_client = _safe_legacy_client("ollama")
nvidia_client = _safe_legacy_client("nvidia")
openai_client = _safe_legacy_client("openai")
claude_client = _safe_legacy_client("anthropic")
google_client = _safe_legacy_client("google")
huggingface_client = _safe_legacy_client("huggingface")


__all__ = [
    "ENV_PATH",
    "ProviderConfig",
    "LLMSettings",
    "settings",
    "env",
    "env_bool",
    "env_int",
    "required_env",
    "normalize_provider",
    "get_provider_config",
    "get_model",
    "get_client",
    "clear_client_cache",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_USERNAME",
    "OLLAMA_PASSWORD",
    "OLLAMA_MODEL_GEMMA",
    "OLLAMA_MODEL_LLAMA",
    "OLLAMA_MODEL",
    "OLLAMA_HEADERS",
    "NVIDIA_BASE_URL",
    "NVIDIA_API_KEY",
    "NVIDIA_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "CLAUDE_MODEL",
    "GOOGLE_API_KEY",
    "GOOGLE_MODEL",
    "HF_TOKEN",
    "HF_MODEL",
    "ollama_client",
    "nvidia_client",
    "openai_client",
    "claude_client",
    "google_client",
    "huggingface_client",
    "BASE_URL_AUDIO",
    "MODEL_KOKORO",
    "MODEL_PIPER_CLAUDE",
    "MODEL_PIPEL_DANIELA",
]
