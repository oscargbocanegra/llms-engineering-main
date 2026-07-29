import base64
import os
import anthropic

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from pathlib import Path

# ============================================================
# Carga de variables de entorno
# ============================================================

ENV_PATH = Path.home() / "work" / ".config" / "llm" / "apis.env"

if not ENV_PATH.is_file():
    raise FileNotFoundError(
        f"No existe el archivo de configuración: {ENV_PATH}"
    )

load_dotenv(ENV_PATH, override=False)


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value or not value.strip():
        raise RuntimeError(
            f"Variable requerida ausente o vacía: {name}"
        )

    return value.strip()


# ============================================================
# Ollama
# ============================================================

OLLAMA_BASE_URL = required_env("OLLAMA_BASE_URL").rstrip("/")
OLLAMA_USERNAME = required_env("OLLAMA_USERNAME")
OLLAMA_PASSWORD = required_env("OLLAMA_PASSWORD")

OLLAMA_MODEL_GEMMA = os.getenv(
    "OLLAMA_MODEL_GEMMA",
    "gemma4:12b",
).strip()

OLLAMA_MODEL_LLAMA = os.getenv(
    "OLLAMA_MODEL_LLAMA",
    "qwen3.5:9b",
).strip()


def ollama_basic_auth_header() -> str:
    credentials = (
        f"{OLLAMA_USERNAME}:{OLLAMA_PASSWORD}"
    ).encode("utf-8")

    encoded_credentials = base64.b64encode(
        credentials
    ).decode("ascii")

    return f"Basic {encoded_credentials}"


OLLAMA_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": ollama_basic_auth_header(),
}



# Cliente Ollama compatible con OpenAI
ollama_client = OpenAI(
    base_url=f"{OLLAMA_BASE_URL}/v1",
    api_key="ollama",  # Valor ficticio; Ollama no usa API key
    default_headers={
        "Authorization": ollama_basic_auth_header(),
    },
)
# ============================================================
# NVIDIA NIM
# ============================================================

NVIDIA_BASE_URL = required_env("NVIDIA_BASE_URL").rstrip("/")
NVIDIA_API_KEY = required_env("NVIDIA_API_KEY")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "z-ai/glm-5.2",
).strip()

nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
)


# ============================================================
# OpenAI
# ============================================================

OPENAI_API_KEY = required_env("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
).strip()

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)


# ============================================================
# Anthropic Claude
# ============================================================

ANTHROPIC_API_KEY = required_env("ANTHROPIC_API_KEY")

CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL",
    "claude-3-5-sonnet-latest",
).strip()

claude_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
)


# ============================================================
# Google Gemini - SDK actual
# ============================================================

GOOGLE_API_KEY = required_env("GOOGLE_API_KEY")

GOOGLE_MODEL = os.getenv(
    "GOOGLE_MODEL",
    "gemini-2.0-flash",
).strip()

google_client = genai.Client(
    api_key=GOOGLE_API_KEY,
)