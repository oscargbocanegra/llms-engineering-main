"""Reusable provider-neutral LLM functions for the laboratory."""

from __future__ import annotations

import sys
from typing import Any, Optional

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = display = None


def _value(name):
    value = globals().get(name)
    if value is not None:
        return value
    # `%run` can execute settings and helpers in different namespaces.
    # In notebooks, the loaded settings normally live in __main__.
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        value = getattr(main_module, name, None)
        if value is not None:
            return value
    try:
        from config import llm_settings
        return getattr(llm_settings, name)
    except (ImportError, AttributeError):
        return None


def _show(text, enabled=True):
    if not enabled:
        return
    if display and Markdown:
        display(Markdown(text))
    else:
        print(text)


def _chunk_text(chunk):
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    return getattr(getattr(choices[0], "delta", None), "content", None) or ""


def call_model(client, model, messages, use_stream=False, temperature=None,
               max_tokens=None, display_output=True, **kwargs):
    """Call Ollama, NVIDIA, OpenAI, or any OpenAI-compatible client."""
    if client is None:
        raise ValueError(f"No client configured for model '{model}'.")
    params = {"model": model, "messages": messages, "stream": use_stream, **kwargs}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    response = client.chat.completions.create(**params)
    if use_stream:
        result = ""
        for chunk in response:
            fragment = _chunk_text(chunk)
            if fragment:
                result += fragment
                print(fragment, end="", flush=True)
        print()
    else:
        result = getattr(response.choices[0].message, "content", None) or ""
    if display_output:
        _show(result)
    return result


def _chat(provider, system_msg, user_msg, model=None, stream=False,
          max_tokens=100, temperature=None, display_output=True, **kwargs):
    names = {
        "ollama": ("ollama_client", "OLLAMA_MODEL"),
        "nvidia": ("nvidia_client", "NVIDIA_MODEL"),
        "openai": ("openai_client", "OPENAI_MODEL"),
    }
    client_name, model_name = names[provider]
    messages = [] if not system_msg else [{"role": "system", "content": system_msg}]
    messages.append({"role": "user", "content": user_msg})
    return call_model(_value(client_name), model or _value(model_name), messages,
                      use_stream=stream, temperature=temperature,
                      max_tokens=max_tokens, display_output=display_output, **kwargs)


def call_provider(provider, system_msg, user_msg, model=None, stream=False,
                  max_tokens=100, temperature=None, display_output=True, **kwargs):
    """Common entry point for ollama, nvidia, openai, anthropic, and google."""
    key = provider.lower().replace("-", "").replace("_", "")
    if key in {"ollama", "nvidia", "nim", "openai"}:
        return _chat("nvidia" if key == "nim" else key, system_msg, user_msg,
                     model, stream, max_tokens, temperature, display_output, **kwargs)
    if key in {"anthropic", "claude"}:
        client, model = _value("claude_client"), model or _value("CLAUDE_MODEL")
        if client is None:
            raise ValueError("Anthropic client is not configured.")
        params = {"model": model, "max_tokens": max_tokens or 100,
                  "system": system_msg, "messages": [{"role": "user", "content": user_msg}], **kwargs}
        if temperature is not None:
            params["temperature"] = temperature
        if stream:
            result = ""
            with client.messages.stream(**params) as response:
                for fragment in response.text_stream:
                    result += fragment
                    print(fragment, end="", flush=True)
            print()
        else:
            response = client.messages.create(**params)
            result = "".join(getattr(block, "text", "") for block in response.content)
        if display_output:
            _show(result)
        return result
    if key in {"google", "gemini"}:
        client, model = _value("google_client"), model or _value("GOOGLE_MODEL")
        if client is None:
            raise ValueError("Google Gemini client is not configured.")
        from google.genai import types
        config = types.GenerateContentConfig(system_instruction=system_msg or None,
                                              temperature=temperature,
                                              max_output_tokens=max_tokens)
        method = client.models.generate_content_stream if stream else client.models.generate_content
        response = method(model=model, contents=user_msg, config=config)
        result = ""
        if stream:
            for chunk in response:
                fragment = getattr(chunk, "text", "") or ""
                result += fragment
                print(fragment, end="", flush=True)
            print()
        else:
            result = getattr(response, "text", "") or ""
        if display_output:
            _show(result)
        return result
    raise ValueError("Unknown provider: use ollama, nvidia, openai, anthropic, or google.")


def call_ollama(system_msg, user_msg, max_tokens=100, stream=False, **kwargs):
    return call_provider("ollama", system_msg, user_msg, max_tokens=max_tokens, stream=stream, **kwargs)


def call_nvidia(system_msg, user_msg, max_tokens=100, stream=False, **kwargs):
    return call_provider("nvidia", system_msg, user_msg, max_tokens=max_tokens, stream=stream, **kwargs)


def call_openai(system_msg, user_msg, max_tokens=100, stream=False, **kwargs):
    return call_provider("openai", system_msg, user_msg, max_tokens=max_tokens, stream=stream, **kwargs)


def call_claude(system_msg, user_msg, max_tokens=100, stream=False, **kwargs):
    return call_provider("anthropic", system_msg, user_msg, max_tokens=max_tokens, stream=stream, **kwargs)


def call_gemini(system_msg, user_msg, max_tokens=100, stream=False, **kwargs):
    return call_provider("google", system_msg, user_msg, max_tokens=max_tokens, stream=stream, **kwargs)


# Names retained for existing notebooks.
callModelOllama = call_ollama
callModelNvidia = call_nvidia


def validate_client(client, model, provider, test_connection=False):
    result = {"provider": provider, "model": model, "client_loaded": client is not None,
              "model_configured": bool(model), "connection_test": "SKIPPED",
              "status": "FAILED", "error": None}
    try:
        if client is None or not model:
            raise ValueError("Client or model is not configured.")
        if not hasattr(client, "chat") or not hasattr(client.chat.completions, "create"):
            raise TypeError("Client does not expose chat.completions.create.")
        if test_connection:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Respond only: OK"}],
                max_tokens=8, stream=False)
            if not getattr(response.choices[0].message, "content", None):
                raise RuntimeError("The model returned no content.")
            result["connection_test"] = "PASSED"
        result["status"] = "PASSED"
    except Exception as error:
        result["error"] = str(error)
    return result


def validate_loaded_clients(ollama_client=None, ollama_model=None,
                            nvidia_client=None, nvidia_model=None,
                            test_connection=False):
    return {
        "ollama": validate_client(ollama_client or _value("ollama_client"),
                                   ollama_model or _value("OLLAMA_MODEL"), "Ollama", test_connection),
        "nvidia": validate_client(nvidia_client or _value("nvidia_client"),
                                   nvidia_model or _value("NVIDIA_MODEL"), "NVIDIA NIM", test_connection),
    }
