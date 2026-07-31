"""Reusable provider-neutral LLM functions for notebooks, Gradio, and services."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Optional

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = None
    display = None


ProviderName = Literal[
    "ollama", "nvidia", "nim", "openai",
    "anthropic", "claude", "google", "gemini"
]
Message = Mapping[str, str]
TokenCallback = Callable[[str], None]

_OPENAI_COMPATIBLE_PROVIDERS: dict[str, tuple[str, str]] = {
    "ollama": ("ollama_client", "OLLAMA_MODEL"),
    "nvidia": ("nvidia_client", "NVIDIA_MODEL"),
    "openai": ("openai_client", "OPENAI_MODEL"),
}

_PROVIDER_ALIASES: dict[str, str] = {
    "ollama": "ollama",
    "nvidia": "nvidia",
    "nim": "nvidia",
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "google",
    "gemini": "google",
}


def _value(name: str) -> Any:
    value = globals().get(name)
    if value is not None:
        return value

    main_module = sys.modules.get("__main__")
    if main_module is not None:
        value = getattr(main_module, name, None)
        if value is not None:
            return value

    try:
        from config import llm_settings
    except ImportError:
        return None

    return getattr(llm_settings, name, None)


def _normalize_provider(provider: str) -> str:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Provider must be a non-empty string.")

    key = provider.lower().replace("-", "").replace("_", "").strip()
    try:
        return _PROVIDER_ALIASES[key]
    except KeyError as error:
        supported = ", ".join(sorted(_PROVIDER_ALIASES))
        raise ValueError(
            f"Unknown provider '{provider}'. Supported values: {supported}."
        ) from error


def _normalize_max_tokens(
    max_tokens: Optional[int],
    default: int = 100,
) -> int:
    value = default if max_tokens is None else max_tokens

    if isinstance(value, bool):
        raise TypeError("max_tokens must be an integer, not bool.")

    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError("max_tokens must be an integer.") from error

    if normalized < 1:
        raise ValueError("max_tokens must be greater than zero.")

    return normalized


def _normalize_temperature(
    temperature: Optional[float],
) -> Optional[float]:
    if temperature is None:
        return None

    if isinstance(temperature, bool):
        raise TypeError("temperature must be numeric, not bool.")

    try:
        normalized = float(temperature)
    except (TypeError, ValueError) as error:
        raise TypeError("temperature must be numeric.") from error

    if not 0.0 <= normalized <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0.")

    return normalized


def _show(text: str, enabled: bool = True) -> None:
    if not enabled:
        return

    if display is not None and Markdown is not None:
        display(Markdown(text))
    else:
        print(text)


def _emit(fragment: str, callback: Optional[TokenCallback]) -> None:
    if fragment and callback is not None:
        callback(fragment)


def _extract_openai_chunk_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""

    delta = getattr(choices[0], "delta", None)
    return getattr(delta, "content", None) or ""


def _extract_openai_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("The model returned no choices.")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) or ""
    return content if isinstance(content, str) else str(content)


def _build_messages(
    system_msg: Optional[str],
    user_msg: str,
) -> list[dict[str, str]]:
    if not isinstance(user_msg, str) or not user_msg.strip():
        raise ValueError("user_msg must be a non-empty string.")

    messages: list[dict[str, str]] = []

    if system_msg and system_msg.strip():
        messages.append(
            {"role": "system", "content": system_msg.strip()}
        )

    messages.append(
        {"role": "user", "content": user_msg.strip()}
    )
    return messages


def call_model(
    client: Any,
    model: str,
    messages: Sequence[Message],
    *,
    use_stream: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    display_output: bool = False,
    on_token: Optional[TokenCallback] = None,
    **kwargs: Any,
) -> str:
    """Call an OpenAI-compatible chat-completions client."""
    if client is None:
        raise ValueError(f"No client configured for model '{model}'.")

    if not model:
        raise ValueError("A model name must be configured.")

    params: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": bool(use_stream),
        **kwargs,
    }

    normalized_temperature = _normalize_temperature(temperature)
    if normalized_temperature is not None:
        params["temperature"] = normalized_temperature

    if max_tokens is not None:
        params["max_tokens"] = _normalize_max_tokens(max_tokens)

    response = client.chat.completions.create(**params)

    if use_stream:
        fragments: list[str] = []

        for chunk in response:
            fragment = _extract_openai_chunk_text(chunk)
            if fragment:
                fragments.append(fragment)
                _emit(fragment, on_token)

        result = "".join(fragments)
    else:
        result = _extract_openai_response_text(response)

    if display_output:
        _show(result)

    return result


def _call_openai_compatible(
    provider: str,
    system_msg: Optional[str],
    user_msg: str,
    *,
    model: Optional[str] = None,
    stream: bool = False,
    max_tokens: Optional[int] = 100,
    temperature: Optional[float] = None,
    display_output: bool = False,
    on_token: Optional[TokenCallback] = None,
    **kwargs: Any,
) -> str:
    client_name, model_name = _OPENAI_COMPATIBLE_PROVIDERS[provider]

    return call_model(
        client=_value(client_name),
        model=model or _value(model_name),
        messages=_build_messages(system_msg, user_msg),
        use_stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        display_output=display_output,
        on_token=on_token,
        **kwargs,
    )


def _call_anthropic(
    system_msg: Optional[str],
    user_msg: str,
    *,
    model: Optional[str] = None,
    stream: bool = False,
    max_tokens: Optional[int] = 100,
    temperature: Optional[float] = None,
    display_output: bool = False,
    on_token: Optional[TokenCallback] = None,
    **kwargs: Any,
) -> str:
    client = _value("claude_client")
    resolved_model = model or _value("CLAUDE_MODEL")

    if client is None:
        raise ValueError("Anthropic client is not configured.")
    if not resolved_model:
        raise ValueError("Anthropic model is not configured.")
    if not isinstance(user_msg, str) or not user_msg.strip():
        raise ValueError("user_msg must be a non-empty string.")

    params: dict[str, Any] = {
        "model": resolved_model,
        "max_tokens": _normalize_max_tokens(max_tokens),
        "messages": [
            {"role": "user", "content": user_msg.strip()}
        ],
        **kwargs,
    }

    if system_msg and system_msg.strip():
        params["system"] = system_msg.strip()

    normalized_temperature = _normalize_temperature(temperature)
    if normalized_temperature is not None:
        params["temperature"] = normalized_temperature

    if stream:
        fragments: list[str] = []

        with client.messages.stream(**params) as response:
            for fragment in response.text_stream:
                if fragment:
                    fragments.append(fragment)
                    _emit(fragment, on_token)

        result = "".join(fragments)
    else:
        response = client.messages.create(**params)
        blocks = getattr(response, "content", None) or []
        result = "".join(
            getattr(block, "text", "") or ""
            for block in blocks
        )

    if display_output:
        _show(result)

    return result


def _call_google(
    system_msg: Optional[str],
    user_msg: str,
    *,
    model: Optional[str] = None,
    stream: bool = False,
    max_tokens: Optional[int] = 100,
    temperature: Optional[float] = None,
    display_output: bool = False,
    on_token: Optional[TokenCallback] = None,
    **kwargs: Any,
) -> str:
    client = _value("google_client")
    resolved_model = model or _value("GOOGLE_MODEL")

    if client is None:
        raise ValueError("Google Gemini client is not configured.")
    if not resolved_model:
        raise ValueError("Google Gemini model is not configured.")
    if not isinstance(user_msg, str) or not user_msg.strip():
        raise ValueError("user_msg must be a non-empty string.")

    try:
        from google.genai import types
    except ImportError as error:
        raise ImportError(
            "Google Gemini support requires the 'google-genai' package."
        ) from error

    config_params: dict[str, Any] = {
        "max_output_tokens": _normalize_max_tokens(max_tokens),
        **kwargs,
    }

    if system_msg and system_msg.strip():
        config_params["system_instruction"] = system_msg.strip()

    normalized_temperature = _normalize_temperature(temperature)
    if normalized_temperature is not None:
        config_params["temperature"] = normalized_temperature

    config = types.GenerateContentConfig(**config_params)

    if stream:
        response = client.models.generate_content_stream(
            model=resolved_model,
            contents=user_msg.strip(),
            config=config,
        )

        fragments: list[str] = []
        for chunk in response:
            fragment = getattr(chunk, "text", "") or ""
            if fragment:
                fragments.append(fragment)
                _emit(fragment, on_token)

        result = "".join(fragments)
    else:
        response = client.models.generate_content(
            model=resolved_model,
            contents=user_msg.strip(),
            config=config,
        )
        result = getattr(response, "text", "") or ""

    if display_output:
        _show(result)

    return result


def call_provider(
    provider: ProviderName | str,
    system_msg: Optional[str],
    user_msg: str,
    *,
    model: Optional[str] = None,
    stream: bool = False,
    max_tokens: Optional[int] = 100,
    temperature: Optional[float] = None,
    display_output: bool = False,
    on_token: Optional[TokenCallback] = None,
    **kwargs: Any,
) -> str:
    """Provider-neutral entry point."""
    normalized_provider = _normalize_provider(provider)

    if normalized_provider in _OPENAI_COMPATIBLE_PROVIDERS:
        return _call_openai_compatible(
            normalized_provider,
            system_msg,
            user_msg,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            display_output=display_output,
            on_token=on_token,
            **kwargs,
        )

    if normalized_provider == "anthropic":
        return _call_anthropic(
            system_msg,
            user_msg,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            display_output=display_output,
            on_token=on_token,
            **kwargs,
        )

    if normalized_provider == "google":
        return _call_google(
            system_msg,
            user_msg,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            display_output=display_output,
            on_token=on_token,
            **kwargs,
        )

    raise AssertionError(
        f"Unhandled normalized provider: {normalized_provider}"
    )


def call_ollama(
    system_msg: Optional[str],
    user_msg: str,
    max_tokens: Optional[int] = 100,
    stream: bool = False,
    **kwargs: Any,
) -> str:
    return call_provider(
        "ollama",
        system_msg,
        user_msg,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )


def call_nvidia(
    system_msg: Optional[str],
    user_msg: str,
    max_tokens: Optional[int] = 100,
    stream: bool = False,
    **kwargs: Any,
) -> str:
    return call_provider(
        "nvidia",
        system_msg,
        user_msg,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )


def call_openai(
    system_msg: Optional[str],
    user_msg: str,
    max_tokens: Optional[int] = 100,
    stream: bool = False,
    **kwargs: Any,
) -> str:
    return call_provider(
        "openai",
        system_msg,
        user_msg,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )


def call_claude(
    system_msg: Optional[str],
    user_msg: str,
    max_tokens: Optional[int] = 100,
    stream: bool = False,
    **kwargs: Any,
) -> str:
    return call_provider(
        "anthropic",
        system_msg,
        user_msg,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )


def call_gemini(
    system_msg: Optional[str],
    user_msg: str,
    max_tokens: Optional[int] = 100,
    stream: bool = False,
    **kwargs: Any,
) -> str:
    return call_provider(
        "google",
        system_msg,
        user_msg,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )


callModelOllama = call_ollama
callModelNvidia = call_nvidia


def responder_llm(
    prompt: str,
    provider: str = "ollama",
    max_tokens: int = 1000,
    temperature: float = 0.7,
    *,
    system_message: str = (
        "Eres un asistente útil, preciso y claro. "
        "Responde siempre en español."
    ),
    model: Optional[str] = None,
    expose_errors: bool = True,
) -> str:
    """Gradio-compatible provider-neutral responder."""
    prompt = (prompt or "").strip()

    if not prompt:
        return "Escribe un mensaje antes de enviar."

    try:
        result = call_provider(
            provider=provider,
            system_msg=system_message,
            user_msg=prompt,
            model=model,
            max_tokens=_normalize_max_tokens(
                max_tokens,
                default=1000,
            ),
            temperature=_normalize_temperature(temperature),
            stream=False,
            display_output=False,
        )

        result = result.strip()

        if not result:
            return (
                f"El proveedor '{provider}' devolvió "
                "una respuesta vacía."
            )

        return result

    except Exception as error:
        if expose_errors:
            return (
                f"Error al consultar '{provider}': "
                f"{type(error).__name__}: {error}"
            )

        return "No fue posible procesar la solicitud."


def make_gradio_responder(
    provider: str,
    *,
    system_message: str = (
        "Eres un asistente útil, preciso y claro. "
        "Responde siempre en español."
    ),
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    expose_errors: bool = True,
) -> Callable[[str], str]:
    """Create a one-input responder for gr.Interface."""
    normalized_provider = _normalize_provider(provider)
    normalized_max_tokens = _normalize_max_tokens(
        max_tokens,
        default=1000,
    )
    normalized_temperature = _normalize_temperature(temperature)

    def responder(prompt: str) -> str:
        return responder_llm(
            prompt=prompt,
            provider=normalized_provider,
            max_tokens=normalized_max_tokens,
            temperature=(
                0.7
                if normalized_temperature is None
                else normalized_temperature
            ),
            system_message=system_message,
            model=model,
            expose_errors=expose_errors,
        )

    return responder


def validate_client(
    client: Any,
    model: Optional[str],
    provider: str,
    *,
    test_connection: bool = False,
) -> dict[str, Any]:
    normalized_provider = _normalize_provider(provider)

    result: dict[str, Any] = {
        "provider": normalized_provider,
        "model": model,
        "client_loaded": client is not None,
        "model_configured": bool(model),
        "connection_test": "SKIPPED",
        "status": "FAILED",
        "error": None,
    }

    try:
        if client is None:
            raise ValueError("Client is not configured.")
        if not model:
            raise ValueError("Model is not configured.")

        if normalized_provider in _OPENAI_COMPATIBLE_PROVIDERS:
            if (
                not hasattr(client, "chat")
                or not hasattr(client.chat, "completions")
                or not hasattr(
                    client.chat.completions,
                    "create",
                )
            ):
                raise TypeError(
                    "Client does not expose "
                    "chat.completions.create."
                )

            if test_connection:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Respond only with: OK",
                        }
                    ],
                    max_tokens=8,
                    stream=False,
                )
                if not _extract_openai_response_text(
                    response
                ).strip():
                    raise RuntimeError(
                        "The model returned no content."
                    )

        elif normalized_provider == "anthropic":
            if (
                not hasattr(client, "messages")
                or not hasattr(client.messages, "create")
            ):
                raise TypeError(
                    "Client does not expose messages.create."
                )

            if test_connection:
                response = client.messages.create(
                    model=model,
                    max_tokens=8,
                    messages=[
                        {
                            "role": "user",
                            "content": "Respond only with: OK",
                        }
                    ],
                )
                text = "".join(
                    getattr(block, "text", "") or ""
                    for block in (
                        getattr(response, "content", None)
                        or []
                    )
                )
                if not text.strip():
                    raise RuntimeError(
                        "The model returned no content."
                    )

        elif normalized_provider == "google":
            if (
                not hasattr(client, "models")
                or not hasattr(
                    client.models,
                    "generate_content",
                )
            ):
                raise TypeError(
                    "Client does not expose "
                    "models.generate_content."
                )

            if test_connection:
                response = client.models.generate_content(
                    model=model,
                    contents="Respond only with: OK",
                )
                if not (
                    getattr(response, "text", "") or ""
                ).strip():
                    raise RuntimeError(
                        "The model returned no content."
                    )

        result["connection_test"] = (
            "PASSED" if test_connection else "SKIPPED"
        )
        result["status"] = "PASSED"

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def validate_loaded_clients(
    *,
    test_connection: bool = False,
) -> dict[str, dict[str, Any]]:
    configurations = {
        "ollama": ("ollama_client", "OLLAMA_MODEL"),
        "nvidia": ("nvidia_client", "NVIDIA_MODEL"),
        "openai": ("openai_client", "OPENAI_MODEL"),
        "anthropic": ("claude_client", "CLAUDE_MODEL"),
        "google": ("google_client", "GOOGLE_MODEL"),
    }

    return {
        provider: validate_client(
            client=_value(client_name),
            model=_value(model_name),
            provider=provider,
            test_connection=test_connection,
        )
        for provider, (
            client_name,
            model_name,
        ) in configurations.items()
    }


def responder_stream(
    prompt: str,
    provider: str,
    max_tokens: int,
    temperature: float,
    system_message: str,
):
    prompt = (prompt or "").strip()
    provider = (provider or "").strip().lower()

    if not prompt:
        yield "Escribe un mensaje antes de enviar."
        return

    try:
        if isinstance(max_tokens, bool):
            raise TypeError(
                "max_tokens no puede ser booleano."
            )

        if isinstance(temperature, bool):
            raise TypeError(
                "temperature no puede ser booleano."
            )

        request = LLMRequest(
            user_message=prompt,
            system_message=system_message,
            provider=provider,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )

        yield from stream_provider(
            request,
            cumulative=True,
        )

    except Exception as error:
        yield (
            f"Error al consultar '{provider}': "
            f"{type(error).__name__}: {error}"
        )





__all__ = [
    "call_model",
    "call_provider",
    "call_ollama",
    "call_nvidia",
    "call_openai",
    "call_claude",
    "call_gemini",
    "callModelOllama",
    "callModelNvidia",
    "responder_llm",
    "make_gradio_responder",
    "validate_client",
    "validate_loaded_clients",
]
