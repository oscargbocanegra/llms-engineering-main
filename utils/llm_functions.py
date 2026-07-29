from IPython.display import display, Markdown


def call_model(
    client,
    model,
    messages,
    use_stream=False,
    temperature=None,
    max_tokens=None,
    **kwargs,
):
    """
    Ejecuta una llamada contra Ollama, NVIDIA NIM u otro cliente
    compatible con OpenAI.

    Si temperature o max_tokens son None, no se envían al proveedor
    y se utilizan sus valores predeterminados.
    """

    request_params = {
        "model": model,
        "messages": messages,
        "stream": use_stream,
        **kwargs,
    }

    if temperature is not None:
        request_params["temperature"] = temperature

    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**request_params)

    if use_stream:
        response_text = ""
        display_handle = None

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""

            if not content:
                continue

            response_text += content

            if display_handle is None:
                display_handle = display(
                    Markdown(response_text),
                    display_id=True,
                )
            else:
                display_handle.update(
                    Markdown(response_text)
                )

        return response_text

    result = completion.choices[0].message.content or ""
    display(Markdown(result))

    return result




def callModelOllama(
    messages,
    use_stream=False,
    model=None,
    temperature=None,
    max_tokens=None,
    **kwargs,
):
    return call_model(
        client=ollama_client,
        model=model or OLLAMA_MODEL_LLAMA,
        messages=messages,
        use_stream=use_stream,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def callModelNvidia(
    messages,
    use_stream=False,
    model=None,
    temperature=None,
    max_tokens=None,
    **kwargs,
):
    return call_model(
        client=nvidia_client,
        model=model or NVIDIA_MODEL,
        messages=messages,
        use_stream=use_stream,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


# =================================================================================
# Validacion de clientes
# =================================================================================

def validate_client(
    client,
    model,
    provider,
    test_connection=False,
):
    """
    Valida la configuración de un cliente compatible con OpenAI.

    test_connection=True ejecuta una llamada mínima al modelo.
    """

    result = {
        "provider": provider,
        "model": model,
        "client_loaded": client is not None,
        "model_configured": bool(model),
        "connection_test": None,
        "status": "FAILED",
        "error": None,
    }

    try:
        if client is None:
            raise ValueError("El cliente no está cargado.")

        if not model:
            raise ValueError("El modelo no está configurado.")

        if not hasattr(client, "chat") or not hasattr(
            client.chat.completions,
            "create",
        ):
            raise TypeError(
                "El cliente no expone chat.completions.create."
            )

        result["connection_test"] = "SKIPPED"

        if test_connection:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Responde únicamente: OK",
                    }
                ],
                max_tokens=8,
                stream=False,
            )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "El modelo no devolvió contenido."
                )

            result["connection_test"] = "PASSED"

        result["status"] = "PASSED"

    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = str(error)

    return result

def validate_loaded_clients(test_connection=False):
    """
    Valida los clientes configurados para Ollama y NVIDIA.
    Requiere haber ejecutado previamente llm_settings.py.
    """

    validations = {}

    validations["ollama"] = validate_client(
        client=ollama_client,
        model=OLLAMA_MODEL_LLAMA,
        provider="Ollama",
        test_connection=test_connection,
    )

    validations["nvidia"] = validate_client(
        client=nvidia_client,
        model=NVIDIA_MODEL,
        provider="NVIDIA NIM",
        test_connection=test_connection,
    )

    return validations

# =
