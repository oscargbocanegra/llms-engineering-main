# ============================================================
# IA-VOICE-R1.7C
# Funciones reutilizables para generación de audio local
#
# No modifica servicios ni configuraciones.
# ============================================================

"""Funciones reutilizables para generación de audio mediante Speaches."""

from __future__ import annotations

import sys
import requests
from pathlib import Path
from typing import Any
from IPython.display import Audio, display

def _value(name: str) -> Any:
    """Obtiene una variable cargada o definida en config.llm_settings."""

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
    except ImportError as error:
        raise ImportError(
            "No se pudo importar config.llm_settings. "
            "Ejecuta el notebook desde la raíz del proyecto."
        ) from error

    return getattr(llm_settings, name, None)


def generar_audio(
    texto: str,
    base_url: str | None = None,
    modelo: str | None = None,
    archivo_salida: str | Path = "audio.wav",
    voz: str = "daniela",
    velocidad: float = 0.7,
    timeout: int = 300,
) -> Path:
    """Convierte texto en audio WAV mediante Speaches."""

    texto_limpio = texto.strip()

    if not texto_limpio:
        raise ValueError("El texto no puede estar vacío.")

    if velocidad <= 0:
        raise ValueError("La velocidad debe ser mayor que cero.")

    resolved_base_url = base_url or _value("BASE_URL_AUDIO")
    resolved_model = modelo or _value("MODEL_PIPEL_DANIELA")

    if not resolved_base_url:
        raise ValueError(
            "BASE_URL_AUDIO no está configurada en apis.env."
        )

    if not resolved_model:
        raise ValueError(
            "MODEL_PIPEL_DANIELA no está configurado en apis.env."
        )

    endpoint = (
        f"{str(resolved_base_url).rstrip('/')}/v1/audio/speech"
    )

    response = requests.post(
        endpoint,
        json={
            "model": resolved_model,
            "voice": voz,
            "input": texto_limpio,
            "response_format": "wav",
            "speed": velocidad,
        },
        timeout=timeout,
    )

    response.raise_for_status()

    if not response.content:
        raise ValueError(
            "El servicio de audio devolvió una respuesta vacía."
        )

    ruta_audio = Path(archivo_salida)
    ruta_audio.parent.mkdir(parents=True, exist_ok=True)
    ruta_audio.write_bytes(response.content)

    return ruta_audio



def hablar(
    mensaje: str,
    voz: str = "daniela",
    modelo: str | None = None,
    velocidad: float = 0.7,
) -> None:
    ruta_audio = generar_audio(
        texto=mensaje,
        modelo=modelo,
        voz=voz,
        velocidad=velocidad,
        archivo_salida="outputs/ultimo_audio.wav",
    )
    display(Audio(filename=str(ruta_audio), autoplay=True))




__all__ = [
    "generar_audio",
    "hablar",
]