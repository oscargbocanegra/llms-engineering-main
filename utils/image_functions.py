from __future__ import annotations
from typing import Optional
from PIL import Image
from huggingface_hub import InferenceClient

import sys

def _value(name):
    """
    Obtain configuration values regardless of whether they come from:

    - current notebook
    - __main__
    - config.llm_settings
    """
    value = globals().get(name)

    if value is not None:
        return value

    main = sys.modules.get("__main__")

    if main is not None:
        value = getattr(main, name, None)

        if value is not None:
            return value

    try:
        from config import llm_settings

        return getattr(llm_settings, name)

    except (ImportError, AttributeError):
        return None


# -------------------------------------------------------------------------
# HuggingFace
# -------------------------------------------------------------------------

def _hf_client() -> InferenceClient:
    """
    Create a Hugging Face inference client.
    """

    token = _value("HF_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN is not configured."
        )

    return InferenceClient(
        api_key=token
    )


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def generate_image(
    prompt: str,
    model: Optional[str] = None,
    provider: str = "huggingface",
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
) -> Image.Image:
    """
    Generate an image.

    Parameters
    ----------
    prompt
        Image description.

    model
        Hugging Face model.

    provider
        Future-proof parameter.
        Currently only supports "huggingface".

    Returns
    -------
    PIL.Image
    """

    provider = provider.lower()

    if provider != "huggingface":
        raise ValueError(
            f"Unknown provider: {provider}"
        )

    client = _hf_client()

    model = model or _value("HF_MODEL")

    image = client.text_to_image(
        prompt=prompt,
        model=model,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    return image


def save_image(
    image: Image.Image,
    filename: str,
):
    """
    Save a PIL image.
    """

    image.save(filename)

    return filename