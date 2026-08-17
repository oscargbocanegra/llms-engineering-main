"""Utilities for downloading and cleaning website content."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup


def fetch_website_contents(
    url: str,
    timeout: int = 20,
    max_characters: int = 20_000,
) -> str:
    """
    Download a public webpage and return its visible text.

    This implementation uses browser-like headers because some sites reject
    requests made with the default Python Requests user agent.
    """
    url = (url or "").strip()

    if not url:
        raise ValueError("La URL no puede estar vacía.")

    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "La URL debe comenzar con http:// o https://"
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    with requests.Session() as session:
        response = session.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )

    if response.status_code == 403:
        raise RuntimeError(
            "El sitio web rechazó la descarga automática con HTTP 403. "
            "Puede tener protección anti-bots o requerir JavaScript."
        )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    if "text/html" not in content_type:
        raise ValueError(
            f"La URL no devolvió HTML. Content-Type: {content_type}"
        )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
            "svg",
        ]
    ):
        element.decompose()

    text = soup.get_text(
        separator="\n",
        strip=True,
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    cleaned_text = "\n".join(lines)

    if not cleaned_text:
        raise RuntimeError(
            "El sitio respondió, pero no se pudo extraer texto visible."
        )

    return cleaned_text[:max_characters]