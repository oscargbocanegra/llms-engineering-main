# LLMs Engineering — Instrucciones para agentes (Copilot)

Este repo es principalmente **notebooks** (no hay `requirements.txt` en el root). La forma “correcta” de trabajar es editar/ejecutar celdas y, cuando haga falta, instalar dependencias desde el notebook.

## Mapa rápido del repo
- Proyectos principales: `Project1-week1.ipynb`, `Project2-week2.ipynb`, `Project3-week3.ipynb`.
- Lecciones por semana: carpetas `week1/`, `week2/`, `week3/`, `week4/` (ej.: `week1/scraper.py`, `week4/day3.ipynb`).

## Entorno y configuración (patrón real del repo)
- Muchos notebooks asumen que el `.env` vive en `/workspace/.env` (ej.: Project 1 y Project 3). Si un notebook usa `load_dotenv(override=True)` y no carga llaves, cámbialo al patrón de ruta explícita.

```python
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='/workspace/.env', override=True)

print('OLLAMA_BASE_URL=', os.getenv('OLLAMA_BASE_URL'))
print('OPENAI_API_KEY set=', bool(os.getenv('OPENAI_API_KEY')))
```

Variables típicas: `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`, `HUGGINGFACE_API_KEY`.

## Patrones LLM (OpenAI + Ollama)
- Se usa el cliente `openai.OpenAI` tanto para OpenAI como para Ollama (endpoint compatible): `OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key=OLLAMA_API_KEY)`.
- Conmutación de proveedor: en Project 1 existe `USE_PROVIDER` (y fallback a `LLM_PROVIDER`).
- Streaming: `client.chat.completions.create(..., stream=True)` y concatenar `chunk.choices[0].delta.content`.

## Tool / Function calling (Project 2)
- Definir tools con schema JSON, ejecutar funciones Python localmente y reinyectar resultados al historial.
- Modelo usado en el notebook: `deepseek-v3.1:671b-cloud` vía Ollama Cloud.

## Multimodal (Project 2)
- Imágenes: Pollinations (`https://image.pollinations.ai/prompt/{prompt}`).
- Audio: Whisper (`whisper.load_model("base")`) + gTTS; a veces se acelera audio con `pydub`.

## Transformers / cuantización (Week 3 + Project 3)
- `BitsAndBytesConfig(load_in_4bit=True, ...)` + `device_map="auto"`.
- Limpieza de VRAM al final: `del ...; torch.cuda.empty_cache()`.

## Scraping (week1/scraper.py)
- Siempre usar header `User-Agent` y limpiar tags `script/style/img/input` antes de extraer texto.

## C++ en notebooks (week4/day3.ipynb)
- Ojo: el notebook corre comandos `!` en el **entorno del kernel** (frecuentemente Linux en contenedor). Compilación típica:
  - `!g++ -O3 -std=c++17 -o optimized optimized.cpp`
  - `!./optimized`
- Al guardar C++ generado por un LLM, eliminar fences tipo ```cpp/``` antes de escribir `optimized.cpp` (ya hay helper `write_output`).
