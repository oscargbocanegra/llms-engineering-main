# LLMs Engineering - Copilot Instructions

## Project Overview
This repository is a professional training program for LLM Engineering, organized into weekly modules and projects. It combines Jupyter Notebooks for interactive learning and Python scripts for utility functions.

## Codebase Structure
- **Root**: Contains `README.md`, license, and project-level notebooks (e.g., `week1-Project1.ipynb`).
- **Weekly Folders** (`week1/`, `week2/`, etc.):
  - `dayX.ipynb`: Daily lesson notebooks.
  - `*.py`: Helper scripts (e.g., `week1/scraper.py`).
  - `__pycache__/`: Compiled python files (ignore).

## Development Workflow
- **Primary Environment**: Jupyter Notebooks running in a Python 3.10+ environment.
- **Environment Variables**:
  - Load configuration from the global `.env` file located at `/workspace/.env`.
  - Use `dotenv.load_dotenv(dotenv_path='/workspace/.env', override=True)` to ensure variables are loaded.
  - Validate critical variables (`OLLAMA_BASE_URL`, `OPENAI_API_KEY`) immediately after loading, using professional logging or print statements.
- **Local LLMs (Ollama)**:
  - Use the `openai` Python client to interact with Ollama.
  - Configure the client with `base_url=f"{ollama_base_url}/v1"` and the API key from env.
- **Cloud LLMs (OpenAI)**:
  - Use the standard OpenAI API configuration when accessing frontier models.

## Coding Conventions
- **Notebooks**:
  - Use Markdown cells to explain concepts before code cells.
  - Keep code cells modular and independent where possible.
  - Use `IPython.display` for rich output (Markdown, HTML).
  - Use `ipywidgets` for interactive elements when appropriate.
- **Code Quality & Patterns**:
  - Follow PEP 8 style and enforce type hinting (`typing`).
  - Apply SOLID principles and well-defined design patterns (Factory, Strategy, Singleton) where appropriate.
  - Write professional, production-ready code with robust error handling.
  - Use docstrings for all functions and classes.
- **Dependencies**:
  - Common libraries: `openai`, `requests`, `beautifulsoup4`, `pandas`, `numpy`, `python-dotenv`.
  - Web scraping: Use `BeautifulSoup` for parsing HTML. Always include a `User-Agent` header in `requests` to mimic a browser (e.g., Chrome on Windows).

## Key Integration Points
- **Ollama via OpenAI Client**: The project standard is to use the `openai` library to interface with local Ollama instances.
- **Web Scraping**: Custom utilities in `week1/scraper.py` demonstrate the pattern of fetching and cleaning content. Mimic the `fetch_website_contents` logic (cleaning scripts/styles) for new scrapers.

## Specific Patterns
- **Streaming**: Implement streaming responses for LLM interactions to improve user experience (typewriter effect).
- **Tool Use**: Function calling patterns are used (e.g., in Project 2).
- **Multimodal**: Integration of text, image, and voice (Whisper).

## Debugging
- If a notebook cell fails, check if the required local services (like Ollama) are active.
- Verify API keys are set in the environment.

## Interaction Guidelines
- **No Emojis**: Do not use emojis in chat responses, code outputs, or comments.
- **Tone**: Maintain a strictly professional and technical tone.
