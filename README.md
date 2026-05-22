# LLM Engineering Portfolio

**Applied LLM systems, multimodal assistants, retrieval workflows, and local-model experimentation.**

This repository is a curated portfolio of hands-on LLM engineering work focused on **real application patterns**, not just prompts or isolated notebook experiments. It brings together practical prototypes across **multimodal AI**, **synthetic data generation**, **retrieval-augmented generation (RAG)**, and **evaluation-oriented workflows**.

---

## Why this repository matters

The goal of this repository is to demonstrate practical LLM engineering skills across multiple dimensions:

- building end-to-end AI applications
- integrating models with tools, interfaces, and workflows
- working with both frontier and local/open-source models
- handling retrieval, validation, and evaluation concerns
- turning exploratory work into reusable portfolio artifacts

This repository originated from structured training material, but it has been reframed as a **public engineering portfolio** centered on applied systems and technical experimentation.

---

## Featured case studies

### 1. FlightAI — Multimodal Airline Assistant
**A multimodal customer-service assistant with tool calling, transcription, image generation, and interactive UI.**

**What it demonstrates**
- tool calling and orchestration
- multimodal interaction (text, voice, image)
- Whisper-based audio transcription
- TTS response generation
- Gradio interface design
- practical product-style AI workflow design

**Key technologies**
`Python` `Whisper` `gTTS` `Gradio` `Ollama` `Function Calling`

**Main artifact**
- [`Project2-week2.ipynb`](./Project2-week2.ipynb)

---

### 2. Synthetic Data Studio
**A local-model synthetic dataset generator with schema-aware output handling and validation.**

**What it demonstrates**
- local LLM deployment
- quantized model usage
- structured prompting
- schema-driven output generation
- CSV parsing resilience
- validation and quality checks
- resource-aware AI experimentation

**Key technologies**
`Python` `Transformers` `Llama 3.1` `BitsAndBytes` `Gradio` `Pandas` `PyTorch`

**Main artifact**
- [`Project3-week3.ipynb`](./Project3-week3.ipynb)
- Reusable extraction: `src/llm_portfolio/synthetic_data_studio/`

---

### 3. Private Knowledge Chatbot (RAG)
**A retrieval-based chatbot over a private local knowledge base using embeddings and a vector database.**

**What it demonstrates**
- document ingestion
- chunking strategy
- embeddings pipeline
- vector database persistence
- conversational retrieval
- grounding answers on retrieved context
- retrieval debugging and embedding-space inspection

**Key technologies**
`Python` `LangChain` `ChromaDB` `OpenAI` `Gradio` `Plotly` `scikit-learn` `RAG`

**Main artifact**
- [`Project5-week5.ipynb`](./Project5-week5.ipynb)

---

## Supporting experiments

These artifacts are valuable, but they are positioned as supporting experiments rather than the primary flagship story.

### Price Estimation / Evaluation Work
Exploration of model comparison and evaluation-oriented workflows around product price prediction and benchmarking.

**Relevant files**
- [`week6/testing.py`](./week6/testing.py)
- [`week6/items.py`](./week6/items.py)
- [`week6/loaders.py`](./week6/loaders.py)
- `week6/day4.ipynb`

### Early Local / Prompting / Foundation Work
Initial exploratory work around prompt engineering, local-model setup, and interactive tutoring workflows.

**Relevant files**
- [`Project1-week1.ipynb`](./Project1-week1.ipynb)

---

## What this repository currently proves

This portfolio currently demonstrates:

- end-to-end LLM application prototyping
- multimodal system assembly
- local-model experimentation and quantization-aware workflows
- retrieval-based architectures
- model/tool integration
- exploratory evaluation thinking
- notebook-driven rapid iteration with practical outcomes

---

## Repository structure

```text
Project1-week1.ipynb       Foundation project
Project2-week2.ipynb       Multimodal airline assistant
Project3-week3.ipynb       Synthetic data generation
Project5-week5.ipynb       Private RAG chatbot

week1/ ... week6/          Supporting weekly experiments and exploratory work
knowledge-base/            Example private knowledge artifacts
src/llm_portfolio/         Reusable code extracted from flagship case studies
```

Generated vector stores, visualization exports, audio outputs, and local caches are treated as reproducible demo artifacts and should not define the public signal of the repository.

---

## Technical themes

### LLM Applications
- prompt engineering
- tool calling
- multimodal workflows
- local and hosted model comparison

### Retrieval & Knowledge Systems
- embeddings
- vector databases
- private knowledge retrieval
- grounded responses

### Local Model Engineering
- quantization
- inference constraints
- resource-aware experimentation
- model optimization tradeoffs

### Evaluation & Reliability
- validation of structured outputs
- benchmarking-oriented workflows
- debugging embeddings and retrieval behavior
- comparing practical system behaviors

---

## Getting started

### Notebook-first workflow

This repository is still primarily **notebook-first**, but it now also includes an emerging reusable code surface under `src/llm_portfolio/`.

Recommended workflow:

1. Create and activate a Python virtual environment.
2. Install the minimal dependencies.
3. Run one flagship notebook (for example `Project5-week5.ipynb`) for the end-to-end experience.
4. Use the extracted modules in `src/llm_portfolio/` when you want reusable logic instead of notebook cells.
5. Use the supporting `week*/` directories only when you need deeper context or experiments.

### Environment setup

```bash
python -m venv llms-env
source llms-env/bin/activate  # On Windows: llms-env\Scripts\activate
pip install -r requirements.txt
```

### Reusable module smoke example

```bash
python examples/synthetic_data_studio_smoke.py
```

### Packaging

```bash
pip install -e .
```

### Configuration

Create a `.env` file when a notebook requires external providers.

```env
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=llama3.2
LLM_PROVIDER=ollama
```

---

## Current limitations

This repository is still partially **notebook-first** and contains some generated/demo artifacts that should be further curated over time.

Current limitations include:

- most core logic is still concentrated in notebooks, although extraction has started for Synthetic Data Studio under `src/llm_portfolio/`
- reproducibility and packaging can be improved
- some demonstrations still rely on local/generated artifacts that should stay out of the flagship story
- not all ideas are implemented as polished case studies yet

That said, the strongest completed artifacts already provide meaningful public proof of applied LLM engineering work.

---

## Roadmap

Short-term improvements:

- extract reusable code from the remaining flagship notebooks into `.py` modules
- expand packaging and reproducibility beyond the first extracted Synthetic Data Studio slice
- add stronger case-study framing, architecture notes, and demo summaries
- isolate flagship work from archive/learning material more clearly

---

## Origins

This repository includes work that originated in structured learning and course-driven experimentation. The public goal, however, is broader: to turn that exploration into a **credible engineering portfolio** centered on practical LLM systems.

---

## Related portfolio projects

You may also be interested in:

- [lab-infra-ia-bigdata](https://github.com/oscargbocanegra/lab-infra-ia-bigdata) — self-hosted AI + Big Data platform
- [semantic-concept-extraction-pipeline](https://github.com/oscargbocanegra/semantic-concept-extraction-pipeline) — semantic NLP extraction showcase
- [mcp-for-beginners](https://github.com/oscargbocanegra/mcp-for-beginners) — MCP implementations and tooling

---

## Contact

- **GitHub:** [@oscargbocanegra](https://github.com/oscargbocanegra)
- **Portfolio:** [oscargbocanegra.github.io](https://oscargbocanegra.github.io/)
- **LinkedIn:** [oscargbocanegra](https://www.linkedin.com/in/oscargbocanegra/)

If you're interested in **LLM systems, RAG, multimodal AI, or applied AI engineering**, feel free to connect.
