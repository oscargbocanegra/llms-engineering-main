# Extracted reusable modules

This document explains the reusable code that has been extracted from flagship notebook case studies.

## Why these modules exist

The repository is still notebook-first, but selected case studies now expose reusable code under `src/llm_portfolio/` so the public portfolio shows more than exploratory notebooks.

## Current extracted modules

### `synthetic_data_studio/`
Derived from `Project3-week3.ipynb`.

Includes:
- schema definitions
- prompt construction
- CSV parsing helpers
- quality validation
- orchestration workflow

Use when you want to reuse the synthetic-data generation flow outside the notebook.

### `private_rag_chatbot/`
Derived from `Project5-week5.ipynb`.

Includes:
- configuration helpers
- knowledge-base path resolution
- lazy-imported RAG workflow functions
- retriever / vectorstore / conversation chain assembly

Use when you want a lightweight reusable entrypoint into the private knowledge chatbot stack without forcing every notebook dependency into the base install.

## Smoke examples

- `examples/synthetic_data_studio_smoke.py`
- `examples/private_rag_chatbot_smoke.py`

## Notes

- These modules are an extraction layer, not a full product package.
- Heavy notebook dependencies remain optional where possible.
- The next step would be to keep extracting stable logic while leaving demo/UI behavior in notebooks or app surfaces.
