"""Minimal smoke example for the extracted Private Knowledge Chatbot helpers.

This script only exercises path/config helpers so it stays lightweight. The full
RAG stack requires optional extras installed via `pip install -e .[rag]`.
"""

from llm_portfolio.private_rag_chatbot import RagConfig, resolve_kb_dir


if __name__ == "__main__":
    config = RagConfig()
    print(config)
    print(resolve_kb_dir())
    print("Install optional extras to run the full RAG workflow: pip install -e .[rag]")
