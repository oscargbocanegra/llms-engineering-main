"""Workflow helpers for the Private Knowledge Chatbot (RAG) case study.

These functions use lazy imports so the main package can be imported even when
RAG-specific dependencies are not installed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from llm_portfolio.private_rag_chatbot.config import RagConfig, resolve_kb_dir


class MissingRagDependencyError(RuntimeError):
    """Raised when optional RAG dependencies are not installed."""


def _require(module_name: str, install_hint: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extras
        raise MissingRagDependencyError(
            f"Missing optional dependency '{module_name}'. Install the RAG extras: {install_hint}"
        ) from exc


def load_pdf_documents(base_dir: str | Path = ".", knowledge_dir: str = "knowledge-base"):
    directory_loader = _require("langchain_community.document_loaders", "pip install -e .[rag]")
    kb_dir = resolve_kb_dir(base_dir, knowledge_dir)
    loader = directory_loader.DirectoryLoader(
        str(kb_dir),
        glob="**/*.pdf",
        loader_cls=directory_loader.PyPDFLoader,
        show_progress=True,
    )
    return loader.load()


def chunk_documents(documents: Sequence, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitters = _require("langchain_text_splitters", "pip install -e .[rag]")
    text_splitter = splitters.CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(list(documents))


def build_embeddings(provider: str = "openai", model_name: str | None = None):
    provider = provider.lower()
    if provider == "openai":
        openai_mod = _require("langchain_openai", "pip install -e .[rag]")
        return openai_mod.OpenAIEmbeddings(model=model_name) if model_name else openai_mod.OpenAIEmbeddings()
    if provider in {"huggingface", "hf"}:
        community = _require("langchain_community.embeddings", "pip install -e .[rag]")
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return community.HuggingFaceEmbeddings(model_name=model_name)
    raise ValueError(f"Unsupported embeddings provider: {provider}")


def build_vectorstore(documents: Sequence, embeddings, persist_directory: str = "vector_db", reset_existing: bool = True):
    chroma_mod = _require("langchain_chroma", "pip install -e .[rag]")
    if reset_existing and os.path.exists(persist_directory):
        chroma_mod.Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        ).delete_collection()
    return chroma_mod.Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def make_retriever(vectorstore, k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_conversation_chain(vectorstore, model: str = "gpt-4o-mini", temperature: float = 0.7, k: int = 4):
    openai_mod = _require("langchain_openai", "pip install -e .[rag]")
    memory_mod = _require("langchain_classic.memory", "pip install -e .[rag]")
    chains_mod = _require("langchain_classic.chains", "pip install -e .[rag]")

    llm = openai_mod.ChatOpenAI(model=model, temperature=temperature)
    retriever = make_retriever(vectorstore, k=k)
    memory = memory_mod.ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return chains_mod.ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )


def build_private_rag_stack(base_dir: str | Path = ".", config: RagConfig | None = None):
    config = config or RagConfig()
    documents = load_pdf_documents(base_dir=base_dir)
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    embeddings = build_embeddings("openai")
    vectorstore = build_vectorstore(chunks, embeddings, config.persist_directory)
    chain = build_conversation_chain(vectorstore, config.model, k=config.retrieval_k)
    return {
        "documents": documents,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "conversation_chain": chain,
    }
