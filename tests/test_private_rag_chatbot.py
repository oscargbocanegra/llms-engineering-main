from llm_portfolio.private_rag_chatbot import RagConfig, resolve_kb_dir


def test_resolve_kb_dir_joins_paths():
    path = resolve_kb_dir(base_dir=".", knowledge_dir="knowledge-base")
    assert str(path).endswith("knowledge-base")


def test_rag_config_defaults_are_stable():
    config = RagConfig()
    assert config.model == "gpt-4o-mini"
    assert config.persist_directory == "vector_db"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.retrieval_k == 4
