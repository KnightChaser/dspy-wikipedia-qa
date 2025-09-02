# wikiqa/config.py
from __future__ import annotations

# Milvus vector database configuration
DEFAULT_URI = "./milvus.db"
DEFAULT_COLLECTION_NAME = "wikiqa_chunks"
DEFAULT_EMBED_DIM = 1536
DEFAULT_METRIC_NAME = "COSINE"

# OpenAI model configuration
DEFAULT_EMBED_MODEL_NAME = "text-embedding-3-small"
DEFAULT_CHAT_MODEL_NAME = "gpt-5-mini"

# RAG & Retriever configuration
DEFAULT_TOP_K = 6
DEFAULT_MIN_SCORE = 0.60
DEFAULT_MIN_HITS = 2

# Chunking configuration
DEFAULT_CHUNK_MAX_TOKENS = 350
DEFAULT_CHUNK_OVERLAP = 50
