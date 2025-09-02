# wikiqa/index_milvus.py
from __future__ import annotations
from typing import Iterable
from pymilvus import MilvusClient
from pymilvus import model as milvus_model

from wikiqa import config
from wikiqa.datatypes import Chunk


def get_client(uri: str = config.DEFAULT_URI) -> MilvusClient:
    """
    Get a Milvus client connected to the given URI.
    """
    return MilvusClient(uri)


def get_openai_ef(
    model_name: str = config.DEFAULT_EMBED_MODEL_NAME,
) -> milvus_model.dense.OpenAIEmbeddingFunction:
    """
    Build an OpenAI embedding function via pymilvus model library.
    Uses OPENAI_API_KEY from env by default
    """
    return milvus_model.dense.OpenAIEmbeddingFunction(
        model_name=model_name,
    )


def ensure_collection(
    client: MilvusClient,
    *,
    collection_name: str = config.DEFAULT_COLLECTION_NAME,
    dim: int = config.DEFAULT_EMBED_DIM,
    overwrite: bool = False,
) -> None:
    """
    Ensure the collection exists with the given schema.
    """
    existing = set(client.list_collections())
    if collection_name in existing and not overwrite:
        # Collection already exists, nothing to do further
        return

    client.create_collection(
        collection_name=collection_name,
        overwrite=overwrite,
        dimension=dim,
        primary_field_name="id",
        vector_field_name="embedding",
        id_type="int",
        metric_type=config.DEFAULT_METRIC_NAME,
        enable_dynamic=True,  # store extra fields in $meta
        max_length=65535,
    )


def upsert_chunks(
    client: MilvusClient,
    collection_name: str,
    chunks: Iterable[Chunk],
    ef: milvus_model.dense.OpenAIEmbeddingFunction,
    *,
    batch_size: int = 64,
) -> int:
    """
    Insert (or upsert) chunk embeddings + metadata in batches.
    Returns the number of inserted items.
    """
    inserted = 0
    batch: list[dict] = []

    def flush() -> None:
        """
        Flush the current batch to Milvus.

        1. Inserts the current batch to Milvus.
        2. Updates the inserted count.
        3. Clears the batch.
        """
        nonlocal inserted
        if not batch:
            return

        client.insert(
            collection_name=collection_name,
            data=batch,
        )
        inserted += len(batch)
        batch.clear()

    texts = [chunk.text for chunk in chunks]
    if not texts:
        # Nothing to do, 0 elements to insert
        return 0

    # Encode documents in bulk for throughput
    docs_embeddings = ef.encode_documents(texts)
    for _, (chunk, embedding) in enumerate(zip(chunks, docs_embeddings)):
        item = {
            "id": chunk.id_int,
            "embedding": embedding.tolist(),
            "text": chunk.text,
            "page_title": chunk.page_title,
            "section_path": chunk.section_path,
            "url": chunk.url,
            "lang": chunk.lang,
            "token_estimate": chunk.token_estimate,
        }
        batch.append(item)
        if len(batch) >= batch_size:
            flush()

    # Flush any remaining items
    flush()

    return inserted
