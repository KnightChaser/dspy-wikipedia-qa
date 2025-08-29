# wikiqa/retriever_milvus.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np
from pymilvus import MilvusClient
from pymilvus import model as milvus_model


@dataclass(frozen=True, slots=True)
class Passage:
    """
    A passage retrieved from the vector store.
    """

    long_text: str
    score: float
    meta: dict[str, Any]


class MilvusRetriever:
    """
    Minimal Milvus-backed retriever for text passages.
    - Encodes queries with pymilvus' OpenAIEmbeddingFunction
    - Searches a collection and returns passages that SimpleRAG can consume
    """

    def __init__(
        self,
        *,
        collection_name: str,
        uri: str,
        ef: milvus_model.dense.OpenAIEmbeddingFunction,
        output_fields: list[str] | None = None,
    ) -> None:
        self.client = MilvusClient(uri=uri)
        self.collection = collection_name
        self.ef = ef
        self.output_fields = output_fields or [
            "text",
            "page_title",
            "section_path",
            "url",
            "lang",
            "token_estimate",
        ]

    def _encode_query(self, query: str) -> list[float]:
        """
        Encode a query string into a vector using the embedding function.
        """
        vec = self.ef.encode_queries([query])[0]
        return vec.tolist() if isinstance(vec, np.ndarray) else vec

    def __call__(self, query: str, k: int = 6) -> list[Passage]:
        """
        Retrieve top-k passages for a given query.
        """
        query_vec = self._encode_query(query)
        res = self.client.search(
            collection_name=self.collection,
            data=[query_vec],
            limit=k,
            output_fields=self.output_fields,
        )

        hits = list(self._iter_hits(res))
        out: list[Passage] = []

        for hit in hits[:k]:
            entity = hit.get("entity", {})
            text = entity.get("text", "")
            url = entity.get("url") or None
            title = entity.get("page_title") or None
            section_path = entity.get("section_path") or None
            lang = entity.get("lang") or None
            distance = float(hit.get("distance", 0.0))

            out.append(
                Passage(
                    long_text=text,
                    score=distance,  # similarity score
                    meta={
                        "url": url,
                        "page_title": title,
                        "section_path": section_path,
                        "lang": lang,
                    },
                )
            )

        return out

    def _hit_to_dict(self, hit: Any) -> dict:
        """
        Normalize a pymilvus Hit/HybridHit or a plain dict into:
        {
            "id": ...,
            "distance": ...,
            "entity": { ... },
        }
        """
        # In case of a plain dictionary
        if isinstance(hit, dict):
            # ensure keys that we expect exist
            return {
                "id": hit.get("id"),
                "distance": float(hit.get("distance", 0.0)),
                "entity": hit.get("entity", {}) or {},
            }

        # Hit-like object (Classic/Hits/HybridHits path)
        hit_id = getattr(hit, "id", None)
        distance = float(getattr(hit, "distance", 0.0))
        entity = getattr(hit, "entity", {}) or {}
        if entity is None:
            entity = getattr(hit, "fields", None)
        if entity is None:  # still none?
            entity = {}

        return {
            "id": hit_id,
            "distance": distance,
            "entity": dict(entity) if isinstance(entity, dict) else entity,
        }

    def _iter_hits(self, res: Any) -> Iterable[dict]:
        """
        Yield normalized hit dicts from pymilvus search() result across versions:
        - New high-level MilvusClient: res behaves like a list; res[0] is a list[dict]
        - Lower-level Collection.search: res[0] is a Hits/HybridHits object (iterable of Hit)
        - Rare dict-like return: {"results": [...]}
        """
        # dict-like (defensive check)
        if isinstance(res, dict):
            for hit in res.get("results", []):
                yield self._hit_to_dict(hit)
            return

        # Sequence-like data type
        try:
            first = res[0]
        except Exception:
            # final fallback: try iterating the result directly.
            try:
                for hit in res:
                    yield self._hit_to_dict(hit)
            except Exception:
                return

        else:
            # case A: already a list[dict]
            if isinstance(first, list):
                for hit in first:
                    yield self._hit_to_dict(hit)
                return

            # case B: iterable of Hit-like objects
            try:
                for hit in first:
                    yield self._hit_to_dict(hit)
            except TypeError:
                pass

        return
