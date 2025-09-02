# wikiqa/rag_dspy.py
from __future__ import annotations

import dspy
from dataclasses import dataclass
from typing import Sequence

from wikiqa import config
from wikiqa.index_milvus import get_client
from wikiqa.retriever_milvus import MilvusRetriever, Passage


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """
    Configuration for the RAG system using dspy and Milvus.
    """

    collection: str = config.DEFAULT_COLLECTION_NAME
    uri: str = config.DEFAULT_URI
    top_k: int = config.DEFAULT_TOP_K
    openai_model: str = config.DEFAULT_EMBED_MODEL_NAME


class GenerateAnswer(dspy.Signature):
    """
    A signature for generating answers using retrieved context and a question.
    """

    context = dspy.InputField(desc="relevant facts and snippets with headings")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="2–5 sentences with inline citations like [1], [2]")


class SimpleRAG(dspy.Module):
    """
    A simple Retrieval-Augmented Generation (RAG) system using dspy and Milvus.
    """

    def __init__(
        self, retriever: MilvusRetriever, *, top_k: int = config.DEFAULT_TOP_K
    ):
        super().__init__()
        self.retriever = retriever
        self.top_k = top_k
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        """
        Given a question, retrieve relevant passages and generate an answer with citations.
        """
        hits: Sequence[Passage] = self.retriever(question, k=self.top_k)
        if not hits:
            # Zero-context guard: Reject if no passages are found
            msg = (
                "I don't have enough indexed context to answer that safely. "
                'Try indexing relevant pages first (e.g., `index-title "Jupiter"`), '
                "then ask again."
            )
            return dspy.Prediction(answer=msg, sources=[])

        contexts: list[str] = []
        srcs: list[str] = []

        for idx, passage in enumerate(hits, start=1):
            contexts.append(f"[{idx}] {passage.long_text}")
            url = passage.meta.get("url")
            if url:
                srcs.append(f"[{idx}] {url}")
        ctx = "\n\n".join(contexts)
        pred = self.generate(context=ctx, question=question)
        return dspy.Prediction(answer=pred.answer, sources=srcs)


def build_rag_pipeline(
    uri: str = config.DEFAULT_URI,
    collection_name: str = config.DEFAULT_COLLECTION_NAME,
    model_name: str = config.DEFAULT_EMBED_MODEL_NAME,
    k: int = config.DEFAULT_TOP_K,
    min_score: float = config.DEFAULT_MIN_SCORE,
    min_hits: int = config.DEFAULT_MIN_HITS,
    # NOTE: gpt-5-mini requires temperature=1.0, max_tokens>=20000.
    # This number may be adjusted by the caller.
    temperature: float = 1.0,
    max_tokens: int = 20000,
    # Allow passing a custom retriever (e.g., for testing)
    retriever: MilvusRetriever | None = None,
) -> SimpleRAG:
    """
    Builds and configures the complete (fully-configured)
    SimpleRAG pipeline
    """
    db = get_client(uri=uri)
    if retriever is None:
        retriever = MilvusRetriever(
            client=db,
            collection=collection_name,
            min_score=min_score,
            min_hits=min_hits,
            output_fields=("text", "url", "page_title", "section_path", "lang"),
        )

    lm = dspy.LM(model=model_name, temperature=temperature, max_tokens=max_tokens)
    dspy.settings.configure(lm=lm)

    return SimpleRAG(retriever=retriever, top_k=k)
