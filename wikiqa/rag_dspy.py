# wikiqa/rag_dspy.py
from __future__ import annotations

import dspy
from dataclasses import dataclass
from typing import Sequence
from wikiqa.retriever_milvus import MilvusRetriever, Passage


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """
    Configuration for the RAG system using dspy and Milvus.
    """

    collection: str
    uri: str
    top_k: int = 6
    openai_model: str = "gpt-5-mini"


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

    def __init__(self, retriever: MilvusRetriever, *, top_k: int = 6):
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
