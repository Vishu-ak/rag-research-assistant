"""
Cross-encoder reranking for improved retrieval precision.

Cross-encoders jointly encode the (query, document) pair and produce a
relevance score that is significantly more accurate than bi-encoder
embedding similarity. They are slower (O(k) forward passes) so they are
applied to a small candidate set retrieved by the faster first stage.

Models:
  - cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, lightweight, ~22M params)
  - cross-encoder/ms-marco-MiniLM-L-12-v2 (more accurate, ~33M params)
  - cross-encoder/ms-marco-electra-base    (highest quality, ~110M params)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.documents import Document

from config.settings import EmbeddingSettings

logger = structlog.get_logger(__name__)


@dataclass
class RerankedResult:
    """A document after cross-encoder reranking."""

    document: Document
    rerank_score: float
    original_rank: int
    final_rank: int

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    def __repr__(self) -> str:
        return (
            f"RerankedResult(rank={self.final_rank}, "
            f"score={self.rerank_score:.4f}, "
            f"source={self.source!r})"
        )


class CrossEncoderReranker:
    """
    Reranker using a cross-encoder model for fine-grained relevance scoring.

    The cross-encoder takes (query, passage) pairs as input and outputs
    a single relevance score. Unlike bi-encoders, the model can attend
    to both texts simultaneously, enabling much richer interaction.

    This reranker is designed to be the second stage in a two-stage
    retrieve-then-rerank pipeline.

    Example:
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        docs = [Document(...), Document(...), ...]
        reranked = reranker.rerank(query="What is attention?", documents=docs, top_k=3)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model ID for the cross-encoder.
            device: Torch device ('cpu', 'cuda', 'mps').
            max_length: Maximum token length for the (query, doc) pair.
            batch_size: Number of pairs to score in one forward pass.
        """
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed: pip install sentence-transformers"
            ) from e

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info("cross_encoder_reranker.loading", model=model_name, device=device)
        start = time.perf_counter()

        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "cross_encoder_reranker.ready",
            model=model_name,
            load_time_s=round(elapsed, 2),
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """
        Rerank documents by cross-encoder relevance score.

        Args:
            query: The original search query.
            documents: Candidate documents from first-stage retrieval.
            top_k: Number of top results to return. If None, returns all reranked.

        Returns:
            List of RerankedResult sorted by rerank_score descending.
        """
        if not documents:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content[:self.max_length * 4]) for doc in documents]

        logger.debug(
            "cross_encoder_reranker.scoring",
            query=query[:60],
            candidate_count=len(pairs),
        )

        start = time.perf_counter()
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Pair documents with their scores
        scored: list[tuple[int, float, Document]] = [
            (orig_rank + 1, float(score), doc)
            for orig_rank, (score, doc) in enumerate(zip(scores, documents))
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        results = [
            RerankedResult(
                document=doc,
                rerank_score=score,
                original_rank=orig_rank,
                final_rank=final_rank + 1,
            )
            for final_rank, (orig_rank, score, doc) in enumerate(scored)
        ]

        logger.info(
            "cross_encoder_reranker.complete",
            candidates=len(documents),
            returned=len(results),
            top_score=round(results[0].rerank_score, 4) if results else None,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return results

    def rerank_hybrid_results(
        self,
        query: str,
        hybrid_results: list[Any],  # HybridResult from retriever
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """
        Convenience method to rerank HybridRetriever results.

        Args:
            query: Search query.
            hybrid_results: List of HybridResult objects.
            top_k: Number of results to return.

        Returns:
            Reranked list of RerankedResult.
        """
        documents = [r.document for r in hybrid_results]
        return self.rerank(query=query, documents=documents, top_k=top_k)


class NoopReranker:
    """
    Pass-through reranker that preserves the original retrieval ordering.

    Used when reranking is disabled in settings, ensuring the pipeline
    interface remains consistent regardless of whether a real cross-encoder
    is in use.
    """

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Return documents in original order without reranking."""
        results = [
            RerankedResult(
                document=doc,
                rerank_score=1.0 - (idx / max(1, len(documents))),  # pseudo-score
                original_rank=idx + 1,
                final_rank=idx + 1,
            )
            for idx, doc in enumerate(documents)
        ]

        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_hybrid_results(
        self,
        query: str,
        hybrid_results: list[Any],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        documents = [r.document for r in hybrid_results]
        return self.rerank(query=query, documents=documents, top_k=top_k)


def create_reranker(
    settings: EmbeddingSettings,
    enable_reranking: bool = True,
    device: str = "cpu",
) -> CrossEncoderReranker | NoopReranker:
    """
    Factory to create the appropriate reranker based on settings.

    Args:
        settings: EmbeddingSettings with reranker model configuration.
        enable_reranking: If False, return a NoopReranker.
        device: Torch device for the model.

    Returns:
        CrossEncoderReranker or NoopReranker.
    """
    if not enable_reranking:
        logger.info("reranker.disabled_using_noop")
        return NoopReranker()

    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=device,
    )
