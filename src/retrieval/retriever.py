"""
Hybrid retrieval combining dense (semantic) and sparse (BM25) search.

Implements Reciprocal Rank Fusion (RRF) to merge dense and sparse
result lists into a single, calibrated ranking. This typically outperforms
either approach alone, especially for keyword-heavy queries.

References:
    Cormack et al. (2009), "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods"
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.documents import Document

from config.settings import RetrievalSettings
from src.embeddings.embedding_manager import BaseEmbedder
from src.embeddings.vector_store import SearchResult, VectorStore

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Sparse (BM25) retriever
# ---------------------------------------------------------------------------


@dataclass
class BM25Result:
    """A single BM25 retrieval result."""

    document: Document
    score: float
    rank: int


class BM25Index:
    """
    In-memory BM25 index for sparse term-based retrieval.

    Builds an inverted index over document chunks for fast BM25 scoring.
    Designed to be rebuilt from the vector store on each startup (lightweight).

    Parameters k1 and b follow Robertson's recommendations for document retrieval.

    Example:
        index = BM25Index(documents)
        results = index.search("transformer attention mechanism", k=10)
    """

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Args:
            documents: Corpus of document chunks to index.
            k1: Term saturation parameter (typically 1.2–2.0).
            b: Length normalization parameter (0=none, 1=full).
        """
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.corpus_size = len(documents)

        # Build index
        self._tokenized_docs: list[list[str]] = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self._doc_lengths: list[int] = [len(tok) for tok in self._tokenized_docs]
        self._avg_doc_length: float = (
            sum(self._doc_lengths) / max(1, self.corpus_size)
        )

        # Inverted index: term → {doc_idx: term_freq}
        self._inverted_index: dict[str, dict[int, int]] = defaultdict(dict)
        for doc_idx, tokens in enumerate(self._tokenized_docs):
            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
            for term, freq in term_freqs.items():
                self._inverted_index[term][doc_idx] = freq

        # Document frequency for each term
        self._df: dict[str, int] = {
            term: len(postings) for term, postings in self._inverted_index.items()
        }

        logger.info(
            "bm25_index.built",
            doc_count=self.corpus_size,
            vocab_size=len(self._inverted_index),
            avg_doc_length=round(self._avg_doc_length, 1),
        )

    def search(self, query: str, k: int = 20) -> list[BM25Result]:
        """
        Retrieve top-k documents using BM25 scoring.

        Args:
            query: Natural language search query.
            k: Maximum number of results to return.

        Returns:
            List of BM25Result sorted by score descending.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)

        for term in set(query_terms):
            if term not in self._inverted_index:
                continue

            df = self._df[term]
            idf = self._idf(df)

            for doc_idx, tf in self._inverted_index[term].items():
                doc_len = self._doc_lengths[doc_idx]
                normalized_tf = (
                    tf * (self.k1 + 1)
                    / (
                        tf
                        + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                    )
                )
                scores[doc_idx] += idf * normalized_tf

        # Sort by score descending and take top-k
        top_k_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:k]

        return [
            BM25Result(
                document=self.documents[idx],
                score=scores[idx],
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_k_indices)
        ]

    def update(self, new_documents: list[Document]) -> None:
        """Append new documents to the index (incremental update)."""
        start_idx = self.corpus_size

        for doc_idx, doc in enumerate(new_documents, start=start_idx):
            tokens = self._tokenize(doc.page_content)
            self.documents.append(doc)
            self._tokenized_docs.append(tokens)
            self._doc_lengths.append(len(tokens))

            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                self._inverted_index[term][doc_idx] = freq
                self._df[term] = len(self._inverted_index[term])

        self.corpus_size += len(new_documents)
        self._avg_doc_length = sum(self._doc_lengths) / max(1, self.corpus_size)

        logger.debug("bm25_index.updated", added=len(new_documents), total=self.corpus_size)

    def _idf(self, df: int) -> float:
        """Compute IDF with Robertson's smoothed formula."""
        return math.log(
            (self.corpus_size - df + 0.5) / (df + 0.5) + 1.0
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace+punctuation tokenizer with lowercasing."""
        import re

        # Lowercase and remove punctuation
        text = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9-]*\b", text)

        # Remove very short tokens and simple stopwords
        STOPWORDS = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "not", "with", "from", "by", "be",
            "this", "that", "are", "was", "were", "has", "have", "had",
            "do", "does", "did", "will", "would", "could", "should",
        }
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


@dataclass
class HybridResult:
    """Fused retrieval result with combined RRF score."""

    document: Document
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None

    def __repr__(self) -> str:
        src = self.document.metadata.get("source", "?")
        return (
            f"HybridResult(rrf={self.rrf_score:.4f}, "
            f"dense_rank={self.dense_rank}, sparse_rank={self.sparse_rank}, "
            f"source={src!r})"
        )


class HybridRetriever:
    """
    Hybrid dense + sparse retriever with Reciprocal Rank Fusion (RRF).

    Combines the strengths of:
    - Dense retrieval: Semantic similarity via sentence embeddings
    - Sparse retrieval: Exact/keyword matching via BM25

    RRF formula:
        RRF(doc) = Σ 1 / (k + rank_i)
    where k=60 is a smoothing constant from Cormack et al. (2009).

    Example:
        retriever = HybridRetriever(
            vector_store=store,
            bm25_index=bm25,
            settings=retrieval_settings,
        )
        results = retriever.retrieve("What are transformer attention heads?")
    """

    RRF_K = 60  # RRF smoothing constant

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        settings: RetrievalSettings,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.settings = settings
        self.dense_weight = settings.hybrid_dense_weight
        self.sparse_weight = 1.0 - settings.hybrid_dense_weight

        logger.info(
            "hybrid_retriever.initialized",
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            top_k=settings.top_k,
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """
        Retrieve documents using hybrid RRF fusion.

        Args:
            query: Natural language search query.
            k: Number of results to return (defaults to settings.top_k).
            metadata_filter: Optional ChromaDB metadata filter.

        Returns:
            List of HybridResult sorted by fused score descending.
        """
        k = k or self.settings.top_k
        candidate_k = k * 3  # Retrieve more candidates for reranking

        # Parallel dense and sparse retrieval
        dense_results = self.vector_store.similarity_search(
            query=query,
            k=candidate_k,
            where=metadata_filter,
        )

        sparse_results = self.bm25_index.search(query=query, k=candidate_k)

        # Fuse with RRF
        fused = self._rrf_fusion(dense_results, sparse_results)

        logger.info(
            "hybrid_retriever.retrieved",
            query_preview=query[:60],
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(fused),
            returning=min(k, len(fused)),
        )

        return fused[:k]

    def retrieve_dense_only(self, query: str, k: int | None = None) -> list[SearchResult]:
        """Retrieve using dense search only (for ablation or comparison)."""
        k = k or self.settings.top_k
        return self.vector_store.similarity_search(query=query, k=k)

    def retrieve_sparse_only(self, query: str, k: int | None = None) -> list[BM25Result]:
        """Retrieve using BM25 only (for ablation or comparison)."""
        k = k or self.settings.top_k
        return self.bm25_index.search(query=query, k=k)

    def _rrf_fusion(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[BM25Result],
    ) -> list[HybridResult]:
        """
        Apply Reciprocal Rank Fusion to merge dense and sparse rankings.

        Returns a unified list sorted by fused score descending.
        """
        # Build doc_id → rank/score maps
        dense_map: dict[str, tuple[int, float, Document]] = {}
        for rank, result in enumerate(dense_results, start=1):
            doc_key = self._doc_key(result.document)
            dense_map[doc_key] = (rank, result.score, result.document)

        sparse_map: dict[str, tuple[int, float, Document]] = {}
        for rank, result in enumerate(sparse_results, start=1):
            doc_key = self._doc_key(result.document)
            sparse_map[doc_key] = (rank, result.score, result.document)

        # Collect all unique doc keys
        all_keys = set(dense_map.keys()) | set(sparse_map.keys())

        fused_scores: list[HybridResult] = []

        for key in all_keys:
            dense_rank: int | None = None
            dense_score: float | None = None
            sparse_rank: int | None = None
            sparse_score: float | None = None
            doc: Document | None = None

            rrf_score = 0.0

            if key in dense_map:
                d_rank, d_score, d_doc = dense_map[key]
                dense_rank = d_rank
                dense_score = d_score
                rrf_score += self.dense_weight / (self.RRF_K + d_rank)
                doc = d_doc

            if key in sparse_map:
                s_rank, s_score, s_doc = sparse_map[key]
                sparse_rank = s_rank
                sparse_score = s_score
                rrf_score += self.sparse_weight / (self.RRF_K + s_rank)
                if doc is None:
                    doc = s_doc

            if doc is not None:
                fused_scores.append(
                    HybridResult(
                        document=doc,
                        rrf_score=rrf_score,
                        dense_rank=dense_rank,
                        sparse_rank=sparse_rank,
                        dense_score=dense_score,
                        sparse_score=sparse_score,
                    )
                )

        return sorted(fused_scores, key=lambda r: r.rrf_score, reverse=True)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Generate a deduplication key from document metadata."""
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk = doc.metadata.get("chunk_index", "")
        if source:
            return f"{source}::{page}::{chunk}"
        # Fall back to content hash
        import hashlib

        return hashlib.md5(doc.page_content[:200].encode()).hexdigest()
