"""
Embedding generation with support for multiple backends.

Provides a unified interface for generating dense vector embeddings
via OpenAI's API or local SentenceTransformers models. Includes
batching, caching, and retry logic for production reliability.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import EmbeddingSettings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Abstract interface for embedding backends."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (may use a different prompt prefix)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...


# ---------------------------------------------------------------------------
# OpenAI embedder
# ---------------------------------------------------------------------------


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder backed by OpenAI's text-embedding API.

    Uses exponential backoff retries on transient API errors.
    Supports both v1 and v3 embedding models.

    Example:
        embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small")
        vectors = embedder.embed_documents(["Hello world", "Goodbye world"])
    """

    DIMENSION_MAP = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
        request_timeout: int = 60,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("openai package not installed: pip install openai") from e

        self._client = OpenAI(api_key=api_key, timeout=request_timeout)
        self._model = model
        self._batch_size = batch_size
        self._dim = self.DIMENSION_MAP.get(model, 1536)

        logger.info("openai_embedder.initialized", model=model)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in batches with retry on failure."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "openai_embedder.batch_complete",
                batch_num=i // self._batch_size + 1,
                total_batches=(len(texts) + self._batch_size - 1) // self._batch_size,
            )

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embed_batch([text])[0]

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry logic."""
        # Truncate texts that exceed model context length (conservative limit)
        truncated = [t[:32000] for t in texts]

        response = self._client.embeddings.create(
            input=truncated,
            model=self._model,
        )

        # Ensure ordering matches input
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


# ---------------------------------------------------------------------------
# SentenceTransformers embedder
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedder using HuggingFace SentenceTransformers.

    Supports GPU acceleration and model caching. BGE models use
    a query instruction prefix for asymmetric retrieval.

    Example:
        embedder = SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5")
        query_vec = embedder.embed_query("What is machine learning?")
    """

    # BGE models use a query prefix for asymmetric retrieval
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    BGE_MODELS = {"BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"}

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed: pip install sentence-transformers"
            ) from e

        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._show_progress = show_progress_bar
        self._use_query_prefix = model_name in self.BGE_MODELS

        logger.info(
            "st_embedder.loading",
            model=model_name,
            device=device,
        )
        start = time.perf_counter()
        self._model = SentenceTransformer(model_name, device=device)
        elapsed = time.perf_counter() - start

        # Determine actual embedding dimension
        test_vec = self._model.encode("test", convert_to_numpy=True)
        self._dim = int(test_vec.shape[0])

        logger.info(
            "st_embedder.ready",
            model=model_name,
            dimension=self._dim,
            load_time_s=round(elapsed, 2),
        )

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
        if not texts:
            return []

        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=self._show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query, optionally prepending a retrieval prefix."""
        if self._use_query_prefix:
            text = self.BGE_QUERY_PREFIX + text

        vector: np.ndarray = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return vector.tolist()


# ---------------------------------------------------------------------------
# Cached embedder wrapper
# ---------------------------------------------------------------------------


class CachedEmbedder(BaseEmbedder):
    """
    Wraps any BaseEmbedder with a disk-backed embedding cache.

    Caches embeddings keyed by text hash + model name to avoid
    recomputing embeddings for identical text content across runs.

    Example:
        base = SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5")
        cached = CachedEmbedder(base, cache_dir="./.cache/embeddings")
        vecs = cached.embed_documents(texts)  # first call: compute + cache
        vecs = cached.embed_documents(texts)  # second call: instant cache hit
    """

    def __init__(self, embedder: BaseEmbedder, cache_dir: str = "./.cache/embeddings") -> None:
        self._embedder = embedder
        self._cache_dir = Path(cache_dir) / embedder.model_name.replace("/", "_")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

        logger.info("cached_embedder.initialized", cache_dir=str(self._cache_dir))

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents, loading from cache when available."""
        results: list[list[float]] = [None] * len(texts)  # type: ignore
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                results[i] = cached
                self._hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self._misses += 1

        if uncached_texts:
            new_vecs = self._embedder.embed_documents(uncached_texts)
            for idx, (orig_idx, text, vec) in enumerate(
                zip(uncached_indices, uncached_texts, new_vecs)
            ):
                results[orig_idx] = vec
                self._save_to_cache(text, vec)

        logger.debug(
            "cached_embedder.stats",
            hits=self._hits,
            misses=self._misses,
            hit_rate=round(self._hits / max(1, self._hits + self._misses), 3),
        )

        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a query (caches using 'query:' prefix in key)."""
        cache_text = f"query:{text}"
        cached = self._load_from_cache(cache_text)
        if cached is not None:
            return cached

        vec = self._embedder.embed_query(text)
        self._save_to_cache(cache_text, vec)
        return vec

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_path(self, text: str) -> Path:
        return self._cache_dir / f"{self._text_hash(text)}.json"

    def _load_from_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _save_to_cache(self, text: str, vector: list[float]) -> None:
        path = self._cache_path(text)
        try:
            path.write_text(json.dumps(vector))
        except OSError as exc:
            logger.warning("cached_embedder.save_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedder(
    settings: EmbeddingSettings,
    use_cache: bool = True,
    cache_dir: str = "./.cache/embeddings",
    openai_api_key: str = "",
) -> BaseEmbedder:
    """
    Factory function to create an embedder based on settings.

    Args:
        settings: EmbeddingSettings from application config.
        use_cache: Wrap the embedder with a disk cache.
        cache_dir: Directory for embedding cache files.
        openai_api_key: Required if settings.backend == 'openai'.

    Returns:
        Configured BaseEmbedder instance.
    """
    if settings.backend == "openai":
        if not openai_api_key:
            raise ValueError("openai_api_key is required for OpenAI embedding backend.")
        embedder: BaseEmbedder = OpenAIEmbedder(
            api_key=openai_api_key,
            batch_size=settings.embedding_batch_size,
        )
    else:
        embedder = SentenceTransformerEmbedder(
            model_name=settings.sentence_transformer_model,
            device=settings.device,
            batch_size=settings.embedding_batch_size,
        )

    if use_cache:
        embedder = CachedEmbedder(embedder, cache_dir=cache_dir)

    return embedder
