"""
ChromaDB vector store integration.

Provides a high-level wrapper around ChromaDB with CRUD operations,
batch upserts, metadata filtering, and collection management.
Supports both local persistent storage and remote ChromaDB server mode.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.documents import Document

from config.settings import ChromaSettings
from src.embeddings.embedding_manager import BaseEmbedder

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single vector search result with distance score."""

    document: Document
    score: float  # cosine similarity (0-1, higher is better)
    doc_id: str

    def __repr__(self) -> str:
        source = self.document.metadata.get("source", "?")
        return f"SearchResult(score={self.score:.4f}, source={source!r})"


@dataclass
class CollectionStats:
    """Statistics about a ChromaDB collection."""

    name: str
    document_count: int
    embedding_dimension: int
    distance_function: str


# ---------------------------------------------------------------------------
# ChromaDB wrapper
# ---------------------------------------------------------------------------


class VectorStore:
    """
    Production ChromaDB vector store with full CRUD support.

    Manages document embedding, storage, retrieval, and deletion.
    Handles batch processing, ID generation, and metadata filtering.

    Example:
        store = VectorStore(settings=chroma_settings, embedder=embedder)
        store.add_documents(chunks)
        results = store.similarity_search("machine learning", k=5)
    """

    BATCH_SIZE = 100  # Chroma batch limit

    def __init__(
        self,
        settings: ChromaSettings,
        embedder: BaseEmbedder,
        collection_name: str | None = None,
    ) -> None:
        """
        Args:
            settings: ChromaDB connection and collection settings.
            embedder: Embedder for converting text to vectors.
            collection_name: Override the collection name from settings.
        """
        self.settings = settings
        self.embedder = embedder
        self.collection_name = collection_name or settings.collection_name

        self._client = self._build_client(settings)
        self._collection = self._get_or_create_collection(
            self.collection_name, settings.distance_function
        )

        logger.info(
            "vector_store.initialized",
            collection=self.collection_name,
            mode="server" if settings.use_server else "local",
            doc_count=self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[str]:
        """
        Add documents to the vector store.

        Computes embeddings and upserts in batches. Already-existing
        documents (by ID) are updated rather than duplicated.

        Args:
            documents: List of LangChain Documents to embed and store.
            ids: Optional list of document IDs (auto-generated if None).
            show_progress: Log progress for large batches.

        Returns:
            List of document IDs that were added/updated.
        """
        if not documents:
            logger.warning("vector_store.add_empty")
            return []

        doc_ids = ids or [self._generate_id(doc) for doc in documents]

        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [self._sanitize_metadata(doc.metadata) for doc in documents]

        total_added = 0
        all_ids: list[str] = []

        # Process in batches
        for batch_start in range(0, len(texts), self.BATCH_SIZE):
            batch_end = batch_start + self.BATCH_SIZE
            batch_texts = texts[batch_start:batch_end]
            batch_meta = metadatas[batch_start:batch_end]
            batch_ids = doc_ids[batch_start:batch_end]

            start = time.perf_counter()
            batch_embeddings = self.embedder.embed_documents(batch_texts)
            embed_ms = (time.perf_counter() - start) * 1000

            self._collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

            total_added += len(batch_texts)
            all_ids.extend(batch_ids)

            if show_progress:
                logger.info(
                    "vector_store.batch_added",
                    batch=batch_start // self.BATCH_SIZE + 1,
                    count=total_added,
                    total=len(texts),
                    embed_ms=round(embed_ms, 1),
                )

        logger.info("vector_store.add_complete", total_docs=total_added)
        return all_ids

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents from the store by ID.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        self._collection.delete(ids=ids)
        logger.info("vector_store.deleted", count=len(ids))

    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents originating from a given source file.

        Args:
            source: The file path or source identifier to match.

        Returns:
            Number of documents deleted.
        """
        results = self._collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        ids = results.get("ids", [])

        if ids:
            self._collection.delete(ids=ids)
            logger.info("vector_store.delete_by_source", source=source, count=len(ids))

        return len(ids)

    def clear(self) -> None:
        """Delete all documents from the collection."""
        count = self._collection.count()
        if count == 0:
            return

        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection(
            self.collection_name, self.settings.distance_function
        )
        logger.warning("vector_store.cleared", deleted_count=count)

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for documents semantically similar to the query.

        Args:
            query: Natural language query string.
            k: Number of results to return.
            where: Metadata filter dict (ChromaDB `where` syntax).
            where_document: Document content filter dict.

        Returns:
            List of SearchResult ordered by similarity (descending).
        """
        query_vec = self.embedder.embed_query(query)
        return self.similarity_search_by_vector(
            query_vec, k=k, where=where, where_document=where_document
        )

    def similarity_search_by_vector(
        self,
        vector: list[float],
        k: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search using a pre-computed query embedding vector.

        Args:
            vector: Dense embedding vector.
            k: Number of results to return.
            where: Metadata filter.
            where_document: Document content filter.

        Returns:
            List of SearchResult ordered by similarity (descending).
        """
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": min(k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document

        raw = self._collection.query(**query_kwargs)

        results: list[SearchResult] = []
        if not raw or not raw.get("ids"):
            return results

        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]

        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
            # Convert distance to similarity score (cosine: dist = 1 - similarity)
            score = 1.0 - dist if self.settings.distance_function == "cosine" else -dist
            results.append(
                SearchResult(
                    document=Document(page_content=doc_text, metadata=meta or {}),
                    score=score,
                    doc_id=doc_id,
                )
            )

        return results

    def get_document_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a specific document by its ID."""
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
        if not result or not result.get("documents"):
            return None

        return Document(
            page_content=result["documents"][0],
            metadata=result["metadatas"][0] or {},
        )

    def get_stats(self) -> CollectionStats:
        """Return statistics about the current collection."""
        count = self._collection.count()

        # Determine embedding dimension from a sample document if possible
        dim = self.embedder.dimension

        return CollectionStats(
            name=self.collection_name,
            document_count=count,
            embedding_dimension=dim,
            distance_function=self.settings.distance_function,
        )

    def list_sources(self) -> list[str]:
        """Return all unique source document paths in the collection."""
        results = self._collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in results.get("metadatas", []) or []:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(settings: ChromaSettings) -> Any:
        """Construct the ChromaDB client (local or HTTP)."""
        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise ImportError("chromadb not installed: pip install chromadb") from e

        if settings.use_server:
            logger.info(
                "vector_store.connecting_server",
                host=settings.host,
                port=settings.port,
            )
            return chromadb.HttpClient(host=settings.host, port=settings.port)

        logger.info(
            "vector_store.connecting_local",
            persist_dir=settings.persist_dir,
        )
        return chromadb.PersistentClient(path=settings.persist_dir)

    @staticmethod
    def _get_or_create_collection(client: Any, name: str, distance_fn: str) -> Any:
        """Get or create a ChromaDB collection with the specified distance function."""
        try:
            import chromadb.utils.embedding_functions as ef  # type: ignore
        except ImportError:
            pass

        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": distance_fn},
        )

    def _get_or_create_collection(self, name: str, distance_fn: str) -> Any:  # type: ignore[override]
        """Instance method wrapper."""
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": distance_fn},
        )

    @staticmethod
    def _generate_id(doc: Document) -> str:
        """
        Generate a deterministic ID for a document chunk.

        Based on source + page + chunk_index to ensure idempotent upserts.
        Falls back to a random UUID if metadata is insufficient.
        """
        import hashlib

        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk = doc.metadata.get("chunk_index", "")

        if source:
            raw = f"{source}::{page}::{chunk}::{doc.page_content[:100]}"
            return hashlib.sha256(raw.encode()).hexdigest()[:32]

        return str(uuid.uuid4())

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize metadata for ChromaDB storage.

        ChromaDB only supports str, int, float, and bool values.
        Lists and nested dicts are serialized to strings.
        None values are dropped.
        """
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)
        return sanitized
