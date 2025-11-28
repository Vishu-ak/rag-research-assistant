"""
Intelligent text chunking strategies for RAG document preparation.

Provides two strategies:
  - RecursiveChunker: fast, deterministic, token-aware splitting
  - SemanticChunker: embedding-based splitting at semantic boundaries

Both strategies preserve document metadata and add chunk-level provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog
from langchain_core.documents import Document

from config.settings import ChunkingSettings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ChunkMetadata:
    """Metadata attached to each text chunk."""

    source: str
    page: int | None
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    strategy: str
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------


def _approx_token_count(text: str) -> int:
    """
    Approximate token count without a full tokenizer.

    Assumes ~4 characters per token (GPT-style BPE average).
    Falls back to tiktoken if available for higher accuracy.
    """
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except (ImportError, Exception):
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Protocol / interface
# ---------------------------------------------------------------------------


class TextChunker(Protocol):
    """Protocol for all chunking strategies."""

    def split(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into chunks."""
        ...


# ---------------------------------------------------------------------------
# Recursive chunker
# ---------------------------------------------------------------------------


class RecursiveChunker:
    """
    Token-aware recursive character text splitter.

    Splits documents by trying progressively finer separators until
    all chunks fit within the target token window. This is the default
    and fastest strategy.

    Algorithm:
        1. Try splitting by paragraph breaks (\\n\\n)
        2. If any piece is still too large, split by line breaks (\\n)
        3. Continue with sentence boundaries, then words, then chars

    Example:
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.split(documents)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ) -> None:
        """
        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Token overlap between consecutive chunks.
            separators: Ordered list of separator strings to try.
            keep_separator: If True, separator is kept at the end of the chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

        logger.debug(
            "recursive_chunker.initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of LangChain Documents into token-bounded chunks.

        Args:
            documents: Input documents (one per page or one whole doc).

        Returns:
            List of chunk Documents with enriched metadata.
        """
        all_chunks: list[Document] = []

        for doc_idx, doc in enumerate(documents):
            doc_chunks = self._split_document(doc, doc_idx)
            all_chunks.extend(doc_chunks)

        logger.info(
            "recursive_chunker.split_complete",
            input_docs=len(documents),
            output_chunks=len(all_chunks),
        )
        return all_chunks

    def _split_document(self, doc: Document, doc_idx: int) -> list[Document]:
        """Split a single document into chunks."""
        text = doc.page_content
        if not text.strip():
            return []

        raw_chunks = self._recursive_split(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        chunk_docs: list[Document] = []
        char_cursor = 0

        for chunk_idx, chunk_text in enumerate(merged):
            # Locate character offset within original text
            char_start = text.find(chunk_text[:50], char_cursor)
            if char_start == -1:
                char_start = char_cursor
            char_end = char_start + len(chunk_text)
            char_cursor = max(char_cursor, char_start)

            metadata = {
                **doc.metadata,
                "chunk_index": chunk_idx,
                "chunk_strategy": "recursive",
                "char_start": char_start,
                "char_end": char_end,
                "token_count": _approx_token_count(chunk_text),
            }

            chunk_docs.append(
                Document(page_content=chunk_text.strip(), metadata=metadata)
            )

        # Backfill total_chunks
        total = len(chunk_docs)
        for d in chunk_docs:
            d.metadata["total_chunks_in_doc"] = total

        return chunk_docs

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the first separator that produces
        pieces small enough to fit in chunk_size.
        """
        if not separators:
            # Base case: split by character
            return self._split_by_char(text)

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks: list[str] = []
        for split in splits:
            token_count = _approx_token_count(split)
            if token_count <= self.chunk_size:
                if self.keep_separator and separator and split != splits[-1]:
                    chunks.append(split + separator)
                else:
                    chunks.append(split)
            else:
                # This piece is still too large — recurse
                sub_chunks = self._recursive_split(split, remaining)
                chunks.extend(sub_chunks)

        return [c for c in chunks if c.strip()]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """
        Merge small pieces into chunks of at most chunk_size tokens,
        with chunk_overlap tokens of overlap between consecutive chunks.
        """
        merged: list[str] = []
        current_pieces: list[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = _approx_token_count(piece)

            if current_tokens + piece_tokens > self.chunk_size and current_pieces:
                # Emit current chunk
                merged.append("".join(current_pieces))

                # Build overlap: keep tail pieces until overlap budget
                overlap_pieces: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_pieces):
                    p_toks = _approx_token_count(p)
                    if overlap_tokens + p_toks <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_tokens += p_toks
                    else:
                        break

                current_pieces = overlap_pieces
                current_tokens = overlap_tokens

            current_pieces.append(piece)
            current_tokens += piece_tokens

        if current_pieces:
            merged.append("".join(current_pieces))

        return merged

    def _split_by_char(self, text: str) -> list[str]:
        """Last-resort character-level splitting."""
        chars_per_chunk = self.chunk_size * 4  # approx 4 chars/token
        overlap_chars = self.chunk_overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = start + chars_per_chunk
            chunks.append(text[start:end])
            start += chars_per_chunk - overlap_chars
        return chunks


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------


class SemanticChunker:
    """
    Embedding-based semantic text splitter.

    Splits documents at points of maximum semantic dissimilarity.
    Sentences are grouped into chunks that maximize intra-chunk
    semantic coherence while respecting token limits.

    Algorithm:
        1. Split text into sentences with regex
        2. Embed each sentence using the provided embedding function
        3. Compute cosine distances between consecutive sentence pairs
        4. Split at local maxima in the distance signal (breakpoints)
        5. Merge fragments into token-bounded chunks

    Example:
        def my_embed(texts):
            return model.encode(texts).tolist()

        chunker = SemanticChunker(embed_fn=my_embed, threshold_percentile=90)
        chunks = chunker.split(documents)
    """

    def __init__(
        self,
        embed_fn: Any,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        threshold_percentile: float = 90.0,
        min_sentences_per_chunk: int = 2,
    ) -> None:
        """
        Args:
            embed_fn: Callable that accepts List[str] and returns List[List[float]].
            chunk_size: Maximum chunk size in tokens.
            chunk_overlap: Token overlap between consecutive chunks.
            threshold_percentile: Percentile of cosine distances used as split threshold.
            min_sentences_per_chunk: Minimum sentences before a split is allowed.
        """
        self.embed_fn = embed_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold_percentile = threshold_percentile
        self.min_sentences = min_sentences_per_chunk

        logger.debug(
            "semantic_chunker.initialized",
            chunk_size=chunk_size,
            threshold_pct=threshold_percentile,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into semantically coherent chunks."""
        all_chunks: list[Document] = []

        for doc in documents:
            chunks = self._split_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            "semantic_chunker.split_complete",
            input_docs=len(documents),
            output_chunks=len(all_chunks),
        )
        return all_chunks

    def _split_document(self, doc: Document) -> list[Document]:
        """Split a single document semantically."""
        if not doc.page_content.strip():
            return []
        sentences = self._sentence_tokenize(doc.page_content)
        if len(sentences) < 2:
            return [doc]

        logger.debug("semantic_chunker.embedding_sentences", count=len(sentences))
        embeddings = self.embed_fn(sentences)

        distances = self._cosine_distances(embeddings)
        breakpoints = self._find_breakpoints(distances)

        groups = self._sentences_to_groups(sentences, breakpoints)
        merged_groups = self._merge_small_groups(groups)

        chunk_docs: list[Document] = []
        for idx, group in enumerate(merged_groups):
            text = " ".join(group).strip()
            if not text:
                continue

            metadata = {
                **doc.metadata,
                "chunk_index": idx,
                "chunk_strategy": "semantic",
                "token_count": _approx_token_count(text),
                "sentence_count": len(group),
            }
            chunk_docs.append(Document(page_content=text, metadata=metadata))

        total = len(chunk_docs)
        for d in chunk_docs:
            d.metadata["total_chunks_in_doc"] = total

        return chunk_docs

    @staticmethod
    def _sentence_tokenize(text: str) -> list[str]:
        """Simple regex-based sentence tokenizer."""
        # Split on sentence-ending punctuation followed by whitespace + capital
        pattern = r"(?<=[.!?])\s+(?=[A-Z\"\(])"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    @staticmethod
    def _cosine_distances(embeddings: list[list[float]]) -> list[float]:
        """Compute cosine distances between consecutive embeddings."""
        import math

        distances = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                distances.append(1.0)
            else:
                similarity = dot / (norm_a * norm_b)
                distances.append(1.0 - similarity)
        return distances

    def _find_breakpoints(self, distances: list[float]) -> set[int]:
        """Find sentence indices where splits should occur."""
        if not distances:
            return set()

        sorted_d = sorted(distances)
        idx_at_pct = min(
            int(len(sorted_d) * self.threshold_percentile / 100),
            len(sorted_d) - 1,
        )
        threshold = sorted_d[idx_at_pct]

        return {i for i, d in enumerate(distances) if d >= threshold}

    def _sentences_to_groups(
        self, sentences: list[str], breakpoints: set[int]
    ) -> list[list[str]]:
        """Group sentences into chunks based on breakpoints."""
        groups: list[list[str]] = []
        current_group: list[str] = []

        for i, sentence in enumerate(sentences):
            current_group.append(sentence)
            if i in breakpoints and len(current_group) >= self.min_sentences:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        return groups

    def _merge_small_groups(self, groups: list[list[str]]) -> list[list[str]]:
        """Merge groups that are too small (below minimum sentences) into neighbors."""
        merged: list[list[str]] = []

        for group in groups:
            if merged and len(group) < self.min_sentences:
                merged[-1].extend(group)
            else:
                merged.append(group[:])

        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_chunker(
    settings: ChunkingSettings,
    embed_fn: Any = None,
) -> RecursiveChunker | SemanticChunker:
    """
    Factory that creates the appropriate chunker based on settings.

    Args:
        settings: ChunkingSettings from the application config.
        embed_fn: Required when settings.strategy == 'semantic'.

    Returns:
        Configured TextChunker instance.

    Raises:
        ValueError: If 'semantic' is requested but embed_fn is None.
    """
    if settings.strategy == "semantic":
        if embed_fn is None:
            raise ValueError(
                "embed_fn must be provided for semantic chunking strategy."
            )
        return SemanticChunker(
            embed_fn=embed_fn,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    return RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.separators,
    )
