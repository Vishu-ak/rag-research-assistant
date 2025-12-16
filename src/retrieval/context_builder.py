"""
Context window management and citation tracking for RAG generation.

Assembles retrieved document chunks into a structured context string
that fits within the LLM's context window. Tracks source citations
and provides formatted context with page references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_core.documents import Document

from src.retrieval.reranker import RerankedResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A source citation extracted from a retrieved chunk."""

    index: int  # 1-based citation number
    source: str  # file path or URL
    page: int | None
    chunk_index: int | None
    title: str | None
    doi: str | None
    authors: str | None
    date: str | None
    text_snippet: str  # first ~150 chars of the chunk

    def format_short(self) -> str:
        """Return a short [1] style citation reference."""
        return f"[{self.index}]"

    def format_full(self) -> str:
        """Return a full citation string for the references section."""
        parts = [f"[{self.index}]"]

        if self.title:
            parts.append(self.title)
        if self.authors:
            parts.append(f"by {self.authors}")
        if self.date:
            parts.append(f"({self.date})")
        if self.source:
            import os
            filename = os.path.basename(self.source)
            parts.append(f"— {filename}")
        if self.page:
            parts.append(f"p.{self.page}")
        if self.doi:
            parts.append(f"DOI: {self.doi}")

        return " ".join(parts)


@dataclass
class BuiltContext:
    """The assembled context ready for LLM generation."""

    context_text: str           # Formatted context with chunk separators
    citations: list[Citation]   # Ordered list of citations
    total_tokens: int           # Estimated token count
    chunks_included: int        # Number of chunks that fit
    chunks_truncated: int       # Number of chunks dropped for token budget
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_block(self) -> str:
        """Return a formatted references block for appending to answers."""
        if not self.citations:
            return ""
        lines = ["**Sources:**"]
        for c in self.citations:
            lines.append(c.format_full())
        return "\n".join(lines)

    @property
    def source_list(self) -> list[str]:
        """Return deduplicated list of source file paths."""
        seen: set[str] = set()
        sources: list[str] = []
        for c in self.citations:
            if c.source not in seen:
                seen.add(c.source)
                sources.append(c.source)
        return sources


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """
    Assembles retrieved chunks into an LLM-ready context window.

    Handles:
    - Token budget enforcement (never exceeds max_context_tokens)
    - Per-chunk formatting with citation indices [1], [2], …
    - Source deduplication and citation tracking
    - Chunk ordering (by relevance score or by document order)
    - Metadata extraction from chunk metadata

    Example:
        builder = ContextBuilder(max_context_tokens=3000)
        context = builder.build(query="What is attention?", results=reranked_results)
        print(context.context_text)
        print(context.citation_block)
    """

    CHUNK_TEMPLATE = "[{idx}] (Source: {source}, Page: {page})\n{text}\n"
    DEFAULT_SEPARATOR = "\n---\n"

    def __init__(
        self,
        max_context_tokens: int = 3000,
        chunk_separator: str = DEFAULT_SEPARATOR,
        order_by: str = "relevance",  # "relevance" or "document"
        include_metadata_header: bool = True,
    ) -> None:
        """
        Args:
            max_context_tokens: Maximum tokens in the assembled context string.
            chunk_separator: String inserted between chunks.
            order_by: Sort chunks by 'relevance' (reranker score) or 'document'
                (source document order for narrative coherence).
            include_metadata_header: Prepend a brief metadata header to each chunk.
        """
        self.max_context_tokens = max_context_tokens
        self.chunk_separator = chunk_separator
        self.order_by = order_by
        self.include_metadata_header = include_metadata_header

    def build(
        self,
        query: str,
        results: list[RerankedResult],
        max_chunks: int | None = None,
    ) -> BuiltContext:
        """
        Build a context string from reranked retrieval results.

        Args:
            query: The original user query (used for logging/metadata).
            results: Reranked document results from the retrieval stage.
            max_chunks: Hard limit on number of chunks to include.

        Returns:
            BuiltContext with formatted text, citations, and stats.
        """
        if not results:
            logger.warning("context_builder.empty_results", query=query[:60])
            return BuiltContext(
                context_text="No relevant documents were found.",
                citations=[],
                total_tokens=0,
                chunks_included=0,
                chunks_truncated=0,
            )

        ordered = self._order_chunks(results)

        if max_chunks:
            ordered = ordered[:max_chunks]

        context_parts: list[str] = []
        citations: list[Citation] = []
        total_token_budget = self.max_context_tokens
        remaining_tokens = total_token_budget
        truncated = 0

        for idx, result in enumerate(ordered, start=1):
            chunk_text = self._format_chunk(result.document, idx)
            chunk_tokens = self._estimate_tokens(chunk_text)

            if chunk_tokens > remaining_tokens:
                # Try to fit a truncated version
                truncated_text = self._truncate_to_tokens(
                    result.document.page_content,
                    remaining_tokens - 80,  # 80 tokens for header/footer
                )
                if len(truncated_text) < 100:
                    # Not enough budget for even a snippet — stop
                    truncated += len(ordered) - idx + 1
                    break

                chunk_text = self._format_chunk_with_text(result.document, idx, truncated_text)
                chunk_tokens = self._estimate_tokens(chunk_text)
                truncated += len(ordered) - idx

                context_parts.append(chunk_text)
                citations.append(self._extract_citation(result.document, idx))
                remaining_tokens -= chunk_tokens
                break

            context_parts.append(chunk_text)
            citations.append(self._extract_citation(result.document, idx))
            remaining_tokens -= chunk_tokens

        context_text = self.chunk_separator.join(context_parts)

        logger.info(
            "context_builder.built",
            query=query[:60],
            chunks_included=len(context_parts),
            chunks_truncated=truncated,
            total_tokens=total_token_budget - remaining_tokens,
            unique_sources=len({c.source for c in citations}),
        )

        return BuiltContext(
            context_text=context_text,
            citations=citations,
            total_tokens=total_token_budget - remaining_tokens,
            chunks_included=len(context_parts),
            chunks_truncated=truncated,
            metadata={
                "query": query,
                "total_candidates": len(results),
            },
        )

    def build_from_documents(
        self,
        query: str,
        documents: list[Document],
        scores: list[float] | None = None,
    ) -> BuiltContext:
        """
        Build context from raw Documents without RerankedResult wrappers.

        Convenience method for cases where reranking is disabled.

        Args:
            query: User query.
            documents: Retrieved document chunks.
            scores: Optional relevance scores parallel to documents.

        Returns:
            BuiltContext instance.
        """
        from src.retrieval.reranker import RerankedResult

        results = [
            RerankedResult(
                document=doc,
                rerank_score=scores[i] if scores else (1.0 - i / max(1, len(documents))),
                original_rank=i + 1,
                final_rank=i + 1,
            )
            for i, doc in enumerate(documents)
        ]
        return self.build(query=query, results=results)

    def _order_chunks(self, results: list[RerankedResult]) -> list[RerankedResult]:
        """Order chunks for context assembly."""
        if self.order_by == "document":
            # Group by source, sort within source by page/chunk
            def doc_sort_key(r: RerankedResult) -> tuple[str, int, int]:
                meta = r.document.metadata
                return (
                    meta.get("source", ""),
                    meta.get("page", 0) or 0,
                    meta.get("chunk_index", 0) or 0,
                )

            return sorted(results, key=doc_sort_key)

        # Default: relevance order (already sorted by reranker)
        return sorted(results, key=lambda r: r.rerank_score, reverse=True)

    def _format_chunk(self, doc: Document, idx: int) -> str:
        """Format a single chunk with header."""
        return self._format_chunk_with_text(doc, idx, doc.page_content)

    def _format_chunk_with_text(self, doc: Document, idx: int, text: str) -> str:
        """Format a chunk with a specific text (for truncated versions)."""
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page")

        import os
        source_display = os.path.basename(source) if source != "unknown" else source

        if self.include_metadata_header:
            page_str = f", Page {page}" if page else ""
            header = f"[{idx}] {source_display}{page_str}"
            return f"{header}\n{text.strip()}"
        else:
            return f"[{idx}] {text.strip()}"

    @staticmethod
    def _extract_citation(doc: Document, idx: int) -> Citation:
        """Extract a Citation object from a Document's metadata."""
        meta = doc.metadata
        return Citation(
            index=idx,
            source=meta.get("source", "unknown"),
            page=meta.get("page"),
            chunk_index=meta.get("chunk_index"),
            title=meta.get("title"),
            doi=meta.get("doi"),
            authors=meta.get("authors"),
            date=meta.get("date"),
            text_snippet=doc.page_content[:150].replace("\n", " "),
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count (~4 chars per token)."""
        return max(1, len(text) // 4)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens tokens."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        # Try to cut at a sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.7:
            return truncated[: last_period + 1]
        return truncated + "…"
