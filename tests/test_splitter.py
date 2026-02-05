"""
Unit tests for the text splitting module.

Tests cover:
- RecursiveChunker: basic splitting, overlap, edge cases
- SemanticChunker: sentence tokenization, breakpoint detection
- Factory function
- Token counting utility
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.ingestion.text_splitter import (
    RecursiveChunker,
    SemanticChunker,
    _approx_token_count,
    create_chunker,
)
from config.settings import ChunkingSettings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> Document:
    """A realistic multi-paragraph document."""
    text = (
        "Attention mechanisms have become an integral part of compelling sequence "
        "modeling and transduction models in various tasks.\n\n"
        "The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks that include an encoder and a decoder.\n\n"
        "The Transformer model architecture dispenses with recurrence and convolutions "
        "entirely, relying instead on an attention mechanism to draw global dependencies "
        "between input and output.\n\n"
        "Multi-head attention allows the model to jointly attend to information from "
        "different representation subspaces at different positions.\n\n"
        "Scaled Dot-Product Attention computes the dot products of the query with all "
        "keys, divides each by the square root of the dimension, and applies a softmax "
        "function to obtain the weights on the values.\n\n"
        "Positional encoding must be added to give the model some notion of the "
        "position of the tokens in the sequence."
    )
    return Document(page_content=text, metadata={"source": "test.pdf", "page": 1})


@pytest.fixture
def large_document() -> Document:
    """A long document that requires multiple chunks."""
    paragraph = "This is a paragraph about natural language processing and machine learning. " * 10
    text = "\n\n".join([paragraph] * 20)
    return Document(page_content=text, metadata={"source": "large_test.pdf", "page": 1})


@pytest.fixture
def empty_document() -> Document:
    return Document(page_content="", metadata={"source": "empty.pdf"})


@pytest.fixture
def single_sentence_document() -> Document:
    return Document(page_content="Hello world.", metadata={"source": "tiny.pdf"})


# ---------------------------------------------------------------------------
# Token counting tests
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_approx_token_count_basic(self):
        count = _approx_token_count("Hello world")
        assert count >= 1

    def test_approx_token_count_empty(self):
        count = _approx_token_count("")
        assert count >= 1  # min 1

    def test_approx_token_count_long_text(self):
        text = "word " * 1000
        count = _approx_token_count(text)
        # Should be in a reasonable range (5000 chars / 4 ≈ 1250)
        assert 500 < count < 2000

    def test_approx_token_count_increases_with_length(self):
        short = _approx_token_count("Hello")
        long = _approx_token_count("Hello world, this is a much longer sentence.")
        assert long > short


# ---------------------------------------------------------------------------
# RecursiveChunker tests
# ---------------------------------------------------------------------------


class TestRecursiveChunker:
    def test_splits_document_into_multiple_chunks(self, large_document):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([large_document])
        assert len(chunks) > 1

    def test_single_small_document_produces_one_chunk(self, single_sentence_document):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.split([single_sentence_document])
        assert len(chunks) >= 1
        assert chunks[0].page_content.strip() != ""

    def test_empty_document_returns_empty_list(self, empty_document):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.split([empty_document])
        assert chunks == []

    def test_chunks_preserve_metadata(self, sample_document):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([sample_document])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.pdf"
            assert chunk.metadata.get("page") == 1

    def test_chunks_have_chunk_index(self, sample_document):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([sample_document])
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("chunk_index") == i

    def test_chunks_have_token_count(self, sample_document):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([sample_document])
        for chunk in chunks:
            assert "token_count" in chunk.metadata
            assert chunk.metadata["token_count"] > 0

    def test_chunks_respect_size_limit(self, large_document):
        chunk_size = 150
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.split([large_document])
        for chunk in chunks:
            token_count = _approx_token_count(chunk.page_content)
            # Allow 20% slack for final chunk
            assert token_count <= chunk_size * 1.2, (
                f"Chunk too large: {token_count} tokens > {chunk_size}"
            )

    def test_multiple_documents(self, sample_document, large_document):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([sample_document, large_document])
        sources = {c.metadata["source"] for c in chunks}
        assert "test.pdf" in sources
        assert "large_test.pdf" in sources

    def test_chunk_content_not_empty(self, sample_document):
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.split([sample_document])
        for chunk in chunks:
            assert chunk.page_content.strip() != ""

    def test_total_chunks_backfilled(self, large_document):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([large_document])
        total_reported = chunks[0].metadata.get("total_chunks_in_doc")
        assert total_reported == len(chunks)

    def test_custom_separators(self, sample_document):
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=10,
            separators=["\n\n", "\n", " "],
        )
        chunks = chunker.split([sample_document])
        assert len(chunks) >= 1

    def test_overlap_creates_shared_content(self):
        """Adjacent chunks should share some words when overlap > 0."""
        text = " ".join([f"word{i}" for i in range(500)])
        doc = Document(page_content=text, metadata={})
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=25)
        chunks = chunker.split([doc])

        if len(chunks) >= 2:
            words_in_first = set(chunks[0].page_content.split())
            words_in_second = set(chunks[1].page_content.split())
            shared = words_in_first & words_in_second
            assert len(shared) > 0, "Expected some word overlap between consecutive chunks"


# ---------------------------------------------------------------------------
# SemanticChunker tests
# ---------------------------------------------------------------------------


def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embedding function for testing."""
    import hashlib

    result = []
    for text in texts:
        # Create a 4-dim vector based on text hash
        h = hashlib.md5(text.encode()).hexdigest()
        vec = [int(h[i * 8 : (i + 1) * 8], 16) / (16 ** 8) for i in range(4)]
        result.append(vec)
    return result


class TestSemanticChunker:
    def test_splits_multi_sentence_document(self, sample_document):
        chunker = SemanticChunker(
            embed_fn=_mock_embed_fn,
            chunk_size=512,
            chunk_overlap=64,
            threshold_percentile=50.0,
        )
        chunks = chunker.split([sample_document])
        assert len(chunks) >= 1

    def test_short_document_returns_single_chunk(self, single_sentence_document):
        chunker = SemanticChunker(
            embed_fn=_mock_embed_fn,
            chunk_size=512,
        )
        chunks = chunker.split([single_sentence_document])
        assert len(chunks) == 1

    def test_metadata_preserved(self, sample_document):
        chunker = SemanticChunker(embed_fn=_mock_embed_fn)
        chunks = chunker.split([sample_document])
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_strategy_recorded_in_metadata(self, sample_document):
        chunker = SemanticChunker(embed_fn=_mock_embed_fn)
        chunks = chunker.split([sample_document])
        for chunk in chunks:
            assert chunk.metadata.get("chunk_strategy") == "semantic"

    def test_empty_document(self, empty_document):
        chunker = SemanticChunker(embed_fn=_mock_embed_fn)
        chunks = chunker.split([empty_document])
        assert len(chunks) == 0

    def test_sentence_tokenizer_splits_correctly(self):
        text = "This is sentence one. This is sentence two. And this is three!"
        sentences = SemanticChunker._sentence_tokenize(text)
        assert len(sentences) >= 1
        # At least some sentences should be non-empty
        assert all(len(s) > 0 for s in sentences)

    def test_cosine_distances_between_identical_vectors(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        distances = SemanticChunker._cosine_distances([vec, vec])
        assert len(distances) == 1
        assert abs(distances[0]) < 1e-6  # identical vectors → 0 distance

    def test_cosine_distances_between_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        distances = SemanticChunker._cosine_distances([a, b])
        assert abs(distances[0] - 1.0) < 1e-6  # orthogonal → max distance


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateChunker:
    def test_recursive_factory(self):
        settings = ChunkingSettings()
        settings.strategy = "recursive"
        chunker = create_chunker(settings)
        assert isinstance(chunker, RecursiveChunker)

    def test_semantic_factory_with_embed_fn(self):
        settings = ChunkingSettings()
        settings.strategy = "semantic"
        chunker = create_chunker(settings, embed_fn=_mock_embed_fn)
        assert isinstance(chunker, SemanticChunker)

    def test_semantic_factory_without_embed_fn_raises(self):
        settings = ChunkingSettings()
        settings.strategy = "semantic"
        with pytest.raises(ValueError, match="embed_fn"):
            create_chunker(settings, embed_fn=None)

    def test_factory_passes_chunk_size(self):
        settings = ChunkingSettings()
        settings.strategy = "recursive"
        settings.chunk_size = 256
        settings.chunk_overlap = 32
        chunker = create_chunker(settings)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32
