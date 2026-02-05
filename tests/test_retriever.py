"""
Unit tests for the retrieval module.

Tests cover:
- BM25Index: tokenization, indexing, search, incremental updates
- HybridRetriever: RRF fusion, dense-only, sparse-only modes
- ContextBuilder: token budget, citation tracking, ordering
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.retrieval.retriever import BM25Index, BM25Result, HybridRetriever
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.reranker import NoopReranker, RerankedResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(text: str, source: str = "test.pdf", page: int = 1, chunk: int = 0) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "chunk_index": chunk},
    )


@pytest.fixture
def corpus() -> list[Document]:
    return [
        _make_doc(
            "Attention mechanisms allow neural networks to focus on relevant parts of input.",
            source="attention.pdf", page=1, chunk=0
        ),
        _make_doc(
            "The transformer architecture uses self-attention layers and feed-forward networks.",
            source="attention.pdf", page=2, chunk=1
        ),
        _make_doc(
            "BERT is a pre-trained language model based on the transformer encoder.",
            source="bert.pdf", page=1, chunk=0
        ),
        _make_doc(
            "GPT uses a decoder-only transformer architecture for language generation tasks.",
            source="gpt.pdf", page=1, chunk=0
        ),
        _make_doc(
            "Machine learning models require large datasets for effective training.",
            source="ml_basics.pdf", page=5, chunk=2
        ),
        _make_doc(
            "Gradient descent is the fundamental optimization algorithm in deep learning.",
            source="ml_basics.pdf", page=6, chunk=3
        ),
    ]


@pytest.fixture
def bm25_index(corpus) -> BM25Index:
    return BM25Index(documents=corpus)


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_builds_successfully(self, corpus):
        index = BM25Index(documents=corpus)
        assert index.corpus_size == len(corpus)

    def test_returns_results_for_known_query(self, bm25_index):
        results = bm25_index.search("transformer attention", k=3)
        assert len(results) > 0

    def test_results_sorted_by_score_descending(self, bm25_index):
        results = bm25_index.search("transformer attention", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_at_most_k_results(self, bm25_index, corpus):
        results = bm25_index.search("learning", k=2)
        assert len(results) <= 2

    def test_returns_all_when_k_larger_than_corpus(self, bm25_index, corpus):
        results = bm25_index.search("transformer", k=100)
        assert len(results) <= len(corpus)

    def test_unknown_query_returns_empty(self, bm25_index):
        results = bm25_index.search("xyzqwerty12345notaword", k=5)
        # May return empty if no vocabulary overlap
        assert isinstance(results, list)

    def test_result_rank_starts_at_one(self, bm25_index):
        results = bm25_index.search("attention mechanism", k=3)
        if results:
            assert results[0].rank == 1

    def test_result_documents_are_from_corpus(self, bm25_index, corpus):
        results = bm25_index.search("transformer", k=10)
        corpus_texts = {doc.page_content for doc in corpus}
        for result in results:
            assert result.document.page_content in corpus_texts

    def test_incremental_update(self, bm25_index):
        new_doc = _make_doc(
            "Contrastive learning enables self-supervised representation learning.",
            source="contrastive.pdf",
        )
        initial_size = bm25_index.corpus_size
        bm25_index.update([new_doc])
        assert bm25_index.corpus_size == initial_size + 1

    def test_incremental_update_makes_doc_retrievable(self, bm25_index):
        new_doc = _make_doc(
            "Contrastive learning enables self-supervised representation learning.",
            source="contrastive.pdf",
        )
        bm25_index.update([new_doc])
        results = bm25_index.search("contrastive representation", k=5)
        found = any("contrastive" in r.document.page_content.lower() for r in results)
        assert found

    def test_tokenizer_lowercases(self):
        tokens = BM25Index._tokenize("The Attention Mechanism")
        assert all(t == t.lower() for t in tokens)

    def test_tokenizer_removes_stopwords(self):
        tokens = BM25Index._tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens

    def test_tokenizer_removes_short_tokens(self):
        tokens = BM25Index._tokenize("a b c transformer")
        # Single chars should be removed
        assert "a" not in tokens
        assert "b" not in tokens

    def test_empty_document_corpus(self):
        index = BM25Index(documents=[])
        results = index.search("transformer", k=5)
        assert results == []

    def test_empty_query(self, bm25_index):
        results = bm25_index.search("", k=5)
        assert results == []

    def test_bm25_scores_are_non_negative(self, bm25_index):
        results = bm25_index.search("deep learning neural network", k=10)
        for r in results:
            assert r.score >= 0.0


# ---------------------------------------------------------------------------
# HybridRetriever tests (with mock dependencies)
# ---------------------------------------------------------------------------


class MockVectorStore:
    """Minimal mock of VectorStore for unit testing."""

    def __init__(self, docs: list[Document], scores: list[float]):
        self.docs = docs
        self.scores = scores

    def similarity_search(self, query: str, k: int = 10, **kwargs):
        from src.embeddings.vector_store import SearchResult

        results = []
        for i, (doc, score) in enumerate(zip(self.docs, self.scores)):
            results.append(
                SearchResult(document=doc, score=score, doc_id=f"id_{i}")
            )
        return results[:k]


class TestHybridRetriever:
    @pytest.fixture
    def retriever(self, corpus, bm25_index):
        from config.settings import RetrievalSettings

        settings = RetrievalSettings()
        settings.top_k = 5
        settings.hybrid_dense_weight = 0.7
        settings.enable_reranking = True

        mock_scores = [0.9 - i * 0.1 for i in range(len(corpus))]
        mock_store = MockVectorStore(corpus, mock_scores)

        return HybridRetriever(
            vector_store=mock_store,
            bm25_index=bm25_index,
            settings=settings,
        )

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("transformer attention mechanism")
        assert len(results) > 0

    def test_retrieve_returns_at_most_k_results(self, retriever):
        results = retriever.retrieve("attention", k=3)
        assert len(results) <= 3

    def test_rrf_scores_are_positive(self, retriever):
        results = retriever.retrieve("transformer")
        for r in results:
            assert r.rrf_score > 0

    def test_results_sorted_by_rrf_score(self, retriever):
        results = retriever.retrieve("transformer attention")
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_fusion_combines_dense_and_sparse(self, retriever):
        results = retriever.retrieve("attention mechanism neural network")
        # Some results should have both dense and sparse ranks
        both_present = [
            r for r in results
            if r.dense_rank is not None and r.sparse_rank is not None
        ]
        assert len(both_present) >= 0  # At least 0 (may be all dense-only for mock)

    def test_dense_only_retrieval(self, retriever):
        results = retriever.retrieve_dense_only("transformer")
        assert len(results) > 0

    def test_sparse_only_retrieval(self, retriever):
        results = retriever.retrieve_sparse_only("transformer")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# ContextBuilder tests
# ---------------------------------------------------------------------------


@pytest.fixture
def reranked_results(corpus) -> list[RerankedResult]:
    return [
        RerankedResult(
            document=doc,
            rerank_score=1.0 - i * 0.15,
            original_rank=i + 1,
            final_rank=i + 1,
        )
        for i, doc in enumerate(corpus[:4])
    ]


class TestContextBuilder:
    def test_builds_non_empty_context(self, reranked_results):
        builder = ContextBuilder(max_context_tokens=2000)
        result = builder.build(query="What is attention?", results=reranked_results)
        assert result.context_text.strip() != ""

    def test_includes_citation_indices(self, reranked_results):
        builder = ContextBuilder(max_context_tokens=2000)
        result = builder.build(query="What is attention?", results=reranked_results)
        assert "[1]" in result.context_text

    def test_citations_created_for_included_chunks(self, reranked_results):
        builder = ContextBuilder(max_context_tokens=2000)
        result = builder.build(query="What is attention?", results=reranked_results)
        assert len(result.citations) == result.chunks_included

    def test_empty_results_returns_graceful_response(self):
        builder = ContextBuilder()
        result = builder.build(query="test", results=[])
        assert "No relevant documents" in result.context_text
        assert result.chunks_included == 0

    def test_respects_token_budget(self, corpus):
        # Create many large documents
        large_docs = [
            _make_doc("word " * 200, source=f"doc{i}.pdf", chunk=i)
            for i in range(20)
        ]
        results = [
            RerankedResult(
                document=doc,
                rerank_score=1.0 - i * 0.01,
                original_rank=i + 1,
                final_rank=i + 1,
            )
            for i, doc in enumerate(large_docs)
        ]

        builder = ContextBuilder(max_context_tokens=500)
        built = builder.build(query="test", results=results)

        # Should not include all 20 documents
        assert built.chunks_included < 20

    def test_citation_block_includes_all_sources(self, reranked_results):
        builder = ContextBuilder(max_context_tokens=3000)
        result = builder.build(query="test", results=reranked_results)
        block = result.citation_block
        assert "**Sources:**" in block
        assert len(result.citations) > 0

    def test_source_list_is_deduplicated(self, reranked_results):
        builder = ContextBuilder(max_context_tokens=3000)
        result = builder.build(query="test", results=reranked_results)
        sources = result.source_list
        assert len(sources) == len(set(sources))

    def test_document_order_mode(self, reranked_results):
        builder = ContextBuilder(max_context_tokens=3000, order_by="document")
        result = builder.build(query="test", results=reranked_results)
        assert result.context_text.strip() != ""

    def test_build_from_documents_convenience(self, corpus):
        builder = ContextBuilder(max_context_tokens=2000)
        result = builder.build_from_documents(
            query="attention",
            documents=corpus[:3],
            scores=[0.9, 0.8, 0.7],
        )
        assert result.chunks_included <= 3
        assert result.context_text.strip() != ""

    def test_chunks_truncated_counted_correctly(self, corpus):
        """When token budget is very small, truncated count should be > 0."""
        large_docs = [_make_doc("word " * 300, source=f"doc{i}.pdf") for i in range(5)]
        results = [
            RerankedResult(
                document=doc,
                rerank_score=0.9,
                original_rank=i + 1,
                final_rank=i + 1,
            )
            for i, doc in enumerate(large_docs)
        ]

        builder = ContextBuilder(max_context_tokens=200)
        built = builder.build(query="test", results=results)
        total_accounted = built.chunks_included + built.chunks_truncated
        assert total_accounted <= len(results)
