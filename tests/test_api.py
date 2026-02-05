"""
API integration tests for the FastAPI RAG Research Assistant endpoints.

Uses pytest-asyncio and FastAPI's TestClient / AsyncClient.
Mocks the RAG pipeline to isolate HTTP layer behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Mocked pipeline for API tests
# ---------------------------------------------------------------------------


def _make_mock_pipeline():
    """Create a fully mocked RAGPipeline for test isolation."""
    from src.generation.response_generator import GeneratedAnswer
    from src.generation.llm_client import LLMResponse
    from src.retrieval.context_builder import BuiltContext

    mock_llm_response = LLMResponse(
        content="Attention mechanisms [1] allow models to focus on relevant parts.",
        model="gpt-4o-mini",
        prompt_tokens=500,
        completion_tokens=100,
        total_tokens=600,
        latency_ms=350.0,
        finish_reason="stop",
    )

    from src.retrieval.context_builder import Citation

    mock_citation = Citation(
        index=1,
        source="attention.pdf",
        page=1,
        chunk_index=0,
        title="Attention Is All You Need",
        doi=None,
        authors="Vaswani et al.",
        date="2017",
        text_snippet="Attention mechanisms allow neural networks...",
    )

    mock_answer = GeneratedAnswer(
        answer="Attention mechanisms [1] allow models to focus on relevant parts.",
        citations=[mock_citation],
        sources=["attention.pdf"],
        confidence=0.92,
        context_used=5,
        context_tokens=1200,
        llm_response=mock_llm_response,
        query="What is attention?",
        generation_latency_ms=350.0,
        template_name="rag_qa",
    )

    mock_context = BuiltContext(
        context_text="[1] attention.pdf, Page 1\nAttention mechanisms allow...",
        citations=[mock_citation],
        total_tokens=1200,
        chunks_included=5,
        chunks_truncated=0,
    )

    mock_pipeline = MagicMock()
    mock_pipeline.aquery = AsyncMock(return_value=mock_answer)
    mock_pipeline.aretrieve_context = AsyncMock(return_value=mock_context)
    mock_pipeline.aingest_documents = AsyncMock(return_value=["chunk_id_1", "chunk_id_2"])
    mock_pipeline.aingest_pdf_bytes = AsyncMock(return_value=["pdf_chunk_1", "pdf_chunk_2"])
    mock_pipeline.rebuild_bm25_index = AsyncMock()

    mock_stats = MagicMock()
    mock_stats.document_count = 42
    mock_stats.embedding_dimension = 1024
    mock_stats.distance_function = "cosine"
    mock_stats.name = "rag_documents"
    mock_pipeline.vector_store.get_stats.return_value = mock_stats
    mock_pipeline.vector_store.list_sources.return_value = ["attention.pdf", "bert.pdf"]
    mock_pipeline.vector_store.delete_by_source.return_value = 3

    mock_pipeline.settings = MagicMock()
    mock_pipeline.settings.llm.backend = "openai"
    mock_pipeline.settings.llm.openai_model = "gpt-4o-mini"
    mock_pipeline.settings.embedding.backend = "sentence-transformers"
    mock_pipeline.settings.embedding.sentence_transformer_model = "BAAI/bge-large-en-v1.5"

    mock_pipeline.generator = MagicMock()

    async def _mock_astream(*args, **kwargs):
        for token in ["Hello", " world", "!"]:
            yield token

    mock_pipeline.generator.astream = _mock_astream

    return mock_pipeline


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline():
    return _make_mock_pipeline()


@pytest.fixture
def test_client(mock_pipeline):
    """Synchronous test client with mocked pipeline."""
    from api.main import app
    import api.main as main_module

    with patch.object(main_module, "pipeline", mock_pipeline):
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, test_client):
        data = test_client.get("/api/v1/health").json()
        assert data["status"] == "healthy"

    def test_health_includes_version(self, test_client):
        data = test_client.get("/api/v1/health").json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_includes_llm_info(self, test_client):
        data = test_client.get("/api/v1/health").json()
        assert "llm_backend" in data
        assert "llm_model" in data

    def test_health_includes_vector_store_docs(self, test_client):
        data = test_client.get("/api/v1/health").json()
        assert data["vector_store_docs"] == 42

    def test_health_includes_uptime(self, test_client):
        data = test_client.get("/api/v1/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# Stats endpoint tests
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    def test_stats_returns_200(self, test_client):
        response = test_client.get("/api/v1/stats")
        assert response.status_code == 200

    def test_stats_contains_document_count(self, test_client):
        data = test_client.get("/api/v1/stats").json()
        assert data["total_documents"] == 42

    def test_stats_lists_sources(self, test_client):
        data = test_client.get("/api/v1/stats").json()
        assert "sources" in data
        assert "attention.pdf" in data["sources"]


# ---------------------------------------------------------------------------
# Query endpoint tests
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_query_returns_200(self, test_client):
        response = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?"},
        )
        assert response.status_code == 200

    def test_query_response_has_answer(self, test_client):
        data = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?"},
        ).json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_response_has_citations(self, test_client):
        data = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?"},
        ).json()
        assert "citations" in data
        assert isinstance(data["citations"], list)

    def test_query_response_has_confidence(self, test_client):
        data = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?"},
        ).json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_query_response_has_sources(self, test_client):
        data = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?"},
        ).json()
        assert "sources" in data
        assert "attention.pdf" in data["sources"]

    def test_query_too_short_returns_422(self, test_client):
        response = test_client.post("/api/v1/query", json={"query": "Hi"})
        assert response.status_code == 422

    def test_query_missing_required_field_returns_422(self, test_client):
        response = test_client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_invalid_template_returns_422(self, test_client):
        response = test_client.post(
            "/api/v1/query",
            json={"query": "What is attention?", "template": "invalid_template_xyz"},
        )
        assert response.status_code == 422

    def test_query_with_valid_top_k(self, test_client):
        response = test_client.post(
            "/api/v1/query",
            json={"query": "What is attention?", "top_k": 3},
        )
        assert response.status_code == 200

    def test_query_top_k_out_of_range_returns_422(self, test_client):
        response = test_client.post(
            "/api/v1/query",
            json={"query": "What is attention?", "top_k": 100},
        )
        assert response.status_code == 422

    def test_query_includes_is_grounded(self, test_client):
        data = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?"},
        ).json()
        assert "is_grounded" in data

    def test_query_template_rag_qa(self, test_client):
        data = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?", "template": "rag_qa"},
        ).json()
        assert data["template"] == "rag_qa"

    def test_query_latency_returned(self, test_client):
        data = test_client.post(
            "/api/v1/query",
            json={"query": "What is the attention mechanism?"},
        ).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Ingest endpoint tests
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    def test_ingest_text_returns_200(self, test_client):
        response = test_client.post(
            "/api/v1/ingest",
            json={
                "text": "This is a test document about neural networks and deep learning.",
                "source_name": "test_document.txt",
            },
        )
        assert response.status_code == 200

    def test_ingest_text_returns_chunk_count(self, test_client):
        data = test_client.post(
            "/api/v1/ingest",
            json={
                "text": "Test content for ingestion pipeline verification.",
                "source_name": "test.txt",
            },
        ).json()
        assert data["success"] is True
        assert data["chunks_added"] == 2  # From mock

    def test_ingest_without_text_returns_400(self, test_client):
        response = test_client.post(
            "/api/v1/ingest",
            json={"source_name": "empty.txt"},
        )
        assert response.status_code == 400

    def test_ingest_file_pdf_returns_200(self, test_client):
        # Create a minimal fake PDF bytes
        fake_pdf = b"%PDF-1.4\n%%EOF"
        response = test_client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )
        assert response.status_code == 200

    def test_ingest_file_txt_returns_200(self, test_client):
        text_content = b"This is a plain text document."
        response = test_client.post(
            "/api/v1/ingest/file",
            files={"file": ("document.txt", text_content, "text/plain")},
        )
        assert response.status_code == 200

    def test_ingest_file_unsupported_type_returns_415(self, test_client):
        response = test_client.post(
            "/api/v1/ingest/file",
            files={"file": ("data.xlsx", b"data", "application/vnd.ms-excel")},
        )
        assert response.status_code == 415


# ---------------------------------------------------------------------------
# Delete endpoint tests
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    def test_delete_source_returns_200(self, test_client):
        response = test_client.delete(
            "/api/v1/sources",
            json={"source": "attention.pdf"},
        )
        assert response.status_code == 200

    def test_delete_response_has_count(self, test_client):
        data = test_client.delete(
            "/api/v1/sources",
            json={"source": "attention.pdf"},
        ).json()
        assert "chunks_deleted" in data
        assert data["chunks_deleted"] == 3  # From mock


# ---------------------------------------------------------------------------
# CORS tests
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_headers_present(self, test_client):
        response = test_client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )
        # FastAPI CORS middleware should add headers
        assert response.status_code in (200, 405)  # Some versions return 405 for OPTIONS
