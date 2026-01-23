"""
Pydantic v2 request and response schemas for the RAG Research Assistant API.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The natural language question to answer.",
        examples=["What are the key contributions of the attention mechanism?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve before reranking.",
    )
    template: str = Field(
        default="rag_qa",
        description="Prompt template to use. One of: rag_qa, multi_hop_qa, summarize.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional ChromaDB metadata filters (e.g., {\"source\": \"paper.pdf\"}).",
    )
    stream: bool = Field(
        default=False,
        description="If true, stream the answer as server-sent events.",
    )
    conversation_history: list[dict[str, str]] | None = Field(
        default=None,
        description="Prior conversation turns for multi-turn mode.",
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        valid = {"rag_qa", "multi_hop_qa", "summarize", "conversational_rag", "factual_verify"}
        if v not in valid:
            raise ValueError(f"template must be one of {valid}")
        return v


class IngestRequest(BaseModel):
    """Request body for URL/text ingestion via the /ingest endpoint."""

    text: str | None = Field(
        default=None,
        description="Raw text to ingest directly.",
    )
    source_name: str = Field(
        default="uploaded_text",
        description="Logical source identifier (e.g., filename or URL).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata to attach to all chunks from this source.",
    )
    chunk_size: int | None = Field(
        default=None,
        ge=64,
        le=4096,
        description="Override the default chunk size for this ingestion.",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        description="Override the default chunk overlap.",
    )


class DeleteSourceRequest(BaseModel):
    """Request body for deleting a source from the vector store."""

    source: str = Field(
        ...,
        description="The source identifier to delete (e.g., file path).",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CitationResponse(BaseModel):
    """A single source citation in the API response."""

    index: int
    source: str
    page: int | None = None
    title: str | None = None
    doi: str | None = None
    text_snippet: str


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    query: str
    answer: str
    answer_with_citations: str
    citations: list[CitationResponse]
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    context_chunks_used: int
    context_tokens: int
    is_grounded: bool
    model: str
    total_tokens: int
    latency_ms: float
    template: str

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What is the transformer attention mechanism?",
            "answer": "The attention mechanism [1] allows models to weigh the importance "
                      "of different parts of the input sequence when producing an output.",
            "answer_with_citations": "...",
            "citations": [
                {"index": 1, "source": "attention.pdf", "page": 3, "title": "Attention Is All You Need"}
            ],
            "sources": ["attention.pdf"],
            "confidence": 0.92,
            "context_chunks_used": 5,
            "context_tokens": 1847,
            "is_grounded": True,
            "model": "gpt-4o-mini",
            "total_tokens": 2341,
            "latency_ms": 1230.5,
            "template": "rag_qa",
        }
    }}


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""

    success: bool
    message: str
    source: str
    chunks_added: int
    doc_ids: list[str]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str  # "healthy" or "degraded"
    version: str
    vector_store_docs: int
    llm_backend: str
    llm_model: str
    embedding_backend: str
    embedding_model: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Response from the /stats endpoint."""

    collection_name: str
    total_documents: int
    unique_sources: int
    sources: list[str]
    embedding_dimension: int
    distance_function: str


class DeleteResponse(BaseModel):
    """Response from the /delete endpoint."""

    success: bool
    source: str
    chunks_deleted: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    status_code: int
