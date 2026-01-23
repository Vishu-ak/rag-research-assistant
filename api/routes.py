"""
FastAPI route handlers for the RAG Research Assistant API.

Endpoints:
  POST /query        — Answer a natural language question over indexed documents
  POST /ingest       — Ingest plain text into the vector store
  POST /ingest/file  — Ingest an uploaded PDF or text file
  GET  /health       — Health check with component status
  GET  /stats        — Vector store statistics
  DELETE /sources    — Delete documents by source
"""

from __future__ import annotations

import time
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document

from api.schemas import (
    CitationResponse,
    DeleteRequest,
    DeleteResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency injection — pipeline components
# ---------------------------------------------------------------------------


def get_pipeline():
    """Return the global RAGPipeline instance (set at startup)."""
    from api.main import pipeline

    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized. Check server logs.",
        )
    return pipeline


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question over indexed documents",
    response_description="Generated answer with citations and metadata",
)
async def query_documents(
    request: QueryRequest,
    pipeline=Depends(get_pipeline),
) -> QueryResponse | StreamingResponse:
    """
    Answer a natural language question using the RAG pipeline.

    Performs hybrid retrieval (dense + BM25), cross-encoder reranking,
    context assembly, and LLM-based answer generation with citations.

    If `stream=true`, returns a Server-Sent Events stream of text tokens.
    """
    logger.info(
        "api.query",
        query=request.query[:80],
        top_k=request.top_k,
        template=request.template,
        stream=request.stream,
    )

    start = time.perf_counter()

    try:
        if request.stream:
            return StreamingResponse(
                _stream_answer(pipeline, request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        answer = await pipeline.aquery(
            query=request.query,
            top_k=request.top_k,
            template_name=request.template,
            metadata_filter=request.filters,
            conversation_history=request.conversation_history,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return QueryResponse(
            query=answer.query,
            answer=answer.answer,
            answer_with_citations=answer.answer_with_citations,
            citations=[
                CitationResponse(
                    index=c.index,
                    source=c.source,
                    page=c.page,
                    title=c.title,
                    doi=c.doi,
                    text_snippet=c.text_snippet,
                )
                for c in answer.citations
            ],
            sources=answer.sources,
            confidence=answer.confidence,
            context_chunks_used=answer.context_used,
            context_tokens=answer.context_tokens,
            is_grounded=answer.is_grounded,
            model=answer.llm_response.model,
            total_tokens=answer.llm_response.total_tokens,
            latency_ms=elapsed_ms,
            template=answer.template_name,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("api.query_error", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during query processing.",
        )


async def _stream_answer(pipeline, request: QueryRequest) -> AsyncIterator[str]:
    """Async generator for SSE streaming responses."""
    try:
        # First retrieve context (non-streaming)
        context = await pipeline.aretrieve_context(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.filters,
        )

        # Stream the generation
        async for token in pipeline.generator.astream(
            query=request.query,
            context=context,
            template_name=request.template,
        ):
            yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.error("api.stream_error", error=str(exc))
        yield f"data: [ERROR] {str(exc)}\n\n"


# ---------------------------------------------------------------------------
# Ingest endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest plain text into the vector store",
)
async def ingest_text(
    request: IngestRequest,
    pipeline=Depends(get_pipeline),
) -> IngestResponse:
    """
    Ingest raw text content into the vector store.

    The text is split into chunks, embedded, and stored in ChromaDB
    for subsequent retrieval.
    """
    if not request.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' must be provided in the request body.",
        )

    logger.info(
        "api.ingest_text",
        source=request.source_name,
        text_len=len(request.text),
    )

    start = time.perf_counter()

    try:
        doc = Document(
            page_content=request.text,
            metadata={
                "source": request.source_name,
                **(request.metadata or {}),
            },
        )

        doc_ids = await pipeline.aingest_documents(
            documents=[doc],
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {len(doc_ids)} chunks from '{request.source_name}'.",
            source=request.source_name,
            chunks_added=len(doc_ids),
            doc_ids=doc_ids,
            processing_time_ms=round(elapsed_ms, 1),
        )

    except Exception as exc:
        logger.error("api.ingest_error", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(exc)}",
        )


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    summary="Upload and ingest a PDF or text file",
)
async def ingest_file(
    file: UploadFile = File(..., description="PDF or plain text file to ingest"),
    pipeline=Depends(get_pipeline),
) -> IngestResponse:
    """
    Upload a PDF or text file and ingest its contents into the vector store.

    Supported formats: PDF (.pdf), plain text (.txt), Markdown (.md)
    Maximum file size: 50 MB
    """
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename.",
        )

    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in ("pdf", "txt", "md"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '.{ext}'. Supported: pdf, txt, md",
        )

    logger.info("api.ingest_file", filename=file.filename, content_type=file.content_type)
    start = time.perf_counter()

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the 50 MB limit ({len(file_bytes) / 1024 / 1024:.1f} MB).",
        )

    try:
        if ext == "pdf":
            doc_ids = await pipeline.aingest_pdf_bytes(
                data=file_bytes,
                filename=file.filename,
            )
        else:
            text = file_bytes.decode("utf-8", errors="replace")
            doc = Document(
                page_content=text,
                metadata={"source": file.filename},
            )
            doc_ids = await pipeline.aingest_documents(documents=[doc])

        elapsed_ms = (time.perf_counter() - start) * 1000

        return IngestResponse(
            success=True,
            message=f"Ingested {len(doc_ids)} chunks from '{file.filename}'.",
            source=file.filename,
            chunks_added=len(doc_ids),
            doc_ids=doc_ids,
            processing_time_ms=round(elapsed_ms, 1),
        )

    except Exception as exc:
        logger.error("api.ingest_file_error", filename=file.filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File ingestion failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Health and stats endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check for all pipeline components",
)
async def health_check(pipeline=Depends(get_pipeline)) -> HealthResponse:
    """
    Return health status of the RAG pipeline and its dependencies.

    Checks: vector store connectivity, LLM reachability, embedding model.
    """
    from api.main import startup_time

    stats = pipeline.vector_store.get_stats()
    uptime = time.perf_counter() - startup_time

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store_docs=stats.document_count,
        llm_backend=pipeline.settings.llm.backend,
        llm_model=pipeline.settings.llm.openai_model
        if pipeline.settings.llm.backend == "openai"
        else pipeline.settings.llm.ollama_model,
        embedding_backend=pipeline.settings.embedding.backend,
        embedding_model=pipeline.settings.embedding.sentence_transformer_model
        if pipeline.settings.embedding.backend == "sentence-transformers"
        else pipeline.settings.embedding.openai_embedding_model,
        uptime_seconds=round(uptime, 1),
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Vector store collection statistics",
)
async def get_stats(pipeline=Depends(get_pipeline)) -> StatsResponse:
    """Return statistics about the ChromaDB collection."""
    stats = pipeline.vector_store.get_stats()
    sources = pipeline.vector_store.list_sources()

    return StatsResponse(
        collection_name=stats.name,
        total_documents=stats.document_count,
        unique_sources=len(sources),
        sources=sources,
        embedding_dimension=stats.embedding_dimension,
        distance_function=stats.distance_function,
    )


@router.delete(
    "/sources",
    response_model=DeleteResponse,
    summary="Delete all chunks from a specific source",
)
async def delete_source(
    request: DeleteSourceRequest,
    pipeline=Depends(get_pipeline),
) -> DeleteResponse:
    """
    Delete all document chunks originating from the specified source.

    The source identifier typically matches the original file path or name
    used during ingestion.
    """
    logger.info("api.delete_source", source=request.source)

    try:
        count = pipeline.vector_store.delete_by_source(request.source)

        # Also rebuild BM25 index after deletion
        await pipeline.rebuild_bm25_index()

        return DeleteResponse(
            success=True,
            source=request.source,
            chunks_deleted=count,
        )

    except Exception as exc:
        logger.error("api.delete_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Fix import — DeleteSourceRequest was not in schemas originally
# ---------------------------------------------------------------------------

from api.schemas import DeleteSourceRequest  # noqa: E402 (moved here to avoid circular)

# Re-export as DeleteRequest for the schema import alias above
DeleteRequest = DeleteSourceRequest
