"""
FastAPI application entry point for the RAG Research Assistant.

Initializes all pipeline components on startup, configures middleware
(CORS, logging, error handling), and mounts the API router.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
or:
    python api/main.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings

# ---------------------------------------------------------------------------
# Logging setup (must be done before other imports)
# ---------------------------------------------------------------------------

settings = get_settings()


def _configure_logging() -> None:
    """Configure structlog for JSON or console output."""
    import logging

    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, stream=sys.stdout)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.logging.format == "json":
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


_configure_logging()
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline state
# ---------------------------------------------------------------------------

pipeline: "RAGPipeline | None" = None  # type: ignore[name-defined]
startup_time: float = time.perf_counter()


# ---------------------------------------------------------------------------
# RAG Pipeline orchestrator
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    Top-level orchestrator that wires all RAG components together.

    Holds references to:
    - embedder: Embedding model (OpenAI or SentenceTransformers)
    - vector_store: ChromaDB wrapper
    - bm25_index: In-memory BM25 index
    - retriever: Hybrid retriever
    - reranker: Cross-encoder reranker
    - context_builder: Context assembly
    - generator: LLM response generator
    """

    def __init__(self) -> None:
        from config.settings import get_settings

        self.settings = get_settings()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all pipeline components in dependency order."""
        import asyncio

        logger.info("pipeline.initializing")

        # 1. Embedding model
        from src.embeddings.embedding_manager import create_embedder

        self.embedder = create_embedder(
            settings=self.settings.embedding,
            use_cache=True,
            openai_api_key=self.settings.llm.openai_api_key,
        )
        logger.info("pipeline.embedder_ready", model=self.embedder.model_name)

        # 2. Vector store
        from src.embeddings.vector_store import VectorStore

        self.vector_store = VectorStore(
            settings=self.settings.chroma,
            embedder=self.embedder,
        )
        logger.info(
            "pipeline.vector_store_ready",
            docs=self.vector_store.get_stats().document_count,
        )

        # 3. BM25 index (built from all existing vector store documents)
        from src.retrieval.retriever import BM25Index

        existing_docs = self._load_all_documents()
        self.bm25_index = BM25Index(documents=existing_docs)
        logger.info("pipeline.bm25_ready", doc_count=len(existing_docs))

        # 4. Hybrid retriever
        from src.retrieval.retriever import HybridRetriever

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            settings=self.settings.retrieval,
        )

        # 5. Reranker
        from src.retrieval.reranker import create_reranker

        self.reranker = create_reranker(
            settings=self.settings.embedding,
            enable_reranking=self.settings.retrieval.enable_reranking,
        )

        # 6. Context builder
        from src.retrieval.context_builder import ContextBuilder

        self.context_builder = ContextBuilder(max_context_tokens=3500)

        # 7. LLM client
        from src.generation.llm_client import create_llm_client

        self.llm_client = create_llm_client(self.settings.llm)

        # 8. Response generator
        from src.generation.response_generator import ResponseGenerator

        self.generator = ResponseGenerator(
            llm_client=self.llm_client,
            temperature=self.settings.llm.temperature,
            max_tokens=self.settings.llm.max_tokens,
        )

        logger.info("pipeline.ready", llm=self.llm_client.model_name)

    def _load_all_documents(self):
        """Load all documents from vector store for BM25 index construction."""
        from langchain_core.documents import Document

        try:
            raw = self.vector_store._collection.get(include=["documents", "metadatas"])
            docs = []
            for text, meta in zip(
                raw.get("documents", []) or [],
                raw.get("metadatas", []) or [],
            ):
                if text:
                    docs.append(Document(page_content=text, metadata=meta or {}))
            return docs
        except Exception as exc:
            logger.warning("pipeline.bm25_load_failed", error=str(exc))
            return []

    async def aquery(
        self,
        query: str,
        top_k: int = 5,
        template_name: str = "rag_qa",
        metadata_filter: dict | None = None,
        conversation_history: list | None = None,
    ):
        """Async end-to-end RAG query."""
        loop = asyncio.get_event_loop()

        # Retrieval (CPU-bound in thread pool)
        context = await loop.run_in_executor(
            None,
            lambda: self._retrieve_context_sync(query, top_k, metadata_filter),
        )

        # Generation (I/O-bound, truly async)
        return await self.generator.agenerate(
            query=query,
            context=context,
            template_name=template_name,
            conversation_history=conversation_history,
        )

    async def aretrieve_context(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        """Async context retrieval (for streaming)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._retrieve_context_sync(query, top_k, metadata_filter),
        )

    def _retrieve_context_sync(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None,
    ):
        """Synchronous retrieval pipeline (runs in thread pool)."""
        # Hybrid retrieval
        hybrid_results = self.retriever.retrieve(
            query=query,
            k=self.settings.retrieval.top_k,
            metadata_filter=metadata_filter,
        )

        # Reranking
        reranked = self.reranker.rerank_hybrid_results(
            query=query,
            hybrid_results=hybrid_results,
            top_k=top_k,
        )

        # Context assembly
        return self.context_builder.build(query=query, results=reranked)

    async def aingest_documents(
        self,
        documents: list,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Async document ingestion: chunk → embed → store."""
        from src.ingestion.text_splitter import RecursiveChunker
        from src.ingestion.preprocessor import preprocess_documents

        chunker = RecursiveChunker(
            chunk_size=chunk_size or self.settings.chunking.chunk_size,
            chunk_overlap=chunk_overlap or self.settings.chunking.chunk_overlap,
        )

        loop = asyncio.get_event_loop()

        def _ingest():
            chunks = chunker.split(documents)
            processed = preprocess_documents(chunks, deduplicate=True)
            doc_ids = self.vector_store.add_documents(processed)
            self.bm25_index.update(processed)
            return doc_ids

        return await loop.run_in_executor(None, _ingest)

    async def aingest_pdf_bytes(self, data: bytes, filename: str) -> list[str]:
        """Async PDF bytes ingestion."""
        from src.ingestion.pdf_loader import PDFLoader

        loader = PDFLoader()

        loop = asyncio.get_event_loop()

        def _load_and_ingest():
            pdf_doc = loader.load_bytes(data, filename=filename)
            return pdf_doc.pages

        pages = await loop.run_in_executor(None, _load_and_ingest)
        return await self.aingest_documents(pages)

    async def rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index from current vector store contents."""
        from src.retrieval.retriever import BM25Index

        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, self._load_all_documents)
        self.bm25_index = BM25Index(documents=docs)
        self.retriever.bm25_index = self.bm25_index
        logger.info("pipeline.bm25_rebuilt", doc_count=len(docs))


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and teardown the RAG pipeline on app start/stop."""
    global pipeline, startup_time

    startup_time = time.perf_counter()
    logger.info("server.startup", version="1.0.0")

    try:
        pipeline = RAGPipeline()
        logger.info("server.pipeline_ready")
    except Exception as exc:
        logger.error("server.startup_failed", error=str(exc), exc_info=True)
        # Allow app to start even if pipeline fails — health check will report degraded
        pipeline = None

    yield

    logger.info("server.shutdown")
    # Cleanup (ChromaDB handles its own persistence)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Research Assistant API",
    description=(
        "A production-quality Retrieval-Augmented Generation API for "
        "intelligent research assistance over document collections. "
        "Supports hybrid retrieval, cross-encoder reranking, and "
        "multi-backend LLM generation."
    ),
    version="1.0.0",
    docs_url=settings.api.docs_url,
    redoc_url=settings.api.redoc_url,
    lifespan=lifespan,
    contact={
        "name": "RAG Research Team",
        "url": "https://github.com/yourusername/rag-research-assistant",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request logging
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their response times."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 1),
        client=request.client.host if request.client else "unknown",
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a clean JSON error for unhandled exceptions."""
    logger.error(
        "http.unhandled_exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred.",
            "status_code": 500,
        },
    )


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

from api.routes import router  # noqa: E402

app.include_router(router, prefix="/api/v1", tags=["RAG"])


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API docs."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Entry point for `rag-serve` CLI command."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_config=None,  # Structlog handles logging
        access_log=False,  # Our middleware handles access logs
    )


if __name__ == "__main__":
    run_server()
