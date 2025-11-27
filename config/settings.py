"""
Centralized configuration management using Pydantic Settings.

All configuration is loaded from environment variables, with sensible defaults.
Use a .env file for local development (see .env.example).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider and model configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    backend: Literal["openai", "ollama"] = Field(
        default="openai",
        alias="LLM_BACKEND",
        description="Active LLM provider backend.",
    )

    # OpenAI
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="OpenAI API key.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_MODEL",
        description="OpenAI model identifier.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
        description="OpenAI embedding model identifier.",
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Base URL for the Ollama API.",
    )
    ollama_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_MODEL",
        description="Ollama model tag (e.g. 'llama3.2', 'mistral').",
    )

    # Generation hyperparameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32000)
    request_timeout: int = Field(default=120, ge=5, description="HTTP timeout in seconds.")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    backend: Literal["openai", "sentence-transformers"] = Field(
        default="sentence-transformers",
        alias="EMBEDDING_BACKEND",
    )
    sentence_transformer_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        alias="SENTENCE_TRANSFORMER_MODEL",
        description="HuggingFace model ID for SentenceTransformers.",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
        description="HuggingFace cross-encoder model for reranking.",
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for embedding generation.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device for local models ('cpu', 'cuda', 'mps').",
    )

    @field_validator("device", mode="before")
    @classmethod
    def auto_detect_device(cls, v: str) -> str:
        """Auto-detect CUDA or MPS if 'auto' is specified."""
        if v == "auto":
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    return "cuda"
                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v


class ChromaSettings(BaseSettings):
    """ChromaDB vector store configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    persist_dir: str = Field(
        default="./chroma_db",
        alias="CHROMA_PERSIST_DIR",
        description="Local directory for ChromaDB persistence.",
    )
    collection_name: str = Field(
        default="rag_documents",
        alias="CHROMA_COLLECTION_NAME",
    )
    host: str = Field(default="localhost", alias="CHROMA_HOST")
    port: int = Field(default=8000, alias="CHROMA_PORT")
    use_server: bool = Field(
        default=False,
        alias="CHROMA_USE_SERVER",
        description="Connect to a remote ChromaDB server instead of local persistence.",
    )
    distance_function: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance function for vector similarity.",
    )


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    top_k: int = Field(
        default=20,
        alias="RETRIEVAL_TOP_K",
        description="Candidates retrieved before reranking.",
    )
    rerank_top_k: int = Field(
        default=5,
        alias="RERANK_TOP_K",
        description="Final documents returned after reranking.",
    )
    hybrid_dense_weight: float = Field(
        default=0.7,
        alias="HYBRID_DENSE_WEIGHT",
        ge=0.0,
        le=1.0,
        description="Weight for dense retrieval score in RRF fusion.",
    )
    enable_reranking: bool = Field(
        default=True,
        description="Toggle cross-encoder reranking.",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR diversity parameter (0=diverse, 1=relevance-only).",
    )


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chunk_size: int = Field(
        default=512,
        alias="CHUNK_SIZE",
        ge=64,
        le=4096,
        description="Target chunk size in tokens.",
    )
    chunk_overlap: int = Field(
        default=64,
        alias="CHUNK_OVERLAP",
        ge=0,
        description="Overlap between adjacent chunks in tokens.",
    )
    strategy: Literal["recursive", "semantic"] = Field(
        default="recursive",
        alias="CHUNKING_STRATEGY",
    )
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Ordered list of separators for recursive splitting.",
    )


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8080, alias="API_PORT")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        alias="CORS_ORIGINS",
    )
    secret_key: str = Field(
        default="change-me-in-production",
        alias="API_SECRET_KEY",
    )
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated CORS origins from env string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


class LoggingSettings(BaseSettings):
    """Logging and observability configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )
    format: Literal["json", "console"] = Field(
        default="console",
        alias="LOG_FORMAT",
        description="'json' for structured production logs, 'console' for development.",
    )


class EvalSettings(BaseSettings):
    """Evaluation pipeline configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dataset_path: str = Field(
        default="./data/eval_dataset.json",
        alias="EVAL_DATASET_PATH",
    )
    sample_size: int = Field(
        default=100,
        alias="EVAL_SAMPLE_SIZE",
        ge=1,
    )


class Settings(BaseSettings):
    """Top-level aggregated application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Research Assistant"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, alias="DEBUG")

    # Sub-configurations
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)

    def is_openai_available(self) -> bool:
        """Return True if an OpenAI API key is configured."""
        return bool(self.llm.openai_api_key and self.llm.openai_api_key != "sk-...")

    def is_ollama_available(self) -> bool:
        """Best-effort check: try to reach the Ollama endpoint."""
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self.llm.ollama_base_url}/api/tags", timeout=2
            ):
                return True
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached global settings instance."""
    return Settings()
