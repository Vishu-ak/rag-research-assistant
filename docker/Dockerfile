# ============================================================
# RAG Research Assistant — Production Dockerfile
# Multi-stage build: builder stage for deps, slim runtime image
# ============================================================

# ----- Stage 1: Dependency builder -----
FROM python:3.11-slim AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Create a virtual environment and install all dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ----- Stage 2: Runtime image -----
FROM python:3.11-slim AS runtime

LABEL maintainer="RAG Research Team" \
      version="1.0.0" \
      description="RAG Research Assistant — Retrieval-Augmented Generation API"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=raguser:raguser . .

# Create directories for ChromaDB persistence and logs
RUN mkdir -p /app/chroma_db /app/logs && \
    chown -R raguser:raguser /app/chroma_db /app/logs

# Switch to non-root user
USER raguser

# Environment variables with sensible defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    CHROMA_PERSIST_DIR=/app/chroma_db

# Expose the API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Entry point
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--log-config", "/dev/null", \
     "--no-access-log"]
