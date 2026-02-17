"""
RAG Research Assistant — End-to-End Demo Script

This script demonstrates the full pipeline from document ingestion through
retrieval and answer generation. Run it to verify your environment is
correctly configured.

Usage:
    python notebooks/exploration.py

Requirements:
    - .env file with OPENAI_API_KEY (or set LLM_BACKEND=ollama)
    - Python 3.10+
    - pip install -r requirements.txt
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner() -> None:
    print("\n" + "=" * 65)
    print("   RAG Research Assistant — Interactive Demo")
    print("=" * 65 + "\n")


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def demo_configuration() -> "Settings":
    """Step 1: Load and display configuration."""
    section("1. Configuration")
    from config.settings import get_settings

    settings = get_settings()
    print(f"  LLM Backend:      {settings.llm.backend}")
    print(f"  LLM Model:        {settings.llm.openai_model if settings.llm.backend == 'openai' else settings.llm.ollama_model}")
    print(f"  Embedding:        {settings.embedding.backend}")
    print(f"  Embedding Model:  {settings.embedding.sentence_transformer_model}")
    print(f"  Reranker:         {settings.embedding.reranker_model}")
    print(f"  ChromaDB:         {'Remote' if settings.chroma.use_server else 'Local'} ({settings.chroma.persist_dir})")
    print(f"  Chunk Size:       {settings.chunking.chunk_size} tokens")
    print(f"  Retrieval Top-K:  {settings.retrieval.top_k}")
    print(f"  Rerank Top-K:     {settings.retrieval.rerank_top_k}")
    return settings


def demo_text_splitting() -> None:
    """Step 2: Demonstrate text splitting strategies."""
    section("2. Text Splitting Demo")
    from langchain_core.documents import Document
    from src.ingestion.text_splitter import RecursiveChunker

    # Sample research text
    sample_text = """
    The attention mechanism has fundamentally transformed how neural networks process
    sequential data. Unlike recurrent architectures that process tokens one by one,
    attention allows every position in the sequence to directly attend to every other
    position in a single computation step.

    The scaled dot-product attention formula computes compatibility between query
    and key vectors, normalizes with softmax, and uses the result to weight the values.
    This formulation is highly parallelizable and forms the foundation of the
    Transformer architecture.

    Multi-head attention extends this concept by running multiple attention operations
    in parallel, each with different learned projections. The outputs are concatenated
    and projected, allowing the model to capture diverse types of relationships.

    Positional encodings add information about sequence order, since the attention
    mechanism itself is permutation-invariant. The original Transformer uses sinusoidal
    encodings; modern models often learn these representations directly.

    Applications of attention span machine translation, text summarization, question
    answering, code generation, and many other NLP tasks. The efficiency improvements
    from flash-attention and sparse-attention variants have made these models practical
    at unprecedented scales.
    """

    doc = Document(page_content=sample_text.strip(), metadata={"source": "demo_text.txt", "page": 1})

    chunker = RecursiveChunker(chunk_size=128, chunk_overlap=20)
    chunks = chunker.split([doc])

    print(f"\n  Input: {len(sample_text.split())} words")
    print(f"  Strategy: Recursive character splitting")
    print(f"  Chunk size: 128 tokens | Overlap: 20 tokens")
    print(f"  Output: {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        tokens = chunk.metadata.get("token_count", "?")
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"  [{i}] ~{tokens} tokens: {preview}...")


def demo_preprocessing() -> None:
    """Step 3: Show preprocessing pipeline."""
    section("3. Preprocessing Demo")
    from langchain_core.documents import Document
    from src.ingestion.preprocessor import TextCleaningPipeline, extract_metadata

    messy_text = """
    ATTENTION IS ALL YOU NEED

    Ashish Vaswani, Noam Shazeer, Niki Parmar...

    Abstract—The dominant sequence transduction models are based on complex recurrent
    or convolutional neural networks. Visit https://arxiv.org/abs/1706.03762 for the paper.

    DOI: 10.48550/arXiv.1706.03762

    Keywords: attention mechanism, transformer, neural machine translation
    """

    pipeline = TextCleaningPipeline(
        remove_urls=True,
        normalize_whitespace=True,
        fix_unicode=True,
        remove_headers_footers=True,
    )

    doc = Document(page_content=messy_text.strip(), metadata={})
    cleaned_docs = pipeline.clean_documents([doc])

    print(f"\n  Original length: {len(messy_text)} chars")
    print(f"  Cleaned length:  {len(cleaned_docs[0].page_content)} chars")

    # Extract metadata
    meta = extract_metadata(messy_text)
    print(f"\n  Extracted Metadata:")
    print(f"    Title:    {meta.title}")
    print(f"    DOI:      {meta.doi}")
    print(f"    Keywords: {meta.keywords}")
    print(f"    Words:    {meta.word_count}")


def demo_embedding_creation(settings: "Settings"):
    """Step 4: Initialize and test the embedding model."""
    section("4. Embedding Model Demo")

    print(f"\n  Loading {settings.embedding.sentence_transformer_model}...")
    from src.embeddings.embedding_manager import create_embedder

    try:
        embedder = create_embedder(
            settings=settings.embedding,
            use_cache=False,
            openai_api_key=settings.llm.openai_api_key,
        )
        print(f"  Model dimension: {embedder.dimension}")
        print(f"  Model name:      {embedder.model_name}")

        test_texts = [
            "The transformer architecture uses self-attention.",
            "Self-attention allows each position to attend to all positions.",
            "Convolutional networks use local receptive fields.",
        ]

        print(f"\n  Embedding {len(test_texts)} test sentences...")
        start = time.perf_counter()
        vecs = embedder.embed_documents(test_texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"  Embedding time:  {elapsed_ms:.1f}ms")
        print(f"  Vector shape:    {len(vecs)} × {len(vecs[0])}")

        # Cosine similarity between first two (should be high — both about transformers)
        import math
        a, b = vecs[0], vecs[1]
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b))
        sim_12 = dot / norm if norm > 0 else 0.0

        a, c = vecs[0], vecs[2]
        dot = sum(x * y for x, y in zip(a, c))
        norm = math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in c))
        sim_13 = dot / norm if norm > 0 else 0.0

        print(f"\n  Similarity (transformers pair): {sim_12:.4f}")
        print(f"  Similarity (dissimilar pair):   {sim_13:.4f}")
        print(f"  Semantic gap: {sim_12 - sim_13:.4f} (positive = correct ordering)")

        return embedder
    except Exception as exc:
        print(f"  Warning: Could not load embedding model: {exc}")
        return None


def demo_bm25_retrieval():
    """Step 5: Demonstrate BM25 sparse retrieval."""
    section("5. BM25 Sparse Retrieval Demo")
    from langchain_core.documents import Document
    from src.retrieval.retriever import BM25Index

    docs = [
        Document(page_content="The transformer model uses multi-head self-attention mechanisms.", metadata={"source": "transformers.pdf"}),
        Document(page_content="BERT pre-trains a deep bidirectional transformer for language understanding.", metadata={"source": "bert.pdf"}),
        Document(page_content="GPT-3 is an autoregressive language model with 175 billion parameters.", metadata={"source": "gpt3.pdf"}),
        Document(page_content="Convolutional neural networks excel at image classification tasks.", metadata={"source": "cnn.pdf"}),
        Document(page_content="Recurrent neural networks process sequences step by step using hidden states.", metadata={"source": "rnn.pdf"}),
        Document(page_content="The attention mechanism computes dot products between query and key vectors.", metadata={"source": "attention.pdf"}),
    ]

    index = BM25Index(documents=docs)
    print(f"\n  Indexed {len(docs)} documents")
    print(f"  Vocabulary size: {len(index._inverted_index)} unique terms\n")

    queries = [
        "transformer attention mechanism",
        "language model parameters",
        "image classification convolution",
    ]

    for query in queries:
        results = index.search(query, k=3)
        print(f"  Query: '{query}'")
        for r in results:
            print(f"    [{r.rank}] score={r.score:.3f} | {r.document.metadata['source']}")
        print()


def demo_vector_store_and_hybrid_retrieval(settings: "Settings", embedder) -> None:
    """Step 6: Vector store and hybrid retrieval."""
    section("6. Vector Store + Hybrid Retrieval Demo")

    if embedder is None:
        print("  Skipping (embedding model not available)")
        return

    import tempfile
    from langchain_core.documents import Document
    from src.embeddings.vector_store import VectorStore
    from src.retrieval.retriever import BM25Index, HybridRetriever
    from config.settings import ChromaSettings, RetrievalSettings

    # Use a temporary directory for the demo
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_settings = ChromaSettings()
        chroma_settings.persist_dir = tmpdir
        chroma_settings.collection_name = "demo_collection"

        try:
            store = VectorStore(settings=chroma_settings, embedder=embedder)

            demo_docs = [
                Document(page_content="Attention mechanisms allow models to focus on relevant information.", metadata={"source": "attention.pdf", "page": 1, "chunk_index": 0}),
                Document(page_content="The Transformer architecture replaces recurrence with self-attention.", metadata={"source": "attention.pdf", "page": 2, "chunk_index": 1}),
                Document(page_content="BERT uses masked language modeling and next sentence prediction.", metadata={"source": "bert.pdf", "page": 1, "chunk_index": 0}),
                Document(page_content="Large language models exhibit emergent capabilities at scale.", metadata={"source": "scaling.pdf", "page": 1, "chunk_index": 0}),
            ]

            print(f"\n  Adding {len(demo_docs)} documents to ChromaDB...")
            ids = store.add_documents(demo_docs)
            print(f"  Added with IDs: {ids[:2]}...")

            # Dense retrieval
            query = "how does attention mechanism work"
            dense_results = store.similarity_search(query, k=3)
            print(f"\n  Dense retrieval for: '{query}'")
            for r in dense_results:
                print(f"    score={r.score:.4f} | {r.document.metadata['source']}")

            # Hybrid retrieval
            bm25 = BM25Index(documents=demo_docs)
            retrieval_settings = RetrievalSettings()
            retrieval_settings.top_k = 10
            retrieval_settings.hybrid_dense_weight = 0.7
            retrieval_settings.enable_reranking = False

            hybrid = HybridRetriever(
                vector_store=store,
                bm25_index=bm25,
                settings=retrieval_settings,
            )

            hybrid_results = hybrid.retrieve(query, k=3)
            print(f"\n  Hybrid retrieval for: '{query}'")
            for r in hybrid_results:
                print(f"    rrf={r.rrf_score:.4f} | dense_rank={r.dense_rank} | sparse_rank={r.sparse_rank}")
                print(f"    {r.document.metadata['source']}: {r.document.page_content[:60]}...")

        except Exception as exc:
            print(f"  Demo skipped: {exc}")


def demo_context_building() -> None:
    """Step 7: Context assembly and citation tracking."""
    section("7. Context Building Demo")
    from langchain_core.documents import Document
    from src.retrieval.context_builder import ContextBuilder
    from src.retrieval.reranker import RerankedResult

    docs_and_scores = [
        ("The attention mechanism [Vaswani 2017] maps a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.", "attention.pdf", 1, 0.95),
        ("Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.", "attention.pdf", 2, 0.88),
        ("Position-wise feed-forward networks are applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.", "attention.pdf", 3, 0.75),
    ]

    results = [
        RerankedResult(
            document=Document(
                page_content=text,
                metadata={"source": src, "page": page, "chunk_index": i},
            ),
            rerank_score=score,
            original_rank=i + 1,
            final_rank=i + 1,
        )
        for i, (text, src, page, score) in enumerate(docs_and_scores)
    ]

    builder = ContextBuilder(max_context_tokens=500)
    context = builder.build(query="How does attention work?", results=results)

    print(f"\n  Chunks included:   {context.chunks_included}")
    print(f"  Chunks truncated:  {context.chunks_truncated}")
    print(f"  Total tokens:      {context.total_tokens}")
    print(f"  Citations created: {len(context.citations)}")
    print(f"\n  Context preview:\n")
    print("  " + "\n  ".join(context.context_text.split("\n")[:8]))
    print(f"\n  {context.citation_block}")


def demo_prompt_templates() -> None:
    """Step 8: Show prompt template rendering."""
    section("8. Prompt Template Demo")
    from src.generation.prompt_templates import get_prompt_registry

    registry = get_prompt_registry()
    templates = registry.list_templates()
    print(f"\n  Available templates ({len(templates)}):")
    for t in templates:
        print(f"    • {t['name']}: {t['description']}")

    # Render the RAG QA template
    template = registry.get("rag_qa")
    messages = template.format_messages(
        context="[1] attention.pdf\nThe attention mechanism computes...",
        question="What is the key innovation of the attention mechanism?",
    )

    print(f"\n  Rendered 'rag_qa' template → {len(messages)} messages:")
    for msg in messages:
        role = msg["role"].upper()
        content_preview = msg["content"][:120].replace("\n", " ")
        print(f"    [{role}] {content_preview}...")


def main() -> None:
    """Run the full demo pipeline."""
    print_banner()

    try:
        settings = demo_configuration()
        demo_text_splitting()
        demo_preprocessing()
        embedder = demo_embedding_creation(settings)
        demo_bm25_retrieval()
        demo_vector_store_and_hybrid_retrieval(settings, embedder)
        demo_context_building()
        demo_prompt_templates()

        print("\n" + "=" * 65)
        print("  Demo complete! The pipeline is working correctly.")
        print("  Next steps:")
        print("   1. Set your OPENAI_API_KEY in .env")
        print("   2. Run the API: uvicorn api.main:app --port 8080")
        print("   3. Ingest a PDF: POST /api/v1/ingest/file")
        print("   4. Ask a question: POST /api/v1/query")
        print("=" * 65 + "\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    except Exception as exc:
        print(f"\nDemo failed: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
