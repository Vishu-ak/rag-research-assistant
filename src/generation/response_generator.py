"""
Answer generation with source citations and confidence scoring.

Orchestrates the full RAG generation pipeline: context assembly →
prompt construction → LLM inference → response parsing and citation injection.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

import structlog

from config.settings import LLMSettings, RetrievalSettings
from src.generation.llm_client import BaseLLMClient, LLMResponse
from src.generation.prompt_templates import PromptRegistry, get_prompt_registry
from src.retrieval.context_builder import BuiltContext, Citation

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


@dataclass
class GeneratedAnswer:
    """
    A fully generated RAG answer with citations and metadata.

    Attributes:
        answer: The generated text answer.
        citations: Ordered list of source citations referenced in the answer.
        sources: Unique source file paths used.
        confidence: Heuristic confidence score (0.0 to 1.0).
        context_used: Number of context chunks that informed the answer.
        context_tokens: Total tokens in the context window.
        llm_response: Raw LLMResponse for token/cost accounting.
        query: Original user query.
        answer_with_citations: Answer text followed by formatted references.
    """

    answer: str
    citations: list[Citation]
    sources: list[str]
    confidence: float
    context_used: int
    context_tokens: int
    llm_response: LLMResponse
    query: str
    generation_latency_ms: float
    template_name: str = "rag_qa"
    metadata: dict = field(default_factory=dict)

    @property
    def answer_with_citations(self) -> str:
        """Return the answer followed by a formatted references block."""
        if not self.citations:
            return self.answer

        lines = [self.answer, "", "**References:**"]
        for c in self.citations:
            lines.append(c.format_full())
        return "\n".join(lines)

    @property
    def is_grounded(self) -> bool:
        """Return True if the answer contains at least one citation reference."""
        return bool(re.search(r"\[\d+\]", self.answer))

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "answer_with_citations": self.answer_with_citations,
            "sources": self.sources,
            "confidence": self.confidence,
            "is_grounded": self.is_grounded,
            "context_used": self.context_used,
            "context_tokens": self.context_tokens,
            "generation_latency_ms": round(self.generation_latency_ms, 1),
            "model": self.llm_response.model,
            "total_tokens": self.llm_response.total_tokens,
            "cost_usd": self.llm_response.cost_estimate_usd,
            "template": self.template_name,
        }


# ---------------------------------------------------------------------------
# Response generator
# ---------------------------------------------------------------------------


class ResponseGenerator:
    """
    Orchestrates RAG answer generation from context and LLM.

    Combines:
    1. Prompt selection and construction
    2. LLM completion (sync and async)
    3. Citation injection and response parsing
    4. Confidence scoring based on answer characteristics

    Example:
        generator = ResponseGenerator(llm_client=client)
        answer = generator.generate(
            query="What is RLHF?",
            context=built_context,
        )
        print(answer.answer_with_citations)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_registry: PromptRegistry | None = None,
        default_template: str = "rag_qa",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        rewrite_queries: bool = False,
    ) -> None:
        """
        Args:
            llm_client: The LLM backend to use for generation.
            prompt_registry: Registry of prompt templates.
            default_template: Template name to use by default.
            temperature: Generation temperature (lower = more deterministic).
            max_tokens: Maximum tokens in the generated answer.
            rewrite_queries: If True, rewrite the query before retrieval
                (experimental: requires extra LLM call).
        """
        self.llm_client = llm_client
        self.registry = prompt_registry or get_prompt_registry()
        self.default_template = default_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rewrite_queries = rewrite_queries

        logger.info(
            "response_generator.initialized",
            model=llm_client.model_name,
            template=default_template,
        )

    def generate(
        self,
        query: str,
        context: BuiltContext,
        template_name: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        """
        Generate an answer for a query given assembled context.

        Args:
            query: Original user question.
            context: Assembled BuiltContext from the context_builder module.
            template_name: Override the default prompt template.
            conversation_history: Prior turns for conversational mode.

        Returns:
            GeneratedAnswer with text, citations, and metadata.
        """
        tmpl_name = template_name or self.default_template

        # Build conversation-aware prompt if history provided
        if conversation_history and "conversational" not in tmpl_name:
            tmpl_name = "conversational_rag"

        template = self.registry.get(tmpl_name)

        # Format template kwargs
        format_kwargs: dict[str, str] = {
            "context": context.context_text,
            "question": query,
        }
        if conversation_history:
            format_kwargs["history"] = self._format_history(conversation_history)

        messages = template.format_messages(**format_kwargs)

        logger.info(
            "response_generator.generating",
            query=query[:80],
            template=tmpl_name,
            context_chunks=context.chunks_included,
            context_tokens=context.total_tokens,
            model=self.llm_client.model_name,
        )

        start = time.perf_counter()
        llm_response = self.llm_client.complete(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse and enrich the answer
        answer_text = llm_response.content
        active_citations = self._extract_active_citations(answer_text, context.citations)
        confidence = self._compute_confidence(answer_text, active_citations, context)

        logger.info(
            "response_generator.complete",
            query=query[:80],
            answer_len=len(answer_text),
            citations=len(active_citations),
            confidence=round(confidence, 3),
            latency_ms=round(elapsed_ms, 1),
            total_tokens=llm_response.total_tokens,
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=active_citations,
            sources=list({c.source for c in active_citations}),
            confidence=confidence,
            context_used=context.chunks_included,
            context_tokens=context.total_tokens,
            llm_response=llm_response,
            query=query,
            generation_latency_ms=elapsed_ms,
            template_name=tmpl_name,
        )

    async def agenerate(
        self,
        query: str,
        context: BuiltContext,
        template_name: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        """Async variant of generate()."""
        tmpl_name = template_name or self.default_template

        if conversation_history and "conversational" not in tmpl_name:
            tmpl_name = "conversational_rag"

        template = self.registry.get(tmpl_name)

        format_kwargs: dict[str, str] = {
            "context": context.context_text,
            "question": query,
        }
        if conversation_history:
            format_kwargs["history"] = self._format_history(conversation_history)

        messages = template.format_messages(**format_kwargs)

        start = time.perf_counter()
        llm_response = await self.llm_client.acomplete(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        answer_text = llm_response.content
        active_citations = self._extract_active_citations(answer_text, context.citations)
        confidence = self._compute_confidence(answer_text, active_citations, context)

        return GeneratedAnswer(
            answer=answer_text,
            citations=active_citations,
            sources=list({c.source for c in active_citations}),
            confidence=confidence,
            context_used=context.chunks_included,
            context_tokens=context.total_tokens,
            llm_response=llm_response,
            query=query,
            generation_latency_ms=elapsed_ms,
            template_name=tmpl_name,
        )

    def stream(
        self,
        query: str,
        context: BuiltContext,
        template_name: str | None = None,
    ) -> Iterator[str]:
        """
        Stream answer tokens as they are generated.

        Args:
            query: User question.
            context: Assembled context.
            template_name: Prompt template to use.

        Yields:
            String text deltas from the LLM.
        """
        tmpl_name = template_name or self.default_template
        template = self.registry.get(tmpl_name)
        messages = template.format_messages(
            context=context.context_text,
            question=query,
        )

        yield from self.llm_client.stream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def astream(
        self,
        query: str,
        context: BuiltContext,
        template_name: str | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming answer generation."""
        tmpl_name = template_name or self.default_template
        template = self.registry.get(tmpl_name)
        messages = template.format_messages(
            context=context.context_text,
            question=query,
        )

        async for token in self.llm_client.astream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            yield token

    def rewrite_query(self, query: str) -> dict[str, str | list[str]]:
        """
        Use the LLM to rewrite a query for improved retrieval.

        Returns a dict with 'reformulated', 'alternatives', and 'key_terms'.
        """
        template = self.registry.get("query_rewrite")
        messages = template.format_messages(query=query)

        response = self.llm_client.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=256,
        )

        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback to original query
        return {
            "reformulated": query,
            "alternatives": [],
            "key_terms": query.split()[:5],
        }

    def summarize(self, context: BuiltContext) -> str:
        """Generate a structured summary of the context documents."""
        template = self.registry.get("summarize")
        messages = template.format_messages(context=context.context_text)

        response = self.llm_client.complete(
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )
        return response.content

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_active_citations(
        answer_text: str, available_citations: list[Citation]
    ) -> list[Citation]:
        """
        Extract only the citations actually referenced in the answer.

        Args:
            answer_text: The generated answer.
            available_citations: All citations from the context.

        Returns:
            Filtered and re-indexed list of active citations.
        """
        referenced_indices = set(
            int(m) for m in re.findall(r"\[(\d+)\]", answer_text)
        )

        # Filter to only referenced citations, preserving order
        active = [
            c for c in available_citations if c.index in referenced_indices
        ]

        # Re-index citations to be 1-based and contiguous
        remapped: list[Citation] = []
        for new_idx, c in enumerate(active, start=1):
            remapped.append(
                Citation(
                    index=new_idx,
                    source=c.source,
                    page=c.page,
                    chunk_index=c.chunk_index,
                    title=c.title,
                    doi=c.doi,
                    authors=c.authors,
                    date=c.date,
                    text_snippet=c.text_snippet,
                )
            )

        return remapped

    @staticmethod
    def _compute_confidence(
        answer_text: str,
        citations: list[Citation],
        context: BuiltContext,
    ) -> float:
        """
        Compute a heuristic confidence score for the generated answer.

        Considers:
        - Whether the answer contains citation references
        - Answer length (too short answers may be refusals)
        - Whether "cannot find" or "insufficient" phrases are present
        - Ratio of cited chunks to available context chunks
        """
        if not answer_text.strip():
            return 0.0

        # Signals of low confidence / refusal
        refusal_patterns = [
            "cannot find", "insufficient information", "not mentioned",
            "not provided", "unable to answer", "no information",
            "cannot answer", "not available in the context",
        ]
        if any(p in answer_text.lower() for p in refusal_patterns):
            return 0.1

        score = 0.5  # Base score

        # Citation grounding: answers with references are more confident
        if citations:
            citation_ratio = len(citations) / max(1, context.chunks_included)
            score += min(0.25, citation_ratio * 0.5)

        # Answer length heuristic
        word_count = len(answer_text.split())
        if word_count > 50:
            score += 0.1
        if word_count > 150:
            score += 0.05

        # Presence of hedging language reduces confidence
        hedging_terms = ["possibly", "might be", "perhaps", "unclear", "seems"]
        hedge_count = sum(1 for term in hedging_terms if term in answer_text.lower())
        score -= min(0.15, hedge_count * 0.05)

        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _format_history(history: list[dict[str, str]]) -> str:
        """Format conversation history for injection into prompts."""
        lines = []
        for msg in history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
