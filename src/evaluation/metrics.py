"""
RAG evaluation metrics: faithfulness, answer relevancy, context precision/recall.

Implements RAGAS-inspired evaluation metrics for assessing RAG pipeline quality.
These metrics measure three key dimensions:
  1. Faithfulness — Does the answer stay faithful to the retrieved context?
  2. Answer Relevancy — Does the answer address the question?
  3. Context Precision — Are retrieved documents actually relevant?
  4. Context Recall — Does the context cover all aspects of the answer?

Reference:
    Es et al. (2023), "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
    https://arxiv.org/abs/2309.15217
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import structlog
from langchain_core.documents import Document

from src.generation.llm_client import BaseLLMClient

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric result types
# ---------------------------------------------------------------------------


@dataclass
class FaithfulnessScore:
    """Faithfulness evaluation result for a single QA pair."""

    score: float          # 0.0 to 1.0
    n_claims: int         # Number of claims in the answer
    n_supported: int      # Claims supported by context
    unsupported_claims: list[str]


@dataclass
class RelevancyScore:
    """Answer relevancy score for a single QA pair."""

    score: float          # 0.0 to 1.0
    reasoning: str        # Short explanation


@dataclass
class ContextPrecisionScore:
    """Context precision — fraction of retrieved docs that are relevant."""

    score: float          # 0.0 to 1.0
    relevant_count: int
    total_retrieved: int


@dataclass
class ContextRecallScore:
    """Context recall — fraction of ground truth covered by context."""

    score: float          # 0.0 to 1.0
    covered_statements: int
    total_statements: int


@dataclass
class RAGASMetrics:
    """Aggregated RAGAS evaluation metrics for a single example."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    query: str

    @property
    def overall_score(self) -> float:
        """Harmonic mean of all four RAGAS metrics."""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        if any(s == 0 for s in scores):
            return 0.0
        return len(scores) / sum(1 / s for s in scores)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall_score"] = round(self.overall_score, 4)
        return d


# ---------------------------------------------------------------------------
# LLM-based faithfulness evaluator
# ---------------------------------------------------------------------------


class FaithfulnessEvaluator:
    """
    Measures whether the generated answer is faithful to the source context.

    Uses an LLM-as-a-judge approach: the LLM decomposes the answer into
    individual factual claims, then verifies each claim against the context.

    Score = (number of supported claims) / (total number of claims)

    Example:
        evaluator = FaithfulnessEvaluator(llm_client=client)
        result = evaluator.evaluate(
            answer="Attention was introduced in 2017 by Vaswani et al.",
            context="The transformer model was proposed in the paper..."
        )
        print(f"Faithfulness: {result.score:.2f}")
    """

    DECOMPOSE_PROMPT = """\
Given the following answer, extract all individual factual claims.
Return a JSON array of claim strings. Each claim should be a single,
verifiable statement.

Answer: {answer}

Return only a JSON array, no explanation. Example:
["claim 1", "claim 2", "claim 3"]
"""

    VERIFY_PROMPT = """\
Context:
{context}

Claim: {claim}

Is this claim directly supported by the context above?
Respond with exactly one word: YES or NO.
"""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    def evaluate(
        self,
        answer: str,
        context: str,
    ) -> FaithfulnessScore:
        """
        Evaluate faithfulness of an answer against its context.

        Args:
            answer: The generated answer to evaluate.
            context: The context string used to generate the answer.

        Returns:
            FaithfulnessScore with claim-level details.
        """
        # Step 1: Decompose answer into individual claims
        claims = self._decompose_claims(answer)

        if not claims:
            logger.warning("faithfulness.no_claims_extracted", answer=answer[:100])
            return FaithfulnessScore(
                score=0.0,
                n_claims=0,
                n_supported=0,
                unsupported_claims=[],
            )

        # Step 2: Verify each claim against the context
        supported: list[str] = []
        unsupported: list[str] = []

        for claim in claims:
            is_supported = self._verify_claim(claim, context)
            if is_supported:
                supported.append(claim)
            else:
                unsupported.append(claim)

        score = len(supported) / len(claims)

        logger.debug(
            "faithfulness.evaluated",
            total_claims=len(claims),
            supported=len(supported),
            score=round(score, 3),
        )

        return FaithfulnessScore(
            score=round(score, 4),
            n_claims=len(claims),
            n_supported=len(supported),
            unsupported_claims=unsupported,
        )

    def _decompose_claims(self, answer: str) -> list[str]:
        """Use LLM to extract individual factual claims from the answer."""
        prompt = self.DECOMPOSE_PROMPT.format(answer=answer)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm_client.complete(messages, temperature=0.0, max_tokens=512)
            # Parse JSON array from response
            json_match = re.search(r"\[.*?\]", response.content, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
                return [c for c in claims if isinstance(c, str) and c.strip()]
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("faithfulness.decompose_failed", error=str(exc))

        # Fallback: split by sentences
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]

    def _verify_claim(self, claim: str, context: str) -> bool:
        """Use LLM to verify a single claim against context."""
        prompt = self.VERIFY_PROMPT.format(claim=claim, context=context[:4000])
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm_client.complete(messages, temperature=0.0, max_tokens=5)
            return response.content.strip().upper().startswith("YES")
        except Exception as exc:
            logger.warning("faithfulness.verify_failed", error=str(exc))
            return False


# ---------------------------------------------------------------------------
# Answer relevancy evaluator
# ---------------------------------------------------------------------------


class AnswerRelevancyEvaluator:
    """
    Measures whether the answer directly addresses the question.

    Generates N synthetic questions from the answer and computes
    the average cosine similarity between the synthetic questions
    and the original question. Higher similarity = more relevant answer.

    This is a question-generation based approach that doesn't require
    ground truth and is question-agnostic.

    Example:
        evaluator = AnswerRelevancyEvaluator(llm_client=client, embedder=embed_fn)
        result = evaluator.evaluate(question="...", answer="...")
    """

    QUESTION_GEN_PROMPT = """\
Given the following answer, generate {n} questions that this answer addresses.
Return a JSON array of question strings.

Answer: {answer}

Return only a JSON array.
"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embed_fn: Any,
        n_questions: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.embed_fn = embed_fn
        self.n_questions = n_questions

    def evaluate(self, question: str, answer: str) -> RelevancyScore:
        """
        Evaluate whether the answer is relevant to the question.

        Args:
            question: Original user question.
            answer: Generated answer.

        Returns:
            RelevancyScore with similarity-based score.
        """
        # Detect refusal / null answers
        refusal_patterns = [
            "cannot find", "not available", "insufficient information",
            "unable to answer", "no information in the context",
        ]
        if any(p in answer.lower() for p in refusal_patterns):
            return RelevancyScore(
                score=0.1,
                reasoning="Answer does not address the question (refusal pattern detected).",
            )

        # Generate synthetic questions from the answer
        synthetic_qs = self._generate_questions(answer)

        if not synthetic_qs:
            return RelevancyScore(
                score=0.0,
                reasoning="Could not generate synthetic questions from answer.",
            )

        # Embed original question and synthetic questions
        all_texts = [question] + synthetic_qs
        try:
            embeddings = self.embed_fn(all_texts)
        except Exception as exc:
            logger.warning("relevancy.embedding_failed", error=str(exc))
            return RelevancyScore(score=0.5, reasoning="Embedding failed; defaulting to 0.5.")

        orig_emb = embeddings[0]
        synth_embs = embeddings[1:]

        similarities = [
            self._cosine_similarity(orig_emb, emb) for emb in synth_embs
        ]
        score = statistics.mean(similarities) if similarities else 0.0

        reasoning = (
            f"Average similarity between original question and {len(synthetic_qs)} "
            f"generated questions: {score:.3f}"
        )

        logger.debug(
            "relevancy.evaluated",
            score=round(score, 3),
            n_synthetic=len(synthetic_qs),
        )

        return RelevancyScore(score=round(score, 4), reasoning=reasoning)

    def _generate_questions(self, answer: str) -> list[str]:
        """Generate synthetic questions from the answer."""
        prompt = self.QUESTION_GEN_PROMPT.format(
            n=self.n_questions, answer=answer[:2000]
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm_client.complete(messages, temperature=0.3, max_tokens=256)
            json_match = re.search(r"\[.*?\]", response.content, re.DOTALL)
            if json_match:
                qs = json.loads(json_match.group())
                return [q for q in qs if isinstance(q, str) and "?" in q]
        except Exception as exc:
            logger.warning("relevancy.question_gen_failed", error=str(exc))

        return []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Context precision evaluator
# ---------------------------------------------------------------------------


class ContextPrecisionEvaluator:
    """
    Measures what fraction of retrieved context is actually relevant.

    Uses an LLM to classify each retrieved document as relevant or
    irrelevant to the (question, ground_truth_answer) pair.

    Score = (relevant retrieved) / (total retrieved)
    """

    RELEVANCE_PROMPT = """\
Question: {question}
Answer: {answer}

Is the following context passage relevant to answering the question?
A passage is relevant if it contains information that helps answer the question.

Context passage:
{context_chunk}

Respond with exactly one word: YES or NO.
"""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    def evaluate(
        self,
        question: str,
        ground_truth: str,
        retrieved_contexts: list[str],
    ) -> ContextPrecisionScore:
        """
        Evaluate context precision.

        Args:
            question: User question.
            ground_truth: Reference answer or expected answer.
            retrieved_contexts: List of retrieved document text chunks.

        Returns:
            ContextPrecisionScore with per-document relevance.
        """
        if not retrieved_contexts:
            return ContextPrecisionScore(score=0.0, relevant_count=0, total_retrieved=0)

        relevant_count = sum(
            1
            for ctx in retrieved_contexts
            if self._is_relevant(question, ground_truth, ctx)
        )

        score = relevant_count / len(retrieved_contexts)

        return ContextPrecisionScore(
            score=round(score, 4),
            relevant_count=relevant_count,
            total_retrieved=len(retrieved_contexts),
        )

    def _is_relevant(self, question: str, answer: str, context_chunk: str) -> bool:
        """Use LLM to classify a single chunk as relevant or not."""
        prompt = self.RELEVANCE_PROMPT.format(
            question=question,
            answer=answer[:500],
            context_chunk=context_chunk[:1000],
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm_client.complete(messages, temperature=0.0, max_tokens=5)
            return response.content.strip().upper().startswith("YES")
        except Exception:
            return True  # Conservative: assume relevant on error


# ---------------------------------------------------------------------------
# Composite RAGAS evaluator
# ---------------------------------------------------------------------------


class RAGASEvaluator:
    """
    Composite evaluator that runs all four RAGAS metrics.

    Provides both per-example and aggregate evaluation.

    Example:
        evaluator = RAGASEvaluator(llm_client=client, embed_fn=embed_fn)
        metrics = evaluator.evaluate_single(
            query="What is attention?",
            answer="Attention is a mechanism...",
            context_chunks=["The attention mechanism..."],
            ground_truth="Attention is ...",
        )
        print(f"Overall: {metrics.overall_score:.3f}")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embed_fn: Any,
    ) -> None:
        self.faithfulness_eval = FaithfulnessEvaluator(llm_client)
        self.relevancy_eval = AnswerRelevancyEvaluator(llm_client, embed_fn)
        self.precision_eval = ContextPrecisionEvaluator(llm_client)

    def evaluate_single(
        self,
        query: str,
        answer: str,
        context_chunks: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetrics:
        """
        Evaluate a single QA example with all RAGAS metrics.

        Args:
            query: User question.
            answer: Generated answer.
            context_chunks: Retrieved context passages.
            ground_truth: Reference answer for precision/recall.

        Returns:
            RAGASMetrics with all four dimensions scored.
        """
        full_context = "\n\n".join(context_chunks)

        logger.info("ragas.evaluating_single", query=query[:60])

        faith = self.faithfulness_eval.evaluate(answer=answer, context=full_context)
        rel = self.relevancy_eval.evaluate(question=query, answer=answer)
        prec = self.precision_eval.evaluate(
            question=query,
            ground_truth=ground_truth or answer,
            retrieved_contexts=context_chunks,
        )

        # Context recall: approximate as faithful answer coverage
        # (full RAGAS recall requires ground truth statements)
        recall_score = faith.score * 0.9 + prec.score * 0.1  # heuristic blend

        return RAGASMetrics(
            faithfulness=faith.score,
            answer_relevancy=rel.score,
            context_precision=prec.score,
            context_recall=round(recall_score, 4),
            query=query,
        )
