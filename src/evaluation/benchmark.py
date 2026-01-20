"""
Evaluation benchmark pipeline for systematic RAG quality assessment.

Runs a complete evaluation over a dataset of (query, ground_truth) pairs,
computing RAGAS metrics and producing a structured report.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import structlog

from src.evaluation.metrics import RAGASEvaluator, RAGASMetrics

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkExample:
    """A single evaluation example."""

    query: str
    ground_truth: str
    metadata: dict[str, Any] | None = None


@dataclass
class BenchmarkResult:
    """Full benchmark run result."""

    examples: list[RAGASMetrics]
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float
    mean_overall: float
    total_examples: int
    failed_examples: int
    duration_seconds: float
    model_name: str
    dataset_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "dataset": self.dataset_name,
                "model": self.model_name,
                "total_examples": self.total_examples,
                "failed_examples": self.failed_examples,
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "metrics": {
                "faithfulness": round(self.mean_faithfulness, 4),
                "answer_relevancy": round(self.mean_answer_relevancy, 4),
                "context_precision": round(self.mean_context_precision, 4),
                "context_recall": round(self.mean_context_recall, 4),
                "overall": round(self.mean_overall, 4),
            },
            "per_example": [e.to_dict() for e in self.examples],
        }

    def print_report(self) -> None:
        """Print a formatted benchmark report to stdout."""
        print("\n" + "=" * 60)
        print(f"  RAG Benchmark Report — {self.dataset_name}")
        print("=" * 60)
        print(f"  Model:              {self.model_name}")
        print(f"  Examples evaluated: {self.total_examples}")
        print(f"  Failed:             {self.failed_examples}")
        print(f"  Duration:           {self.duration_seconds:.1f}s")
        print("-" * 60)
        print(f"  Faithfulness:       {self.mean_faithfulness:.4f}")
        print(f"  Answer Relevancy:   {self.mean_answer_relevancy:.4f}")
        print(f"  Context Precision:  {self.mean_context_precision:.4f}")
        print(f"  Context Recall:     {self.mean_context_recall:.4f}")
        print(f"  Overall (HM):       {self.mean_overall:.4f}")
        print("=" * 60 + "\n")


class RAGBenchmark:
    """
    Systematic evaluation benchmark for RAG pipelines.

    Accepts a RAG query function and evaluation dataset, runs all examples
    through the pipeline, evaluates with RAGAS metrics, and produces a
    structured report.

    Example:
        def my_rag_fn(query: str) -> tuple[str, list[str]]:
            # Returns (answer, context_chunks)
            ...

        benchmark = RAGBenchmark(
            rag_fn=my_rag_fn,
            evaluator=ragas_evaluator,
            model_name="gpt-4o-mini",
        )
        result = benchmark.run(examples=my_dataset)
        result.print_report()
    """

    def __init__(
        self,
        rag_fn: Callable[[str], tuple[str, list[str]]],
        evaluator: RAGASEvaluator,
        model_name: str = "unknown",
        max_workers: int = 1,
        fail_fast: bool = False,
    ) -> None:
        """
        Args:
            rag_fn: A callable that takes a query string and returns
                (answer: str, context_chunks: List[str]).
            evaluator: Configured RAGASEvaluator instance.
            model_name: LLM model name for the report.
            max_workers: Number of parallel evaluation workers (1=sequential).
            fail_fast: If True, stop on first evaluation failure.
        """
        self.rag_fn = rag_fn
        self.evaluator = evaluator
        self.model_name = model_name
        self.max_workers = max_workers
        self.fail_fast = fail_fast

    def run(
        self,
        examples: list[BenchmarkExample],
        dataset_name: str = "unnamed",
        sample_size: int | None = None,
    ) -> BenchmarkResult:
        """
        Run the full benchmark over a dataset.

        Args:
            examples: List of BenchmarkExample instances.
            dataset_name: Descriptive name for the dataset (for reporting).
            sample_size: If set, randomly sample this many examples.

        Returns:
            BenchmarkResult with per-example metrics and aggregates.
        """
        if sample_size and sample_size < len(examples):
            import random

            examples = random.sample(examples, sample_size)
            logger.info("benchmark.sampled", sample_size=sample_size)

        logger.info(
            "benchmark.start",
            dataset=dataset_name,
            n_examples=len(examples),
            model=self.model_name,
        )

        start = time.perf_counter()
        metrics_list: list[RAGASMetrics] = []
        failed = 0

        for i, example in enumerate(examples, start=1):
            logger.info(
                "benchmark.example",
                idx=i,
                total=len(examples),
                query=example.query[:60],
            )

            try:
                answer, context_chunks = self.rag_fn(example.query)

                metrics = self.evaluator.evaluate_single(
                    query=example.query,
                    answer=answer,
                    context_chunks=context_chunks,
                    ground_truth=example.ground_truth,
                )
                metrics_list.append(metrics)

                logger.info(
                    "benchmark.example_done",
                    idx=i,
                    overall=round(metrics.overall_score, 3),
                )

            except Exception as exc:
                logger.error(
                    "benchmark.example_failed",
                    idx=i,
                    query=example.query[:60],
                    error=str(exc),
                )
                failed += 1
                if self.fail_fast:
                    raise

        duration = time.perf_counter() - start

        if not metrics_list:
            logger.error("benchmark.all_failed")
            return BenchmarkResult(
                examples=[],
                mean_faithfulness=0.0,
                mean_answer_relevancy=0.0,
                mean_context_precision=0.0,
                mean_context_recall=0.0,
                mean_overall=0.0,
                total_examples=len(examples),
                failed_examples=failed,
                duration_seconds=duration,
                model_name=self.model_name,
                dataset_name=dataset_name,
            )

        result = BenchmarkResult(
            examples=metrics_list,
            mean_faithfulness=statistics.mean(m.faithfulness for m in metrics_list),
            mean_answer_relevancy=statistics.mean(m.answer_relevancy for m in metrics_list),
            mean_context_precision=statistics.mean(m.context_precision for m in metrics_list),
            mean_context_recall=statistics.mean(m.context_recall for m in metrics_list),
            mean_overall=statistics.mean(m.overall_score for m in metrics_list),
            total_examples=len(examples),
            failed_examples=failed,
            duration_seconds=duration,
            model_name=self.model_name,
            dataset_name=dataset_name,
        )

        logger.info(
            "benchmark.complete",
            overall=round(result.mean_overall, 4),
            duration_s=round(duration, 1),
        )

        return result

    def save_report(self, result: BenchmarkResult, output_path: str) -> None:
        """Save benchmark results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("benchmark.report_saved", path=str(path))


def load_benchmark_dataset(path: str) -> list[BenchmarkExample]:
    """
    Load a benchmark dataset from a JSON file.

    Expected format:
        [
            {"query": "...", "ground_truth": "...", "metadata": {...}},
            ...
        ]

    Args:
        path: Path to the JSON file.

    Returns:
        List of BenchmarkExample instances.
    """
    data = json.loads(Path(path).read_text())

    examples: list[BenchmarkExample] = []
    for item in data:
        examples.append(
            BenchmarkExample(
                query=item["query"],
                ground_truth=item.get("ground_truth", ""),
                metadata=item.get("metadata"),
            )
        )

    logger.info("benchmark.dataset_loaded", path=path, n=len(examples))
    return examples
