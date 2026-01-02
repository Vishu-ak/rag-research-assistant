"""
LLM client abstraction supporting OpenAI and Ollama (local) backends.

Provides a unified async/sync interface so the rest of the pipeline
can switch between cloud and local inference without code changes.
Includes retry logic, token counting, and streaming support.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import structlog
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import LLMSettings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str = "stop"

    @property
    def cost_estimate_usd(self) -> float | None:
        """Rough cost estimate for OpenAI models (None for Ollama)."""
        pricing: dict[str, tuple[float, float]] = {
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.000150, 0.000600),
            "gpt-4-turbo": (0.010, 0.030),
            "gpt-3.5-turbo": (0.0005, 0.0015),
        }
        for model_prefix, (input_price, output_price) in pricing.items():
            if self.model.startswith(model_prefix):
                return (
                    self.prompt_tokens / 1000 * input_price
                    + self.completion_tokens / 1000 * output_price
                )
        return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """Synchronous completion."""
        ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """Asynchronous completion."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Synchronous streaming completion (yields text deltas)."""
        ...

    @abstractmethod
    async def astream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Asynchronous streaming completion (yields text deltas)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier."""
        ...

    def count_tokens(self, text: str) -> int:
        """Estimate token count (approximate if tiktoken unavailable)."""
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.encoding_for_model("gpt-4o")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


class OpenAIClient(BaseLLMClient):
    """
    OpenAI Chat Completions API client.

    Supports all OpenAI chat models (gpt-4o, gpt-4o-mini, etc.)
    with async support and exponential backoff retries on transient errors.

    Example:
        client = OpenAIClient(api_key="sk-...", model="gpt-4o-mini")
        resp = client.complete([{"role": "user", "content": "Hello!"}])
        print(resp.content)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("openai not installed: pip install openai") from e

        self._model = model
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._max_retries = max_retries

        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._async_client = AsyncOpenAI(api_key=api_key, timeout=timeout)

        logger.info("openai_client.initialized", model=model)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Synchronous chat completion with retry."""
        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        start = time.perf_counter()

        for attempt in Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore
                    temperature=temp,
                    max_tokens=tok,
                    **kwargs,
                )

        elapsed_ms = (time.perf_counter() - start) * 1000
        usage = response.usage
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        logger.debug(
            "openai_client.complete",
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=round(elapsed_ms, 1),
        )

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=elapsed_ms,
            finish_reason=finish_reason,
        )

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Asynchronous chat completion with retry."""
        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        start = time.perf_counter()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore
                    temperature=temp,
                    max_tokens=tok,
                    **kwargs,
                )

        elapsed_ms = (time.perf_counter() - start) * 1000
        usage = response.usage
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=elapsed_ms,
            finish_reason=finish_reason,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Streaming chat completion."""
        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        with self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=temp,
            max_tokens=tok,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def astream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming chat completion."""
        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        async with await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=temp,
            max_tokens=tok,
            stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


class OllamaClient(BaseLLMClient):
    """
    Ollama local LLM client for running models like Llama 3.2, Mistral, etc.

    Communicates with the Ollama HTTP API. Requires Ollama to be running
    (default: http://localhost:11434).

    Example:
        client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")
        resp = client.complete([{"role": "user", "content": "Explain RAG in one paragraph."}])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        try:
            import httpx  # type: ignore
        except ImportError as e:
            raise ImportError("httpx not installed: pip install httpx") from e

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._timeout = timeout

        import httpx  # type: ignore

        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)
        self._async_client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

        logger.info("ollama_client.initialized", model=model, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string for Ollama /generate."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"Human: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Synchronous completion via Ollama /api/chat endpoint."""
        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        start = time.perf_counter()

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": tok,
            },
        }

        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        content = data.get("message", {}).get("content", "")

        # Ollama eval counts
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        logger.debug(
            "ollama_client.complete",
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 1),
        )

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=elapsed_ms,
            finish_reason=data.get("done_reason", "stop"),
        )

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Asynchronous completion via Ollama."""
        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        start = time.perf_counter()

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp, "num_predict": tok},
        }

        response = await self._async_client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=elapsed_ms,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Streaming completion via Ollama."""
        import json

        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temp, "num_predict": tok},
        }

        with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break

    async def astream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming completion via Ollama."""
        import json

        temp = temperature if temperature is not None else self._default_temperature
        tok = max_tokens if max_tokens is not None else self._default_max_tokens

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temp, "num_predict": tok},
        }

        async with self._async_client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_client(settings: LLMSettings) -> BaseLLMClient:
    """
    Factory that creates the configured LLM client from application settings.

    Args:
        settings: LLMSettings from the application config.

    Returns:
        BaseLLMClient instance (OpenAIClient or OllamaClient).

    Raises:
        ValueError: If backend is 'openai' and no API key is configured.
    """
    if settings.backend == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set in environment when LLM_BACKEND=openai."
            )
        return OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.request_timeout,
        )

    if settings.backend == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.request_timeout,
        )

    raise ValueError(f"Unknown LLM backend: {settings.backend!r}. Choose 'openai' or 'ollama'.")
