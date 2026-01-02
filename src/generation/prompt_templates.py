"""
Prompt engineering templates for RAG generation.

Provides carefully tuned prompts for different RAG scenarios:
  - Standard QA with context and citations
  - Multi-hop reasoning over multiple sources
  - Summarization and synthesis
  - Factual verification / contradiction detection
  - Query rewriting for improved retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any


@dataclass
class PromptTemplate:
    """A named prompt template with variable substitution."""

    name: str
    system_prompt: str
    user_template: str
    description: str = ""
    few_shot_examples: list[dict[str, str]] | None = None

    def format_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """
        Render the template into OpenAI-style messages.

        Args:
            **kwargs: Variables to substitute into the user template.

        Returns:
            List of {'role': ..., 'content': ...} message dicts.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add few-shot examples if present
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        # Render user prompt with variable substitution
        user_content = self.user_template.format(**kwargs)
        messages.append({"role": "user", "content": user_content})

        return messages


# ---------------------------------------------------------------------------
# Core RAG QA prompt
# ---------------------------------------------------------------------------

RAG_QA_TEMPLATE = PromptTemplate(
    name="rag_qa",
    description="Standard RAG question-answering with inline citations.",
    system_prompt="""\
You are a precise, scholarly research assistant. Your task is to answer questions
based strictly on the provided context passages. Follow these rules:

1. **Answer from context only.** Do not use outside knowledge or make assumptions.
   If the answer is not present in the context, say so clearly.
2. **Cite your sources.** Use inline citation numbers like [1], [2] corresponding
   to the numbered context passages provided.
3. **Be precise and concise.** Prefer direct answers over verbose explanations.
4. **Acknowledge uncertainty.** If the context is ambiguous or incomplete, say so.
5. **Preserve technical accuracy.** Do not paraphrase technical terms incorrectly.

When the question cannot be answered from the provided context, respond with:
"Based on the provided documents, I cannot find sufficient information to answer this question."
""",
    user_template="""\
Context passages:
{context}

---

Question: {question}

Please provide a comprehensive answer based solely on the context above.
Include inline citations [n] for each claim made.
""",
)


# ---------------------------------------------------------------------------
# Multi-hop reasoning prompt
# ---------------------------------------------------------------------------

MULTI_HOP_TEMPLATE = PromptTemplate(
    name="multi_hop_qa",
    description="Multi-step reasoning across multiple source documents.",
    system_prompt="""\
You are an expert research analyst capable of synthesizing information
across multiple documents to answer complex, multi-step questions.

Your reasoning process:
1. Identify which context passages are relevant to sub-questions
2. Chain reasoning steps explicitly, showing how information connects
3. Synthesize a final answer that integrates findings from multiple sources
4. Cite all supporting passages with [n] notation

If intermediate reasoning steps require information not present in the context,
clearly flag this as a knowledge gap.
""",
    user_template="""\
Context passages:
{context}

---

Complex question: {question}

Step-by-step analysis:
First, identify the key sub-questions needed to answer this question.
Then, address each sub-question using the context, and finally synthesize
a comprehensive answer.
""",
)


# ---------------------------------------------------------------------------
# Summarization prompt
# ---------------------------------------------------------------------------

SUMMARIZE_TEMPLATE = PromptTemplate(
    name="summarize",
    description="Structured document summarization with key themes extraction.",
    system_prompt="""\
You are a skilled academic summarizer. Your task is to produce clear,
structured summaries of research content. Your summaries should:

1. Capture the main thesis or argument
2. Identify key findings, methods, and conclusions
3. Extract important entities, concepts, and terminology
4. Preserve technical accuracy without oversimplification
5. Be organized with clear structure (use headers when appropriate)
""",
    user_template="""\
Please summarize the following document content:

{context}

Provide:
1. **Overview** (2-3 sentences)
2. **Key Findings** (bullet points)
3. **Methodology** (if applicable)
4. **Conclusions**
5. **Notable Concepts/Terms**
""",
)


# ---------------------------------------------------------------------------
# Factual verification prompt
# ---------------------------------------------------------------------------

VERIFY_TEMPLATE = PromptTemplate(
    name="factual_verify",
    description="Verify claims against retrieved context for faithfulness checking.",
    system_prompt="""\
You are a rigorous fact-checker. For each claim presented, determine whether
it is:
- **SUPPORTED**: Directly supported by the context with clear evidence
- **CONTRADICTED**: The context says something different
- **UNVERIFIABLE**: The context does not contain enough information
- **PARTIALLY SUPPORTED**: Context supports part of the claim but not all

Provide specific quotes from the context to justify each classification.
""",
    user_template="""\
Context passages:
{context}

---

Claims to verify:
{claims}

For each claim, provide:
1. Classification (SUPPORTED / CONTRADICTED / UNVERIFIABLE / PARTIALLY SUPPORTED)
2. Evidence: Direct quote or reference from context
3. Explanation: Brief reasoning
""",
)


# ---------------------------------------------------------------------------
# Query rewriting prompt
# ---------------------------------------------------------------------------

QUERY_REWRITE_TEMPLATE = PromptTemplate(
    name="query_rewrite",
    description="Expand and rewrite queries to improve retrieval coverage.",
    system_prompt="""\
You are an expert at search query optimization for academic and technical documents.
Your task is to rewrite and expand queries to improve document retrieval.

For each query, produce:
1. A reformulated version that makes implicit assumptions explicit
2. 2-3 alternative phrasings that may use different terminology
3. Key technical terms and synonyms that should be present in relevant documents

Format your response as a JSON object.
""",
    user_template="""\
Original query: "{query}"

Rewrite this query for optimal semantic search retrieval.
Return a JSON object with this structure:
{{
  "reformulated": "improved version of the query",
  "alternatives": ["alternative phrasing 1", "alternative phrasing 2"],
  "key_terms": ["term1", "term2", "term3"]
}}
""",
)


# ---------------------------------------------------------------------------
# Conversation-aware RAG prompt
# ---------------------------------------------------------------------------

CONVERSATIONAL_RAG_TEMPLATE = PromptTemplate(
    name="conversational_rag",
    description="RAG prompt with conversation history awareness.",
    system_prompt="""\
You are a knowledgeable research assistant engaged in an ongoing conversation.
You have access to relevant document excerpts to inform your answers.

Guidelines:
- Use the conversation history for context and continuity
- Prioritize information from the retrieved passages when answering
- Acknowledge and build upon previous exchanges naturally
- Cite sources with [n] notation when drawing from context passages
- If a follow-up question refers to something from prior exchanges,
  acknowledge that connection explicitly
""",
    user_template="""\
Conversation history:
{history}

---

Retrieved context:
{context}

---

Current question: {question}
""",
)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """
    Registry for managing and accessing prompt templates.

    Example:
        registry = PromptRegistry()
        template = registry.get("rag_qa")
        messages = template.format_messages(context="...", question="...")
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all built-in templates."""
        for template in [
            RAG_QA_TEMPLATE,
            MULTI_HOP_TEMPLATE,
            SUMMARIZE_TEMPLATE,
            VERIFY_TEMPLATE,
            QUERY_REWRITE_TEMPLATE,
            CONVERSATIONAL_RAG_TEMPLATE,
        ]:
            self._templates[template.name] = template

    def register(self, template: PromptTemplate) -> None:
        """Add or replace a template in the registry."""
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """Retrieve a template by name."""
        if name not in self._templates:
            available = list(self._templates.keys())
            raise KeyError(
                f"Template {name!r} not found. Available: {available}"
            )
        return self._templates[name]

    def list_templates(self) -> list[dict[str, str]]:
        """Return a list of all registered templates with descriptions."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._templates.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._templates


# Singleton registry for application-wide use
_default_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Return the global prompt registry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry()
    return _default_registry
