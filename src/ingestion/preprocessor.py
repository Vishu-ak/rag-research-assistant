"""
Text preprocessing and metadata extraction for RAG document pipelines.

This module provides tools to clean, normalize, and enrich document chunks
before embedding. It also extracts structured metadata (section headings,
authors, dates, DOIs, etc.) from document text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import structlog
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


@dataclass
class DocumentMetadata:
    """Structured metadata extracted from document text."""

    title: str | None = None
    authors: list[str] | None = None
    date: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    keywords: list[str] | None = None
    section_headings: list[str] | None = None
    language: str = "en"
    word_count: int = 0
    sentence_count: int = 0


_DOI_PATTERN = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE
)
_ARXIV_PATTERN = re.compile(
    r"arxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE
)
_DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})\b"),  # Year only
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b"),  # MM/DD/YYYY
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
]
_HEADING_PATTERN = re.compile(
    r"^(?:\d+\.?\s+)?([A-Z][A-Z\s]{3,60})$", re.MULTILINE
)
_ABSTRACT_PATTERN = re.compile(
    r"(?:Abstract|ABSTRACT)[.\s—:-]+(.{100,2000}?)(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_KEYWORDS_PATTERN = re.compile(
    r"(?:Keywords?|Key\s+[Ww]ords?)[:\s—]+(.{10,300})(?=\n|\Z)", re.IGNORECASE
)


def extract_metadata(text: str, existing_metadata: dict[str, Any] | None = None) -> DocumentMetadata:
    """
    Extract structured metadata from raw document text.

    Args:
        text: Full document text or first-page text.
        existing_metadata: Optional dict of already-known metadata to merge.

    Returns:
        Populated DocumentMetadata instance.
    """
    meta = DocumentMetadata()
    existing = existing_metadata or {}

    # DOI
    doi_match = _DOI_PATTERN.search(text)
    meta.doi = doi_match.group(1) if doi_match else existing.get("doi")

    # ArXiv ID
    arxiv_match = _ARXIV_PATTERN.search(text)
    meta.arxiv_id = arxiv_match.group(1) if arxiv_match else existing.get("arxiv_id")

    # Date (try multiple patterns; prefer most specific)
    for pattern in _DATE_PATTERNS:
        date_match = pattern.search(text[:3000])
        if date_match:
            meta.date = date_match.group(0)
            break

    # Abstract
    abstract_match = _ABSTRACT_PATTERN.search(text)
    if abstract_match:
        meta.abstract = abstract_match.group(1).strip()[:1000]

    # Keywords
    kw_match = _KEYWORDS_PATTERN.search(text)
    if kw_match:
        raw_kw = kw_match.group(1).strip()
        # Try comma or semicolon separation
        sep = ";" if ";" in raw_kw else ","
        meta.keywords = [k.strip() for k in raw_kw.split(sep) if k.strip()][:20]

    # Section headings from lines that look like ALL-CAPS titles
    heading_matches = _HEADING_PATTERN.findall(text)
    meta.section_headings = list(dict.fromkeys(h.strip() for h in heading_matches))[:20]

    # Word and sentence counts
    words = text.split()
    meta.word_count = len(words)
    meta.sentence_count = len(re.findall(r"[.!?]+", text))

    # Title heuristic: first non-empty line that looks like a title
    for line in text[:2000].split("\n"):
        line = line.strip()
        if 10 < len(line) < 200 and not line.endswith("."):
            meta.title = line
            break

    return meta


# ---------------------------------------------------------------------------
# Text cleaning pipeline
# ---------------------------------------------------------------------------


class TextCleaningPipeline:
    """
    Configurable cleaning pipeline for extracted document text.

    Applies a sequence of cleaning steps to normalize text before
    chunking and embedding. Each step is independently togglable.

    Example:
        pipeline = TextCleaningPipeline(remove_urls=True, normalize_whitespace=True)
        clean_text = pipeline.clean("Hello  world!   Visit http://example.com")
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = False,
        remove_special_chars: bool = False,
        remove_numbers: bool = False,
        normalize_whitespace: bool = True,
        fix_unicode: bool = True,
        remove_headers_footers: bool = True,
        min_word_length: int = 0,
        max_repeated_chars: int = 3,
    ) -> None:
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace
        self.fix_unicode = fix_unicode
        self.remove_headers_footers = remove_headers_footers
        self.min_word_length = min_word_length
        self.max_repeated_chars = max_repeated_chars

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline to a text string."""
        if not text:
            return text

        if self.fix_unicode:
            text = self._fix_unicode(text)

        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)

        if self.remove_urls:
            text = self._remove_urls(text)

        if self.remove_emails:
            text = self._remove_emails(text)

        if self.remove_special_chars:
            text = self._remove_special_chars(text)

        if self.remove_numbers:
            text = re.sub(r"\b\d+\b", "", text)

        if self.max_repeated_chars > 0:
            text = self._limit_repeated_chars(text, self.max_repeated_chars)

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.lowercase:
            text = text.lower()

        return text.strip()

    def clean_documents(self, documents: list[Document]) -> list[Document]:
        """Apply cleaning to a list of LangChain Documents in-place."""
        cleaned: list[Document] = []
        for doc in documents:
            original_len = len(doc.page_content)
            clean_content = self.clean(doc.page_content)

            if not clean_content.strip():
                logger.debug(
                    "preprocessor.doc_dropped_empty",
                    source=doc.metadata.get("source", "unknown"),
                )
                continue

            cleaned_doc = Document(
                page_content=clean_content,
                metadata={
                    **doc.metadata,
                    "original_length": original_len,
                    "cleaned_length": len(clean_content),
                    "cleaned_at": datetime.utcnow().isoformat(),
                },
            )
            cleaned.append(cleaned_doc)

        logger.info(
            "preprocessor.cleaning_complete",
            input_count=len(documents),
            output_count=len(cleaned),
            dropped=len(documents) - len(cleaned),
        )
        return cleaned

    @staticmethod
    def _fix_unicode(text: str) -> str:
        """Normalize unicode, fix common encoding issues."""
        # NFKC normalization: compatibility decomposition + canonical composition
        text = unicodedata.normalize("NFKC", text)
        # Remove zero-width characters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove HTTP/HTTPS URLs and bare domain references."""
        text = re.sub(
            r"https?://[^\s<>\"]+|www\.[^\s<>\"]+",
            "[URL]",
            text,
        )
        return text

    @staticmethod
    def _remove_emails(text: str) -> str:
        """Replace email addresses with a placeholder."""
        return re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]",
            text,
        )

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Remove non-alphanumeric characters except common punctuation."""
        return re.sub(r"[^a-zA-Z0-9\s.,;:!?'\"-]", " ", text)

    @staticmethod
    def _limit_repeated_chars(text: str, max_repeats: int) -> str:
        """Collapse runs of repeated characters (e.g., 'aaaa' → 'aaa')."""
        pattern = re.compile(r"(.)\1{" + str(max_repeats) + r",}")
        return pattern.sub(r"\1" * max_repeats, text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize all whitespace to single spaces or single newlines."""
        # Collapse horizontal whitespace (spaces/tabs) within lines
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse more than 2 newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _remove_headers_footers(text: str) -> str:
        """
        Heuristically remove page headers and footers.

        Targets short lines that appear at the start or end of
        paragraphs that look like repeated page metadata (page numbers,
        journal names, etc.).
        """
        lines = text.split("\n")
        filtered: list[str] = []

        for line in lines:
            stripped = line.strip()
            # Skip purely numeric lines (page numbers)
            if re.fullmatch(r"\d+", stripped):
                continue
            # Skip very short lines with no alphabetic content
            if len(stripped) < 4 and not any(c.isalpha() for c in stripped):
                continue
            filtered.append(line)

        return "\n".join(filtered)


# ---------------------------------------------------------------------------
# Document deduplication
# ---------------------------------------------------------------------------


class DocumentDeduplicator:
    """
    Near-duplicate document detection and removal.

    Uses SimHash fingerprinting to detect and drop chunks that are
    highly similar to already-seen content (e.g. repeated legal boilerplate,
    table-of-contents duplicates, etc.).
    """

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        """
        Args:
            similarity_threshold: Fraction of matching bits required
                to classify two docs as duplicates (0.0 to 1.0).
        """
        self.similarity_threshold = similarity_threshold
        self._seen_hashes: dict[int, str] = {}

    def deduplicate(self, documents: list[Document]) -> list[Document]:
        """
        Remove near-duplicate documents from a list.

        Args:
            documents: Input document list.

        Returns:
            Deduplicated document list preserving order.
        """
        unique: list[Document] = []
        dropped = 0

        for doc in documents:
            fingerprint = self._simhash(doc.page_content)

            if self._is_duplicate(fingerprint):
                dropped += 1
                logger.debug(
                    "deduplicator.dropped",
                    source=doc.metadata.get("source", "?"),
                    chunk=doc.metadata.get("chunk_index", "?"),
                )
                continue

            self._seen_hashes[fingerprint] = doc.metadata.get("source", "unknown")
            unique.append(doc)

        logger.info(
            "deduplicator.complete",
            total=len(documents),
            unique=len(unique),
            dropped=dropped,
        )
        return unique

    def _is_duplicate(self, fingerprint: int) -> bool:
        """Check if fingerprint is similar to any seen hash."""
        for seen_fp in self._seen_hashes:
            if self._hamming_similarity(fingerprint, seen_fp) >= self.similarity_threshold:
                return True
        return False

    @staticmethod
    def _simhash(text: str, bits: int = 64) -> int:
        """
        Compute a SimHash fingerprint for the given text.

        SimHash projects the text into a bit vector space where
        similar documents have similar (close Hamming distance) hashes.
        """
        import hashlib

        words = text.lower().split()
        vector = [0] * bits

        for word in set(words):
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(bits):
                bit = (word_hash >> i) & 1
                vector[i] += 1 if bit else -1

        fingerprint = 0
        for i in range(bits):
            if vector[i] > 0:
                fingerprint |= 1 << i

        return fingerprint

    @staticmethod
    def _hamming_similarity(a: int, b: int) -> float:
        """Return fraction of matching bits between two integers."""
        xor = a ^ b
        differing_bits = bin(xor).count("1")
        return 1.0 - differing_bits / 64.0


# ---------------------------------------------------------------------------
# High-level pipeline function
# ---------------------------------------------------------------------------


def preprocess_documents(
    documents: list[Document],
    clean_text: bool = True,
    deduplicate: bool = True,
    extract_doc_metadata: bool = True,
    cleaner: TextCleaningPipeline | None = None,
) -> list[Document]:
    """
    Full preprocessing pipeline for a list of documents.

    Applies cleaning → metadata enrichment → deduplication in order.

    Args:
        documents: Raw documents from PDF loader or text splitter.
        clean_text: Whether to apply text cleaning.
        deduplicate: Whether to drop near-duplicate chunks.
        extract_doc_metadata: Whether to extract and attach document metadata.
        cleaner: Custom cleaning pipeline; defaults to a sensible preset.

    Returns:
        Preprocessed, enriched, and deduplicated document list.
    """
    if not documents:
        return []

    logger.info("preprocessor.pipeline_start", doc_count=len(documents))

    if clean_text:
        pipeline = cleaner or TextCleaningPipeline()
        documents = pipeline.clean_documents(documents)

    if extract_doc_metadata:
        # Group by source, extract metadata from first page/chunk of each source
        source_groups: dict[str, list[Document]] = {}
        for doc in documents:
            src = doc.metadata.get("source", "__unknown__")
            source_groups.setdefault(src, []).append(doc)

        for source, docs in source_groups.items():
            # Use first doc to extract metadata
            full_text = docs[0].page_content
            meta = extract_metadata(full_text, docs[0].metadata)
            meta_dict = {
                k: v for k, v in asdict(meta).items() if v is not None
            }
            for doc in docs:
                doc.metadata.update(meta_dict)

    if deduplicate:
        deduper = DocumentDeduplicator()
        documents = deduper.deduplicate(documents)

    logger.info("preprocessor.pipeline_complete", doc_count=len(documents))
    return documents
