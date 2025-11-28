"""
PDF document loading and parsing module.

Supports multiple PDF parsing backends (pypdf, pdfminer) with automatic
fallback, metadata extraction, and page-level tracking.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from pathlib import Path
from typing import Iterator

import structlog
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


class PDFDocument:
    """Container for a parsed PDF document with metadata."""

    def __init__(
        self,
        pages: list[Document],
        source_path: str,
        file_hash: str,
        total_pages: int,
        parse_backend: str,
        parse_duration_ms: float,
    ) -> None:
        self.pages = pages
        self.source_path = source_path
        self.file_hash = file_hash
        self.total_pages = total_pages
        self.parse_backend = parse_backend
        self.parse_duration_ms = parse_duration_ms

    @property
    def full_text(self) -> str:
        """Return concatenated text from all pages."""
        return "\n\n".join(page.page_content for page in self.pages)

    def __repr__(self) -> str:
        return (
            f"PDFDocument(source={Path(self.source_path).name!r}, "
            f"pages={self.total_pages}, "
            f"backend={self.parse_backend!r})"
        )


def _compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_with_pypdf(path: str) -> list[Document]:
    """Load PDF using pypdf (fast, works on most modern PDFs)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:
        raise PDFLoadError("pypdf not installed. Run: pip install pypdf") from e

    reader = PdfReader(path)
    pages: list[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            text = _clean_extracted_text(text)
            if not text.strip():
                logger.debug("pdf.page_empty", path=path, page=page_num)
                continue

            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "page": page_num,
                        "total_pages": len(reader.pages),
                        "parse_backend": "pypdf",
                    },
                )
            )
        except Exception as exc:
            logger.warning(
                "pdf.page_extraction_error",
                path=path,
                page=page_num,
                error=str(exc),
            )

    return pages


def _load_with_pdfminer(path: str) -> list[Document]:
    """Load PDF using pdfminer.six (slower but more accurate layout analysis)."""
    try:
        from pdfminer.high_level import extract_pages  # type: ignore
        from pdfminer.layout import LAParams, LTTextContainer  # type: ignore
    except ImportError as e:
        raise PDFLoadError(
            "pdfminer.six not installed. Run: pip install pdfminer.six"
        ) from e

    laparams = LAParams(
        line_margin=0.5,
        word_margin=0.1,
        char_margin=2.0,
        boxes_flow=0.5,
        detect_vertical=False,
    )

    pages: list[Document] = []

    for page_num, page_layout in enumerate(
        extract_pages(path, laparams=laparams), start=1
    ):
        page_text_parts: list[str] = []

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text()
                if text.strip():
                    page_text_parts.append(text)

        full_text = _clean_extracted_text(" ".join(page_text_parts))
        if not full_text.strip():
            continue

        pages.append(
            Document(
                page_content=full_text,
                metadata={
                    "source": path,
                    "page": page_num,
                    "parse_backend": "pdfminer",
                },
            )
        )

    # Backfill total_pages now that we know it
    for doc in pages:
        doc.metadata["total_pages"] = len(pages)

    return pages


def _clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.

    Handles common PDF extraction artifacts:
    - Ligature substitution (ﬁ→fi, ﬀ→ff, etc.)
    - Hyphenated line break removal
    - Whitespace normalization
    - Control character removal
    """
    # Ligature replacement
    ligatures = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "—",
        "\u00ad": "",  # soft hyphen
    }
    for char, replacement in ligatures.items():
        text = text.replace(char, replacement)

    # Remove hyphenated line breaks (word- \ncontinued → wordcontinued)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize whitespace: collapse multiple spaces, normalize tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove control characters but preserve newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse more than two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


class PDFLoader:
    """
    Production PDF loader with multi-backend support and fallback.

    Attempts pypdf first (fast), then falls back to pdfminer.six
    for difficult PDFs. Supports both file-path and bytes-IO input.

    Example:
        loader = PDFLoader()
        doc = loader.load("/path/to/paper.pdf")
        print(f"Loaded {doc.total_pages} pages")
        for page in doc.pages:
            print(page.page_content[:200])
    """

    def __init__(
        self,
        preferred_backend: str = "pypdf",
        fallback_to_pdfminer: bool = True,
        min_chars_per_page: int = 50,
    ) -> None:
        """
        Args:
            preferred_backend: Primary parsing backend ('pypdf' or 'pdfminer').
            fallback_to_pdfminer: If pypdf fails or returns sparse text, try pdfminer.
            min_chars_per_page: Pages with fewer chars are considered extraction failures.
        """
        self.preferred_backend = preferred_backend
        self.fallback_to_pdfminer = fallback_to_pdfminer
        self.min_chars_per_page = min_chars_per_page

        logger.info(
            "pdf_loader.initialized",
            backend=preferred_backend,
            fallback=fallback_to_pdfminer,
        )

    def load(self, path: str | Path) -> PDFDocument:
        """
        Load a single PDF file.

        Args:
            path: Path to the PDF file.

        Returns:
            PDFDocument with parsed pages and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            PDFLoadError: If parsing fails with all backends.
        """
        path = str(Path(path).resolve())

        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not path.lower().endswith(".pdf"):
            logger.warning("pdf_loader.not_pdf_extension", path=path)

        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("pdf_loader.loading", path=path, size_mb=round(file_size_mb, 2))

        start = time.perf_counter()
        file_hash = _compute_file_hash(path)

        pages, backend_used = self._load_with_fallback(path)

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "pdf_loader.loaded",
            path=path,
            pages=len(pages),
            backend=backend_used,
            duration_ms=round(duration_ms, 1),
        )

        return PDFDocument(
            pages=pages,
            source_path=path,
            file_hash=file_hash,
            total_pages=len(pages),
            parse_backend=backend_used,
            parse_duration_ms=duration_ms,
        )

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        glob_pattern: str = "**/*.pdf",
    ) -> Iterator[PDFDocument]:
        """
        Load all PDFs from a directory.

        Args:
            directory: Root directory to search.
            recursive: If True, search subdirectories recursively.
            glob_pattern: Glob pattern for file discovery.

        Yields:
            PDFDocument for each successfully parsed PDF.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pdf_paths = list(directory.glob(glob_pattern) if recursive else directory.glob("*.pdf"))
        logger.info("pdf_loader.directory_scan", path=str(directory), files_found=len(pdf_paths))

        for pdf_path in sorted(pdf_paths):
            try:
                yield self.load(pdf_path)
            except (PDFLoadError, FileNotFoundError) as exc:
                logger.error(
                    "pdf_loader.file_failed",
                    path=str(pdf_path),
                    error=str(exc),
                )

    def load_bytes(self, data: bytes, filename: str = "document.pdf") -> PDFDocument:
        """
        Load a PDF from raw bytes (e.g., from an HTTP upload).

        Args:
            data: Raw PDF bytes.
            filename: Logical filename for metadata.

        Returns:
            PDFDocument with parsed content.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            doc = self.load(tmp_path)
            # Overwrite source path with the logical filename
            for page in doc.pages:
                page.metadata["source"] = filename
            doc.source_path = filename
            return doc
        finally:
            os.unlink(tmp_path)

    def _load_with_fallback(self, path: str) -> tuple[list[Document], str]:
        """
        Attempt loading with the preferred backend, falling back if needed.

        Returns:
            Tuple of (pages, backend_name_used).
        """
        loaders = {
            "pypdf": _load_with_pypdf,
            "pdfminer": _load_with_pdfminer,
        }

        # Ordered list of backends to try
        backends = [self.preferred_backend]
        if self.fallback_to_pdfminer and self.preferred_backend != "pdfminer":
            backends.append("pdfminer")

        last_error: Exception | None = None

        for backend_name in backends:
            loader_fn = loaders.get(backend_name)
            if loader_fn is None:
                continue
            try:
                pages = loader_fn(path)

                if pages and self._is_extraction_poor(pages):
                    logger.warning(
                        "pdf_loader.sparse_extraction",
                        backend=backend_name,
                        path=path,
                        avg_chars=self._avg_chars(pages),
                    )
                    if self.fallback_to_pdfminer and backend_name != "pdfminer":
                        continue  # Try next backend

                return pages, backend_name

            except Exception as exc:
                logger.warning(
                    "pdf_loader.backend_failed",
                    backend=backend_name,
                    path=path,
                    error=str(exc),
                )
                last_error = exc

        raise PDFLoadError(
            f"All PDF backends failed for {path}. Last error: {last_error}"
        )

    def _is_extraction_poor(self, pages: list[Document]) -> bool:
        """Return True if the average extracted text per page is suspiciously short."""
        if not pages:
            return True
        return self._avg_chars(pages) < self.min_chars_per_page

    @staticmethod
    def _avg_chars(pages: list[Document]) -> float:
        if not pages:
            return 0.0
        return sum(len(p.page_content) for p in pages) / len(pages)
