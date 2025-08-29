from __future__ import annotations
import os
import re
from typing import Optional
from fn_package.utils.logger import get_logger

logger = get_logger(__name__)

try:
    # pypdf is the successor of PyPDF2
    from pypdf import PdfReader
except Exception as e:
    PdfReader = None
    _import_error = e


class PDFParser:
    """
    Minimal PDF text parser using pypdf.

    Features
    --------
    - Reads all pages (or up to a maximum if specified).
    - Concatenates text from pages into one string.
    - Applies light whitespace normalization.
    - Does not track page numbers – intentionally kept simple.
    """

    def __init__(self, max_pages: Optional[int] = None):
        """
        Initialize the PDF parser.

        Parameters
        ----------
        max_pages : int, optional
            Limit parsing to the first N pages. If None, parses the whole document.

        Raises
        ------
        RuntimeError
            If `pypdf` is not available.
        """
        self.max_pages = max_pages

        if PdfReader is None:
            raise RuntimeError(
                f"pypdf not available: {_import_error}. "
                "Install it with `pip install pypdf`."
            )

    def parse(self, path: str) -> str:
        """
        Parse a PDF file and return its extracted text.

        Parameters
        ----------
        path : str
            Path to the PDF file.

        Returns
        -------
        str
            Extracted and cleaned text.

        Raises
        ------
        ValueError
            If path is not a non-empty string.
        FileNotFoundError
            If the file does not exist.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("parse(path): 'path' must be a non-empty string.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("PDFParser.parse: opening '%s'", path)
        reader = PdfReader(path)

        num_pages_total = len(reader.pages)
        limit = self.max_pages if self.max_pages is not None else num_pages_total
        num_pages = min(limit, num_pages_total)

        texts = []
        for i in range(num_pages):
            page = reader.pages[i]
            raw = page.extract_text() or ""
            texts.append(raw)
            logger.debug("PDFParser.parse: extracted page %s (%s chars)", i + 1, len(raw))

        combined = "\n\n".join(texts)
        cleaned = self._clean_text(combined)

        logger.info(
            "PDFParser.parse: parsed %s/%s pages → %s chars (cleaned)",
            num_pages, num_pages_total, len(cleaned)
        )
        return cleaned

    @staticmethod
    def _clean_text(s: str) -> str:
        """
        Basic text cleanup.

        Steps
        -----
        - Normalize line breaks to spaces.
        - Replace tabs and non-breaking spaces with regular spaces.
        - Collapse multiple consecutive spaces into one.
        - Strip leading and trailing whitespace.

        Parameters
        ----------
        s : str
            Raw text.

        Returns
        -------
        str
            Cleaned text.
        """
        if not s:
            return ""

        # Normalize line breaks
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # Replace newlines with spaces
        s = s.replace("\n", " ")
        # Replace non-breaking spaces and tabs
        s = s.replace("\u00A0", " ").replace("\t", " ")
        # Collapse multiple spaces
        s = re.sub(r"[ ]{2,}", " ", s)
        return s.strip()
