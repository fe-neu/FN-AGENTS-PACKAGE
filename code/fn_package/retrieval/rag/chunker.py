from __future__ import annotations
from typing import List
import tiktoken
from fn_package.utils.logger import get_logger
from fn_package.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBED_MODEL,
)

logger = get_logger(__name__)


class Chunker:
    """
    Utility class for splitting text into chunks based on token length.

    Uses the `tiktoken` library to respect token budgets for LLMs or embedding models.
    Supports overlapping chunks to preserve context across boundaries.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize the Chunker.

        Parameters
        ----------
        model : str, optional
            Model name to determine tokenizer (default from config).
        chunk_size : int, optional
            Maximum number of tokens per chunk (default from config).
        overlap : int, optional
            Number of tokens to overlap between consecutive chunks (default from config).

        Raises
        ------
        Exception
            If tokenizer initialization for the given model fails.
        """
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.enc = tiktoken.encoding_for_model(model)
            logger.info(f"TokenChunker initialized with model={model}")
        except Exception as e:
            logger.error(f"Failed to init TokenChunker for model={model}: {e}")
            raise

    def split(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on token length.

        Parameters
        ----------
        text : str
            The input text to be chunked.

        Returns
        -------
        List[str]
            A list of chunked strings, each within the defined token budget.
        """
        tokens = self.enc.encode(text)
        chunks: List[str] = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.enc.decode(chunk_tokens)
            chunks.append(chunk_text)

            logger.debug(
                f"Created chunk tokens[{start}:{end}] "
                f"(len={len(chunk_tokens)})"
            )

            # Move the window forward with overlap
            start = end - self.overlap
            if start < 0:
                start = 0

        logger.info(
            f"Split text of {len(tokens)} tokens into {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.overlap})"
        )
        return chunks
