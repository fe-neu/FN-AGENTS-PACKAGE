from __future__ import annotations
import numpy as np
from fn_package.utils.logger import get_logger
from fn_package.config import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_DIM, OPENAI_API_KEY

logger = get_logger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIEmbedder:
    """
    Wrapper class for generating embeddings using the OpenAI API.

    Provides a simple interface to create embeddings from text and return them
    as NumPy arrays. The embedding dimension is checked and updated if it differs
    from the expected default.

    Methods
    -------
    embed(text: str) -> np.ndarray
        Generate an embedding for the given text.
    """

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, dim: int = DEFAULT_EMBED_DIM):
        """
        Initialize the OpenAIEmbedder.

        Parameters
        ----------
        model : str, optional
            Name of the OpenAI embedding model to use (default from config).
        dim : int, optional
            Expected embedding dimensionality (default from config).

        Raises
        ------
        RuntimeError
            If the OpenAI package is not installed or the API key is missing.
        """
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run `pip install openai`")

        if OPENAI_API_KEY is None:
            raise RuntimeError("OPENAI_API_KEY not set in environment/.env")

        self.model = model
        self.dim = dim
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        logger.info(f"OpenAIEmbedder initialized with model={model}, dim={dim}")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.

        Parameters
        ----------
        text : str
            The input text for which to generate an embedding.

        Returns
        -------
        np.ndarray
            A NumPy array representing the embedding vector.

        Raises
        ------
        TypeError
            If `text` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        logger.debug(f"Embedding text (len={len(text)}) with model={self.model}")
        resp = self.client.embeddings.create(model=self.model, input=[text])
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)

        # Update dimension if it does not match the expected size
        if vec.shape[0] != self.dim:
            logger.warning(
                f"Embedding dimension mismatch: got {vec.shape[0]}, expected {self.dim} (updating dim)"
            )
            self.dim = int(vec.shape[0])

        return vec
