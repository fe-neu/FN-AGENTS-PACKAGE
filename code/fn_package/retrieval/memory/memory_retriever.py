from __future__ import annotations
from typing import List
import time
import numpy as np
from fn_package.utils.logger import get_logger
from ..core.retriever import Retriever, Hit, T
from .memory_record import MemoryRecord
from fn_package.config import DEFAULT_MEMORY_RECENCY_WEIGHT, DEFAULT_MEMORY_IMPORTANCE_WEIGHT

logger = get_logger(__name__)


class MemoryRetriever(Retriever[MemoryRecord]):
    """
    Extended retriever specialized for memory records.

    Enhancements over the base Retriever:
    - Computes cosine similarity between query and memory embeddings.
    - Adjusts scores by recency: newer memories are weighted more strongly.
    - Optionally adjusts scores by importance if specified.
    """

    def __init__(
            self,
            store,
            recency_weight: float = DEFAULT_MEMORY_RECENCY_WEIGHT,
            importance_weight: float = DEFAULT_MEMORY_IMPORTANCE_WEIGHT
            ):
        """
        Initialize the MemoryRetriever.

        Parameters
        ----------
        store : VectorStore[MemoryRecord]
            The underlying vector store containing memory records.
        recency_weight : float, optional
            Weighting factor for recency (default from config).
        importance_weight : float, optional
            Weighting factor for importance (default from config).
        """
        super().__init__(store)
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        logger.info(
            "MemoryRetriever initialized (recency_weight=%.2f, importance_weight=%.2f)",
            recency_weight,
            importance_weight,
        )

    def _adjust_score(self, hit: Hit[MemoryRecord]) -> float:
        """
        Adjust the similarity score of a hit by applying recency and importance factors.

        Parameters
        ----------
        hit : Hit[MemoryRecord]
            A retrieval hit containing a memory record and its base similarity score.

        Returns
        -------
        float
            The adjusted score.
        """
        base = hit.score

        # Recency adjustment: newer records get higher weight
        age_sec = time.time() - hit.record.created_at
        age_days = age_sec / 86400
        recency_factor = np.exp(-self.recency_weight * age_days)

        # Importance adjustment: higher importance increases score
        imp_factor = 1.0 + self.importance_weight * (hit.record.importance - 0.5)

        return base * recency_factor * imp_factor

    def topk_by_embedding(self, q_vec: np.ndarray, k: int = 5) -> List[Hit[MemoryRecord]]:
        """
        Retrieve the top-k most relevant memory records, adjusted for recency and importance.

        Parameters
        ----------
        q_vec : np.ndarray
            Query embedding vector of shape (dim,).
        k : int, optional
            Number of results to return (default = 5).

        Returns
        -------
        List[Hit[MemoryRecord]]
            A list of hits sorted by adjusted score in descending order.
        """
        hits = super().topk_by_embedding(q_vec, k=k)
        hits = [Hit(h.record, self._adjust_score(h)) for h in hits]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def all_above_threshold(self, q_vec: np.ndarray, min_score: float) -> List[Hit[MemoryRecord]]:
        """
        Retrieve all memory records with adjusted scores above a threshold.

        Parameters
        ----------
        q_vec : np.ndarray
            Query embedding vector of shape (dim,).
        min_score : float
            Minimum base similarity score required before adjustment.

        Returns
        -------
        List[Hit[MemoryRecord]]
            A list of hits sorted by adjusted score in descending order.
        """
        hits = super().all_above_threshold(q_vec, min_score)
        hits = [Hit(h.record, self._adjust_score(h)) for h in hits]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits
