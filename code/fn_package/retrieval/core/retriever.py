from __future__ import annotations
from dataclasses import dataclass
from typing import List, TypeVar, Generic
import numpy as np
from fn_package.utils.logger import get_logger
from .vector_store import VectorStore
from ..core.base_record import BaseRecord  # Abstract base class for records

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseRecord)


@dataclass
class Hit(Generic[T]):
    """
    Represents a single retrieval result.

    Attributes
    ----------
    record : T
        The retrieved record.
    score : float
        The similarity score associated with the record.
    """
    record: T
    score: float


class Retriever(Generic[T]):
    """
    Generic retriever for fetching records from a vector store
    using cosine similarity.
    """

    def __init__(self, store: VectorStore[T]):
        """
        Initialize the retriever with a vector store.

        Parameters
        ----------
        store : VectorStore[T]
            The vector store containing embeddings and records.
        """
        self.store = store
        logger.info("Retriever initialized.")

    def _cosine_similarity(self, q_vec: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query vector and all embeddings.

        Parameters
        ----------
        q_vec : np.ndarray
            Query embedding vector of shape (dim,).
        E : np.ndarray
            Matrix of stored embeddings of shape (N, dim).

        Returns
        -------
        np.ndarray
            Similarity scores of shape (N,).
        """
        q_norm = np.linalg.norm(q_vec) + 1e-8
        E_norms = np.linalg.norm(E, axis=1) + 1e-8
        sims = (E @ q_vec) / (E_norms * q_norm)
        return sims

    def topk_by_embedding(self, q_vec: np.ndarray, k: int = 5) -> List[Hit[T]]:
        """
        Retrieve the top-k most similar records to the given query vector.

        Parameters
        ----------
        q_vec : np.ndarray
            Query embedding vector of shape (dim,).
        k : int, optional
            Number of top results to return (default = 5).

        Returns
        -------
        List[Hit[T]]
            A list of retrieval hits, sorted by similarity score in descending order.

        Raises
        ------
        ValueError
            If the query vector dimension does not match the store's embedding dimension.
        """
        E = self.store.E
        n = E.shape[0]
        if n == 0:
            logger.debug("topk_by_embedding: empty store")
            return []
        if q_vec.shape != (self.store.dim,):
            raise ValueError(
                f"Query dim mismatch: got {q_vec.shape}, expected ({self.store.dim},)"
            )

        scores = self._cosine_similarity(q_vec, E)
        k_eff = min(k, n)
        idxs = np.argpartition(scores, -k_eff)[-k_eff:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]

        hits = [Hit(self.store.records[i], float(scores[i])) for i in idxs]
        logger.debug(
            "topk_by_embedding: k=%s, best_id=%s",
            k_eff,
            hits[0].record.id if hits else None,
        )
        return hits

    def all_above_threshold(self, q_vec: np.ndarray, min_score: float) -> List[Hit[T]]:
        """
        Retrieve all records with similarity scores above a given threshold.

        Parameters
        ----------
        q_vec : np.ndarray
            Query embedding vector of shape (dim,).
        min_score : float
            Minimum similarity score required for a record to be returned.

        Returns
        -------
        List[Hit[T]]
            A list of retrieval hits that meet or exceed the threshold,
            sorted by similarity score in descending order.

        Raises
        ------
        ValueError
            If the query vector dimension does not match the store's embedding dimension.
        """
        E = self.store.E
        n = E.shape[0]
        if n == 0:
            logger.debug("all_above_threshold: empty store")
            return []
        if q_vec.shape != (self.store.dim,):
            raise ValueError(
                f"Query dim mismatch: got {q_vec.shape}, expected ({self.store.dim},)"
            )

        scores = self._cosine_similarity(q_vec, E)
        idxs = np.where(scores >= min_score)[0]
        if idxs.size == 0:
            logger.debug("all_above_threshold: no hits >= %s", min_score)
            return []

        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        hits = [Hit(self.store.records[i], float(scores[i])) for i in idxs]
        logger.debug(
            "all_above_threshold: hits=%s (min_score=%s)", len(hits), min_score
        )
        return hits
