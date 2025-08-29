from fn_package.utils.logger import get_logger
from typing import Optional, List, Dict, TypeVar, Generic
import numpy as np
from ..core.base_record import BaseRecord  # Abstract base class for records

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseRecord)


class VectorStore(Generic[T]):
    """
    A simple in-memory vector store for storing records and their embeddings.

    Attributes
    ----------
    dim : int
        The dimensionality of the stored embeddings.
    E : np.ndarray
        Matrix of embeddings with shape (N, dim).
    ids : List[str]
        List of record IDs corresponding to embeddings.
    records : List[T]
        List of stored record objects.
    id_to_row : Dict[str, int]
        Mapping from record ID to row index in E.
    """

    def __init__(self, dim: int):
        """
        Initialize an empty VectorStore.

        Parameters
        ----------
        dim : int
            Dimensionality of the embeddings.
        """
        self.dim = dim
        self.E = np.empty((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.records: List[T] = []
        self.id_to_row: Dict[str, int] = {}
        logger.info(f"Initialized VectorStore with dim={dim}")

    def add(self, rec: T) -> None:
        """
        Add a record to the store.

        Parameters
        ----------
        rec : T
            Record containing an embedding of shape (dim,).

        Raises
        ------
        AssertionError
            If the record's embedding dimension does not match the store's.
        """
        assert rec.embedding.shape == (self.dim,), "bad embedding dim"
        row = self.E.shape[0]
        self.E = np.vstack([self.E, rec.embedding[np.newaxis, :]])
        self.ids.append(rec.id)
        self.records.append(rec)
        self.id_to_row[rec.id] = row
        logger.debug(f"Added record id={rec.id} at row={row}")

    def delete_by_id(self, del_id: str) -> bool:
        """
        Delete a record by its ID.

        Parameters
        ----------
        del_id : str
            ID of the record to delete.

        Returns
        -------
        bool
            True if the record was deleted, False if not found.
        """
        idx = self.id_to_row.get(del_id)
        if idx is None:
            logger.warning(f"Attempted to delete non-existent id={del_id}")
            return False

        last = self.E.shape[0] - 1
        if idx != last:
            # Move last record into the deleted record's position
            self.E[idx, :] = self.E[last, :]
            self.ids[idx] = self.ids[last]
            self.records[idx] = self.records[last]
            self.id_to_row[self.ids[idx]] = idx
            logger.debug(f"Moved last record to idx={idx} during delete of id={del_id}")

        # Remove the last row/record
        self.E = self.E[:last, :]
        self.id_to_row.pop(del_id, None)
        self.ids.pop()
        self.records.pop()
        logger.info(f"Deleted record id={del_id}")
        return True

    def get_by_id(self, cid: str) -> Optional[T]:
        """
        Retrieve a record by its ID.

        Parameters
        ----------
        cid : str
            The record ID.

        Returns
        -------
        Optional[T]
            The record if found, otherwise None.
        """
        idx = self.id_to_row.get(cid)
        if idx is None:
            logger.debug(f"get_by_id: id={cid} not found")
            return None
        logger.debug(f"get_by_id: id={cid} found at idx={idx}")
        return self.records[idx]

    def count(self) -> int:
        """
        Get the number of records in the store.

        Returns
        -------
        int
            The number of stored records.
        """
        count = self.E.shape[0]
        logger.debug(f"Count called, current count={count}")
        return count
