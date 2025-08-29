from __future__ import annotations
from typing import List, Tuple, Optional
import uuid

from fn_package.utils.logger import get_logger
from fn_package.config import DEFAULT_MEMORY_TOP_K, DEFAULT_MEMORY_THRESHOLD

from ..core.vector_store import VectorStore
from ..core.retriever import Retriever, Hit
from ..core.embedder import OpenAIEmbedder
from .memory_retriever import MemoryRetriever
from .memory_record import MemoryRecord
from .storage import MemoryStorage

logger = get_logger(__name__)


class MemoryService:
    """
    High-level service for managing memory records.

    Combines embedding, retrieval, storage, and context-building
    into a single interface.
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[OpenAIEmbedder] = None,
        retriever: Optional[Retriever] = None,
        storage: Optional[MemoryStorage] = None,
    ):
        """
        Initialize the MemoryService.

        Parameters
        ----------
        store : VectorStore, optional
            Vector store for memory records (defaults to new VectorStore).
        embedder : OpenAIEmbedder, optional
            Embedder to convert text into embeddings (defaults to OpenAIEmbedder).
        retriever : Retriever, optional
            Retriever for similarity search (defaults to Retriever).
        storage : MemoryStorage, optional
            Persistent storage for memory records (defaults to MemoryStorage).
        """
        self.store = store or VectorStore(dim=1536)
        self.embedder = embedder or OpenAIEmbedder()
        self.retriever = retriever or Retriever(self.store)
        self.storage = storage or MemoryStorage()
        logger.info(
            "MemoryService initialized (store_dim=%s, embed_model=%s)",
            self.store.dim,
            getattr(self.embedder, "model", "unknown"),
        )

        # Load existing memories from storage into the store
        memories = self.storage.load_all()
        if memories:
            for memory in memories:
                self.store.add(memory)
            logger.info("Loaded %d memories from storage.", len(memories))

    def add(self, text: str, importance: float = 0.5) -> str:
        """
        Add a new memory.

        Steps:
        - Embed the text.
        - Create a MemoryRecord.
        - Store it in memory and persist to disk.

        Parameters
        ----------
        text : str
            The text content of the memory.
        importance : float, optional
            Importance score in [0,1], default = 0.5.

        Returns
        -------
        str
            The ID of the newly added memory.
        """
        embedding = self.embedder.embed(text)
        memory = MemoryRecord(
            id=str(uuid.uuid4()),
            text=text,
            embedding=embedding,
            importance=importance
        )
        self.store.add(memory)
        self.storage.append(memory)
        logger.info("Added new memory with id=%s", memory.id)
        return memory.id

    def topk(self, query: str, k: int = DEFAULT_MEMORY_TOP_K) -> List[Hit[MemoryRecord]]:
        """
        Retrieve the top-k most similar memories to a query.

        Parameters
        ----------
        query : str
            Query text.
        k : int, optional
            Number of results to return (default from config).

        Returns
        -------
        List[Hit[MemoryRecord]]
            List of hits containing memory records and similarity scores.
        """
        q_vec = self.embedder.embed(query)
        hits = self.retriever.topk_by_embedding(q_vec, k=k)
        logger.info("topk: query='%s', k=%d, found=%d", query, k, len(hits))
        return hits

    def by_threshold(self, query: str, min_score: float = DEFAULT_MEMORY_THRESHOLD) -> List[Hit[MemoryRecord]]:
        """
        Retrieve all memories with scores above a threshold.

        Parameters
        ----------
        query : str
            Query text.
        min_score : float, optional
            Minimum similarity score (default from config).

        Returns
        -------
        List[Hit[MemoryRecord]]
            List of hits containing memory records and similarity scores.
        """
        q_vec = self.embedder.embed(query)
        hits = self.retriever.all_above_threshold(q_vec, min_score=min_score)
        logger.info("by_threshold: query='%s', min_score=%.2f, found=%d", query, min_score, len(hits))
        return hits

    def build_context(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        max_chars: Optional[int] = None,
    ) -> Tuple[List[MemoryRecord], str]:
        """
        Build a context string from relevant memories.

        Steps:
        - Embed the query.
        - Retrieve memories (via top-k or threshold search).
        - Concatenate memory texts into a context string.

        Parameters
        ----------
        query : str
            The query text.
        k : int, optional
            Number of top results to return. Mutually exclusive with threshold.
        threshold : float, optional
            Minimum similarity score for results. Mutually exclusive with k.
        max_chars : int, optional
            Maximum character length for the context string.

        Returns
        -------
        Tuple[List[MemoryRecord], str]
            - The list of memory records in order.
            - The concatenated context string.
        """
        logger.info(
            "build_context: query_len=%s, k=%s, threshold=%s, max_chars=%s",
            len(query),
            k,
            threshold,
            max_chars,
        )
        if k is not None and threshold is not None:
            raise ValueError("Specify either k or threshold, not both.")

        if k is not None:
            hits = self.topk(query, k=k)
        elif threshold is not None:
            hits = self.by_threshold(query, min_score=threshold)
        else:
            raise ValueError("Either k or threshold must be specified.")

        logger.debug("build_context: got %s hits", len(hits))

        records: List[MemoryRecord] = [h.record for h in hits]

        ctx = "\n\n".join(r.text for r in records)
        if max_chars is not None and len(ctx) > max_chars:
            ctx = ctx[:max_chars]

        logger.info(
            "build_context: final_records=%s, context_len=%s",
            len(records),
            len(ctx),
        )
        return records, ctx

    def count(self) -> int:
        """
        Return the total number of stored memories.

        Returns
        -------
        int
            Count of memory records in the store.
        """
        return self.store.count()
