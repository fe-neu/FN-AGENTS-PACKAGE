from __future__ import annotations
from typing import List, Tuple, Optional
import uuid

from fn_package.utils.logger import get_logger
from fn_package.config import DEFAULT_RAG_TOP_K, DEFAULT_RAG_THRESHOLD

from ..core.vector_store import VectorStore
from ..core.retriever import Retriever, Hit
from .chunker import Chunker
from ..core.embedder import OpenAIEmbedder
from .parser import PDFParser
from .chunk_record import ChunkRecord

logger = get_logger(__name__)


class RagService:
    """
    Retrieval-Augmented Generation (RAG) service facade without direct LLM integration.

    Provides functionality for:
    - Ingest: parsing PDFs or plain text, chunking, embedding, and storing.
    - Retrieval: similarity search (top-k or threshold).
    - Context building: expanding neighboring chunks, applying character budgets, and
      assembling context strings.
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[OpenAIEmbedder] = None,
        chunker: Optional[Chunker] = None,
        retriever: Optional[Retriever] = None,
        pdf_parser: Optional[PDFParser] = None,
    ):
        """
        Initialize the RAG service with its main components.

        Parameters
        ----------
        store : VectorStore, optional
            Vector store for embeddings (defaults to new VectorStore).
        embedder : OpenAIEmbedder, optional
            Embedding generator (defaults to OpenAIEmbedder).
        chunker : Chunker, optional
            Splits text into token-based chunks (defaults to Chunker).
        retriever : Retriever, optional
            Similarity-based retriever (defaults to Retriever).
        pdf_parser : PDFParser, optional
            Parser for extracting text from PDFs (defaults to PDFParser).
        """
        self.store = store or VectorStore(dim=1536)
        self.embedder = embedder or OpenAIEmbedder()
        self.chunker = chunker or Chunker()
        self.retriever = retriever or Retriever(self.store)
        self.pdf_parser = pdf_parser or PDFParser()
        logger.info(
            "RagService initialized (store_dim=%s, embed_model=%s)",
            self.store.dim,
            getattr(self.embedder, "model", "unknown"),
        )

    # ---------- Ingest ----------
    def ingest_pdf(self, path: str) -> List[str]:
        """
        Ingest a PDF document into the vector store.

        Steps:
        - Parse the PDF into text.
        - Split into chunks.
        - Embed each chunk.
        - Store embeddings in the vector store.

        Parameters
        ----------
        path : str
            Path to the PDF file.

        Returns
        -------
        List[str]
            List of new chunk IDs.
        """
        self.ingest_text(self.pdf_parser.parse(path), source_path=path)

    def ingest_text(self, text: str, source_path: str = "inline") -> List[str]:
        """
        Ingest plain text into the vector store.

        Steps:
        - Split text into chunks.
        - Embed each chunk.
        - Store as ChunkRecord objects with chaining to neighbors.

        Parameters
        ----------
        text : str
            The input text.
        source_path : str, optional
            Origin of the text (e.g., file path or "inline").

        Returns
        -------
        List[str]
            List of new chunk IDs in chunk order.

        Raises
        ------
        ValueError
            If embedding dimension does not match the vector store.
        """
        logger.info(
            "Ingesting inline text (len=%s) from source_path='%s'",
            len(text),
            source_path,
        )

        if self.embedder.dim != self.store.dim:
            msg = (
                f"Embedding dim ({self.embedder.dim}) != store dim ({self.store.dim}). "
                "Re-initialize VectorStore with the embedder's dim or adjust config."
            )
            logger.error(msg)
            raise ValueError(msg)

        # 1) Chunking
        chunks = self.chunker.split(text)
        if not chunks:
            logger.warning("No chunks produced from input text.")
            return []

        # 2) Generate IDs and link chunks
        ids: List[str] = [uuid.uuid4().hex for _ in range(len(chunks))]
        for i, chunk_text in enumerate(chunks):
            prev_id = ids[i - 1] if i > 0 else None
            next_id = ids[i + 1] if i < len(chunks) - 1 else None

            # 3) Embed the chunk
            emb = self.embedder.embed(chunk_text)
            if emb.shape != (self.store.dim,):
                msg = f"Got embedding shape {emb.shape}, expected ({self.store.dim},)"
                logger.error(msg)
                raise ValueError(msg)

            # 4) Create and store record
            rec = ChunkRecord(
                id=ids[i],
                source_path=source_path,
                text=chunk_text,
                embedding=emb,
                prev_id=prev_id,
                next_id=next_id,
            )
            self.store.add(rec)
            logger.debug("Ingested chunk %s/%s id=%s", i + 1, len(chunks), ids[i])

        logger.info("Ingested %s chunks from source_path='%s'", len(ids), source_path)
        return ids

    # ---------- Retrieval ----------
    def topk(self, query: str, k: int = DEFAULT_RAG_TOP_K) -> List[Hit]:
        """
        Retrieve the top-k most similar chunks for a given query.

        Parameters
        ----------
        query : str
            The query text.
        k : int, optional
            Number of results to return (default from config).

        Returns
        -------
        List[Hit]
            List of retrieval hits with records and similarity scores.
        """
        logger.info("topk: query_len=%s, k=%s", len(query), k)
        q_vec = self.embedder.embed(query)
        hits = self.retriever.topk_by_embedding(q_vec, k=k)
        logger.debug("topk: got %s hits", len(hits))
        return hits

    def by_threshold(self, query: str, min_score: float = DEFAULT_RAG_THRESHOLD) -> List[Hit]:
        """
        Retrieve all chunks with a similarity score above a threshold.

        Parameters
        ----------
        query : str
            The query text.
        min_score : float, optional
            Minimum similarity score (default from config).

        Returns
        -------
        List[Hit]
            List of retrieval hits meeting the threshold.
        """
        logger.info("by_threshold: query_len=%s, min_score=%s", len(query), min_score)
        q_vec = self.embedder.embed(query)
        hits = self.retriever.all_above_threshold(q_vec, min_score=min_score)
        logger.debug("by_threshold: got %s hits", len(hits))
        return hits

    # ---------- Context building ----------
    def expand_neighbors(self, ids: List[str], window: int = 1) -> List[str]:
        """
        Expand a set of chunk IDs by including neighboring chunks.

        For each input ID, include up to `window` previous and next chunks.
        Duplicates are removed and order is preserved.

        Parameters
        ----------
        ids : List[str]
            Seed chunk IDs.
        window : int, optional
            Number of neighbors to include before and after each seed (default = 1).

        Returns
        -------
        List[str]
            Expanded list of chunk IDs.
        """
        logger.info("expand_neighbors: seeds=%s, window=%s", len(ids), window)
        if window <= 0:
            return ids

        out: List[str] = []
        seen: set[str] = set()

        def push(cid: Optional[str]) -> None:
            if cid and cid not in seen and self.store.get_by_id(cid) is not None:
                seen.add(cid)
                out.append(cid)

        for seed in ids:
            # Traverse left neighbors
            cur = self.store.get_by_id(seed)
            left_ids: List[str] = []
            steps = window
            while cur is not None and steps > 0 and cur.prev_id:
                cur = self.store.get_by_id(cur.prev_id)
                if cur is None:
                    break
                left_ids.append(cur.id)
                steps -= 1
            for cid in reversed(left_ids):
                push(cid)

            # Add seed
            push(seed)

            # Traverse right neighbors
            cur = self.store.get_by_id(seed)
            steps = window
            while cur is not None and steps > 0 and cur.next_id:
                cur = self.store.get_by_id(cur.next_id)
                if cur is None:
                    break
                push(cur.id)
                steps -= 1

        logger.info("expand_neighbors: produced %s ids (from %s seeds)", len(out), len(ids))
        return out

    # ---------- One-Stop-Shop Context ----------
    def context_from_hits(
        self,
        hits: List[Hit],
        neighbor_window: int = 1,
        max_chars: Optional[int] = None,
    ) -> Tuple[List[ChunkRecord], str]:
        """
        Build a context string from retrieval hits.

        Steps:
        - Expand neighbors of the hit records.
        - Deduplicate and order records.
        - Concatenate texts into a context string.

        Parameters
        ----------
        hits : List[Hit]
            Retrieval hits.
        neighbor_window : int, optional
            Number of neighbors to include per hit (default = 1).
        max_chars : int, optional
            Maximum character length of the final context string.

        Returns
        -------
        Tuple[List[ChunkRecord], str]
            - List of retrieved and expanded records in order.
            - Concatenated context string.
        """
        logger.info(
            "context_from_hits: hits=%s, neighbor_window=%s, max_chars=%s",
            len(hits),
            neighbor_window,
            max_chars,
        )

        seed_ids = [h.record.id for h in hits]
        expanded_ids = self.expand_neighbors(seed_ids, window=neighbor_window)

        records: List[ChunkRecord] = []
        for cid in expanded_ids:
            rec = self.store.get_by_id(cid)
            if rec is not None:
                records.append(rec)

        ctx = "\n\n".join(r.text for r in records)
        if max_chars is not None and len(ctx) > max_chars:
            ctx = ctx[:max_chars]

        logger.debug(
            "context_from_hits: context_len=%s, records_returned=%s",
            len(ctx),
            len(records),
        )
        return records, ctx

    def build_context(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        neighbor_window: int = 1,
        max_chars: Optional[int] = None,
    ) -> Tuple[List[ChunkRecord], str]:
        """
        One-stop method for retrieving and building context.

        Steps:
        - Embed the query.
        - Perform retrieval (top-k or threshold-based).
        - Expand neighbors.
        - Concatenate texts into a context string.

        Parameters
        ----------
        query : str
            The query text.
        k : int, optional
            Number of top results to return. Mutually exclusive with threshold.
        threshold : float, optional
            Minimum similarity score for results. Mutually exclusive with k.
        neighbor_window : int, optional
            Number of neighbors to include for each retrieved chunk (default = 1).
        max_chars : int, optional
            Maximum length of the context string.

        Returns
        -------
        Tuple[List[ChunkRecord], str]
            - List of records included in context.
            - Final context string.
        """
        logger.info(
            "build_context: query_len=%s, k=%s, threshold=%s, neighbor_window=%s, max_chars=%s",
            len(query),
            k,
            threshold,
            neighbor_window,
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
        records, ctx = self.context_from_hits(
            hits, neighbor_window=neighbor_window, max_chars=max_chars
        )
        logger.info(
            "build_context: final_records=%s, context_len=%s",
            len(records),
            len(ctx),
        )
        return records, ctx

    # ---------- Administration ----------
    def count(self) -> int:
        """
        Get the number of chunks currently stored.

        Returns
        -------
        int
            Number of records in the store.
        """
        return self.store.count()
