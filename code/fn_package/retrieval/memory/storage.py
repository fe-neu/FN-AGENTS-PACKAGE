from __future__ import annotations
import os, csv, time
from typing import List
import numpy as np

from fn_package.utils.logger import get_logger
from fn_package.config import MEMORY_CSV_PATH
from .memory_record import MemoryRecord

logger = get_logger(__name__)


class MemoryStorage:
    """
    Persistent storage for memory records in a CSV file.

    Features
    --------
    - Loads all stored memories as a list of `MemoryRecord` objects.
    - Append-only: new records are added to the CSV, existing ones are not modified.
    - Embeddings are stored as space-separated float strings.
    """

    def __init__(self, path: str = MEMORY_CSV_PATH):
        """
        Initialize the memory storage.

        Parameters
        ----------
        path : str, optional
            Path to the CSV file where memories are stored (default from config).
        """
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        logger.info("MemoryStorage initialized with path=%s", self.path)

    def load_all(self) -> List[MemoryRecord]:
        """
        Load all memories from the CSV file.

        Returns
        -------
        List[MemoryRecord]
            A list of loaded memory records. Returns an empty list if no file exists.
        """
        if not os.path.exists(self.path):
            logger.warning("Memory CSV not found at %s → returning empty list", self.path)
            return []

        records: List[MemoryRecord] = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Parse embedding from space-separated float string
                    emb = np.fromstring(row["embedding"], sep=" ", dtype=np.float32)
                    rec = MemoryRecord(
                        id=row["id"],
                        text=row["text"],
                        embedding=emb,
                        created_at=float(row["created_at"]),
                        importance=float(row.get("importance", 0.5)),
                    )
                    logger.debug(f"New Record found {rec}")
                    records.append(rec)
                except Exception as e:
                    logger.error("Error parsing row in %s: %s", self.path, e)
        logger.info("Loaded %s memories from %s", len(records), self.path)
        return records

    def append(self, record: MemoryRecord) -> None:
        """
        Append a new memory record to the CSV file.

        Parameters
        ----------
        record : MemoryRecord
            The memory record to store.
        """
        file_exists = os.path.exists(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "text", "embedding", "created_at", "importance"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)

            # Write header if file does not exist yet
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "id": record.id,
                "text": record.text,
                "embedding": " ".join(f"{x:.6f}" for x in record.embedding.tolist()),
                "created_at": record.created_at,
                "importance": record.importance,
            })
        logger.info("Appended memory id=%s to %s", record.id, self.path)
