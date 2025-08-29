from ..core.base_record import BaseRecord
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChunkRecord(BaseRecord):
    # Erbt id, text, embedding von BaseRecord
    source_path: str
    prev_id: Optional[str] = None
    next_id: Optional[str] = None