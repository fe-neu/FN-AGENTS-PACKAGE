from ..core.base_record import BaseRecord
from dataclasses import dataclass, field
import time

@dataclass
class MemoryRecord(BaseRecord):
    # Erbt id, text, embedding von BaseRecord
    created_at: float = field(default_factory=lambda: time.time())
    importance: float = 0.5