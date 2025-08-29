from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import Dict

@dataclass
class Thought:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    content: str = ""

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),  # Convert datetime to string
            'content': self.content
        }