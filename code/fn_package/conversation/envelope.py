from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass
class Envelope:
    sender: str # Agent Name or User
    recipient: str # Agent Name or User
    timestamp: datetime
    message: str

    def to_dict(self) -> Dict:
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'timestamp': self.timestamp.isoformat(),  # Convert datetime to string
            'message': self.message
        }