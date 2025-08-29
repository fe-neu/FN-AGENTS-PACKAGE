# base_record.py
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import numpy as np

@dataclass
class BaseRecord(ABC):
    id: str
    text: str
    embedding: np.ndarray