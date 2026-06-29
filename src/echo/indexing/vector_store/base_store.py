import numpy as np
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    @abstractmethod
    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        pass

    @abstractmethod
    def search_ids(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Like ``search`` but returns ``(doc_id, score)`` — the doc_id is the
        position in ``meta``, used as the fusion key for hybrid retrieval."""
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        pass