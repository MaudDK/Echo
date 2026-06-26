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
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        pass