from abc import abstractmethod
import logging
import os
import pickle
from typing import List, Optional, Tuple, Dict, Any
import faiss
import numpy as np

from .base_store import BaseVectorStore

logger = logging.getLogger(__name__)

class BaseFAISSStore(BaseVectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = self._create_index()
        self.meta: List[Dict[str, Any]] = []
        logger.info(f"Initialized {self.__class__.__name__} — dim:{dim}")
    
    @abstractmethod
    def _create_index(self) -> faiss.Index:
        """Create the FAISS index object for FAISS Variant"""
        pass

    @property
    def count(self) -> int:
        return self.index.ntotal

    def _validate_query(self, embedding: np.ndarray, metadata: List[Dict[str, Any]]):
        if embedding.shape[0] != len(metadata):
            raise ValueError(f"Number of query embeddings ({embedding.shape[0]}) does not match metadata count({len(metadata)})")

        if embedding.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {embedding.shape[1]} doesn't match index dim {self.dim}")

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        self._validate_query(embeddings, metadata)
        embeddings.astype(np.float32, copy=False)

        self.meta = metadata
        self.index.add(embeddings)
        logger.info(f"Built index with ({self.count} vectors)")

    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        self._validate_query(embedding, [metadata])
        embedding = embedding.astype(np.float32, copy=False)

        self.index.add(embedding)
        self.meta.extend(metadata)
        logger.info(f"Added {embedding.shape[0]} vectors. Total ({self.count} vectors)")

    def search_ids(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if self.count == 0:
            logger.warning("search() called on an empty index — returning no results")
            return []

        query_embedding = np.atleast_2d(query_embedding).astype(np.float32, copy=False)
        scores, idx = self.index.search(query_embedding, k)
        return [(int(i), float(scores[0][j])) for j, i in enumerate(idx[0]) if i != -1]

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        return [(self.meta[i], score) for i, score in self.search_ids(query_embedding, k)]

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname: os.makedirs(dirname, exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")

        with open(f"{path}.meta", "wb") as f:
            pickle.dump(self.meta, f)

        logger.info(f"Saved index ({self.count} vectors) to {path}.index / {path}.meta")

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            self.meta = pickle.load(f)
        logger.info(f"Loaded index ({self.count} vectors) from {path}.index / {path}.meta")
