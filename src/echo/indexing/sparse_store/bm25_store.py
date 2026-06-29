import logging
import os
import pickle
import re
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25Store:
    def __init__(self, path: Optional[str] = None, k1: float = 1.5, b: float = 0.75):
        self.path = path
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[int] = []
        self._count = 0

        if path and os.path.exists(path):
            self.load(path)

    def build(self, documents: List[str], doc_ids: Optional[List[int]] = None):
        if doc_ids is None:
            doc_ids = list(range(len(documents)))
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids count ({len(doc_ids)}) does not match documents count ({len(documents)})"
            )

        tokenized = [_tokenize(doc or "") for doc in documents]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self.doc_ids = list(doc_ids)
        self._count = len(documents)
        logger.info(f"Built BM25 index with {self._count} docs")

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        if self.bm25 is None or self._count == 0:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        # Top-k by score, dropping zero-score docs (no term overlap).
        top = np.argsort(scores)[::-1][:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Optional[str] = None):
        path = path or self.path
        if not path:
            raise ValueError("BM25Store.save requires a path")
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids, "count": self._count}, f)
        logger.info(f"Saved BM25 index ({self._count} docs) to {path}")

    def load(self, path: Optional[str] = None):
        path = path or self.path
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.bm25 = state["bm25"]
        self.doc_ids = state["doc_ids"]
        self._count = state["count"]
        logger.info(f"Loaded BM25 index ({self._count} docs) from {path}")

    @property
    def count(self) -> int:
        return self._count
