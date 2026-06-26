import faiss
from echo.indexing.vector_store.faiss_store import BaseFAISSStore

class FaissFlatStore(BaseFAISSStore):
    """Exact Search (Brute Force) - best for small datasets or as a baseline"""
    def _create_index(self) -> faiss.Index:
        return faiss.IndexFlatL2(self.dim)
    

class FaissHNSWStore(BaseFAISSStore):
    """Graph Based Approximate Nearest Neighbors - best default for medium to large datasets"""
    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        super().__init__(dim)

    def _create_index(self) -> faiss.Index:
        index = faiss.IndexHNSWFlat(self.dim, self.m)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index
