import logging
from typing import List, Dict, Any, Optional

from echo.retrieval.embedder_client import EmbedderClient
from echo.retrieval.fusion import reciprocal_rank_fusion
from echo.indexing.vector_store.base_store import BaseVectorStore
from echo.indexing.sparse_store.bm25_store import BM25Store

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(
        self,
        embedder_client: EmbedderClient,
        vector_store: BaseVectorStore,
        sparse_store: Optional[BM25Store] = None,
    ):
        self.embedder_client = embedder_client
        self.vector_store = vector_store
        self.sparse_store = sparse_store

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedder_client.embed([query], is_query=True)
        results = self.vector_store.search(query_embedding, top_k)
        logger.debug(f"Query '{query[:50]}...' returned {len(results)} results")
        return results

    def hybrid_retrieve(self, query: str, top_k: int, candidate_k: int = 50, rrf_k: int = 60) -> List[Dict[str, Any]]:
        if self.sparse_store is None:
            logger.warning("hybrid_retrieve called without a sparse store — falling back to dense")
            return self.retrieve(query, top_k)
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedder_client.embed([query], is_query=True)
        dense_hits = self.vector_store.search_ids(query_embedding, candidate_k)
        sparse_hits = self.sparse_store.search(query, candidate_k)

        fused = reciprocal_rank_fusion(
            [[doc_id for doc_id, _ in dense_hits], [doc_id for doc_id, _ in sparse_hits]],
            k=rrf_k,
        )

        results = [(self.vector_store.meta[doc_id], score) for doc_id, score in fused[:top_k]]
        logger.debug(
            f"Hybrid query '{query[:50]}...' fused {len(dense_hits)} dense + "
            f"{len(sparse_hits)} sparse into {len(results)} results"
        )
        return results

    def ensemble_retrieve(self, query: str, top_k: int, embedder_clients: List[EmbedderClient]) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embeddings = [client.embed([query], is_query=True) for client in embedder_clients]
        results = [self.vector_store.search(embedding, top_k) for embedding in query_embeddings]
        logger.debug(f"Ensemble query '{query[:50]}...' returned {len(results)} results")
        return results
