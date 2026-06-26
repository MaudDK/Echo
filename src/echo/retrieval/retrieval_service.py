import logging
from typing import List, Dict, Any

from echo.retrieval.embedder_client import EmbedderClient
from echo.indexing.vector_store.base_store import BaseVectorStore

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, embedder_client: EmbedderClient, vector_store: BaseVectorStore):
        self.embedder_client = embedder_client
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        
        query_embedding = self.embedder_client.embed([query], is_query=True)
        results = self.vector_store.search(query_embedding, top_k)
        logger.debug(f"Query '{query[:50]}...' returned {len(results)} results")
        return results

    def ensemble_retrieve(self, query: str, top_k: int, embedder_clients: List[EmbedderClient]) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embeddings = [client.embed([query], is_query=True) for client in embedder_clients]
        results = [self.vector_store.search(embedding, top_k) for embedding in query_embeddings]
        logger.debug(f"Ensemble query '{query[:50]}...' returned {len(results)} results")
        return results
