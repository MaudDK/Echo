import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from echo.config import load_yaml
from echo.retrieval.embedder_client import EmbedderClient
from echo.retrieval.retrieval_service import RetrievalService
from echo.indexing.vector_store.factory import get_faiss_store
from echo.indexing.sparse_store.bm25_store import BM25Store

#Config Setup
api_config = load_yaml("retrieval/api.yaml").get("api", {})
vector_store_config = load_yaml("vector_store/faiss.yaml").get("vector_store", {})

#Logging
log_level = api_config.get('log_level', "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)


if not api_config:
    raise ValueError("retrieval/api.yaml missing required 'api' section")
if not vector_store_config:
    raise ValueError("vector_store/faiss.yaml missing required 'vector_store' section")

EMBEDDER_URL = api_config.get("embedder_url")
if not EMBEDDER_URL:
    raise ValueError("retrieval/api.yaml missing required field: api.embedder_url")

INDEX_PATH = api_config.get("index_path")

HYBRID_CONFIG = api_config.get("hybrid", {})
HYBRID_ENABLED = HYBRID_CONFIG.get("enabled", False)
CANDIDATE_K = HYBRID_CONFIG.get("candidate_k", 50)
RRF_K = HYBRID_CONFIG.get("rrf_k", 60)

retrieval_service: RetrievalService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_service

    embedder_client = EmbedderClient(base_url=EMBEDDER_URL)
    if not embedder_client.check_health():
        raise RuntimeError(f"Embedder service at {EMBEDDER_URL} is not healthy — cannot start retrieval service")

    vector_store = get_faiss_store(vector_store_config)
    if INDEX_PATH:
        try:
            vector_store.load(INDEX_PATH)
            logger.info(f"Loaded existing index from {INDEX_PATH} ({vector_store.count} vectors)")
        except FileNotFoundError:
            logger.warning(f"Index file not found at {INDEX_PATH}. Starting with an empty index.")

    sparse_store = None
    if HYBRID_ENABLED:
        sparse_store_config = load_yaml("sparse_store/bm25.yaml").get("sparse_store", {})
        if not sparse_store_config:
            raise ValueError("hybrid enabled but sparse_store/bm25.yaml missing 'sparse_store' section")
        sparse_store = BM25Store(**sparse_store_config)
        logger.info(f"Hybrid retrieval enabled — loaded sparse store ({sparse_store.count} docs)")

    retrieval_service = RetrievalService(embedder_client, vector_store, sparse_store)
    yield
    logger.info("Shutting down retrieval service")
    embedder_client.close()


app = FastAPI(title="Echo Retrieval API", version="1.0", lifespan=lifespan)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    hybrid: bool | None = Field(
        default=None,
        description="Override dense/hybrid mode. None uses the server's configured default.",
    )


class SearchResult(BaseModel):
    metadata: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int


def _run_search(query: str, k: int, hybrid: bool | None) -> SearchResponse:
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialized")

    use_hybrid = HYBRID_ENABLED if hybrid is None else hybrid
    if use_hybrid and retrieval_service.sparse_store is None:
        raise HTTPException(status_code=400, detail="Hybrid search requested but no sparse store is configured")

    try:
        if use_hybrid:
            results = retrieval_service.hybrid_retrieve(query, k, candidate_k=CANDIDATE_K, rrf_k=RRF_K)
        else:
            results = retrieval_service.retrieve(query, k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        results=[SearchResult(metadata=meta, score=score) for meta, score in results],
        count=len(results),
    )

@app.get("/health")
def health():
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialized")
    return {
        "status": "ok",
        "index_count": retrieval_service.vector_store.count,
        "embedder_healthy": retrieval_service.embedder_client.check_health(),
        "hybrid": retrieval_service.sparse_store is not None,
        "sparse_count": retrieval_service.sparse_store.count if retrieval_service.sparse_store else 0,
    }

@app.get("/search", response_model=SearchResponse)
def search_get(
    query: str = Query(..., min_length=1, description="The search query"),
    k: int = Query(default=5, ge=1, le=100, description="Number of results to return"),
    hybrid: bool | None = Query(default=None, description="Override the server's dense/hybrid default"),
):
    return _run_search(query, k, hybrid)

@app.post("/search", response_model=SearchResponse)
def search_post(request: SearchRequest):
    return _run_search(request.query, request.k, request.hybrid)

def run():
    import uvicorn
    uvicorn.run(
        "echo.api.retrieval_api:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8090),
        reload=not api_config.get("prod", False)
    )

