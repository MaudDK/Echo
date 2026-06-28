import os
import logging
from contextlib import asynccontextmanager
from typing import List
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from echo.config import load_yaml

# Config Setup
api_config = load_yaml("embedder/api.yaml").get('api', {})
embedder_config = load_yaml("embedder/embedder.yaml").get('embedder', {})

# Logging
log_level = api_config.get('log_level', "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)

if api_config.get("prod", False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from echo.indexing.embedders.embedder import Embedder

embedder: Embedder | None = None

if not embedder_config:
    raise ValueError("embedder.yaml missing required 'embedder' section")
if not api_config:
    raise ValueError("api.yaml missing required 'api' section")

MODEL_NAME = embedder_config.get("model_name")
if not MODEL_NAME:
    raise ValueError("embedder.yaml missing required field: embedder.model_name")

MAX_BATCH_SIZE = api_config.get("max_batch_size")
if MAX_BATCH_SIZE is None:
    raise ValueError("api.yaml missing required field: api.max_batch_size")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    embedder = Embedder(
        model_name=MODEL_NAME,
        device=embedder_config.get("device"),
        batch_size=embedder_config.get("default_batch_size", 32),
        normalize=embedder_config.get("normalize", True),
    )
    yield
    logger.info("Shutting down embedding service")

app = FastAPI(title="Echo Embedder API", version="1.0", lifespan=lifespan)

class EmbedRequest(BaseModel):
    inputs: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of texts to embed. Capped at MAX_BATCH_SIZE per request to bound memory/latency."
    )
    is_query: bool = Field(
        default=False,
        description="True applies the model's query prefix; False applies the passage prefix."
    )
    normalize: bool = embedder_config.get("normalize", True)

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    latency: float

@app.get("/health")
def health():
    if embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL_NAME, "device": embedder.device}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        start = time.perf_counter()
        embeddings = embedder.encode(req.inputs, is_query=req.is_query, normalize=req.normalize)
        latency = time.perf_counter() - start
        logger.info(f"Encoded {len(req.inputs)} texts in {latency:.2f}ms")

    except Exception as e:
        logger.exception("Embedding inference failed")
        raise HTTPException(status_code=500, detail=str(e))

    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimension=embedder.dim,
        model=MODEL_NAME,
        latency=latency
    )

def run():
    import uvicorn
    uvicorn.run(
        "echo.api.embedder_api:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8080),
        reload=not api_config.get("prod", False)
    )

