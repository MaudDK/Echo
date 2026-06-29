import logging
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import APIRouter, FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from echo.config import load_yaml
from echo.api.auth_api import current_user
from echo.store.user_store import User
from echo.generation.llm_client import OllamaClient

logger = logging.getLogger(__name__)

llm_config = load_yaml("generation/llm.yaml").get("llm", {})
if not llm_config:
    raise ValueError("generation/llm.yaml missing required 'llm' section")

router = APIRouter(tags=["llm"])

llm_client: OllamaClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    llm_client = OllamaClient(
        base_url=llm_config.get("base_url"),
        model=llm_config.get("model"),
        timeout=llm_config.get("timeout", 120.0),
        max_retries=llm_config.get("max_retries", 3),
    )
    if not llm_client.check_health():
        raise RuntimeError(f"LLM service at {llm_config.get('base_url')} is not healthy — cannot start generation service")
    yield
    llm_client.close()

class GenerationRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of messages for the LLM")
    tools: List[Dict[str, Any]] = Field(default=None, description="Optional list of tools for the LLM")
    temperature: float = Field(default=0.7, description="Temperature for response generation")


class GenerationResponse(BaseModel):
    response: Dict[str, Any] = Field(..., description="Response from the LLM")
    info: Dict[str, Any] = Field(..., description="Information about the generation")
    tools: List[Dict[str, Any]] = Field(default=None, description="List of tools used in the generation")
    latency: float = Field(..., description="Latency of the generation request")

@router.get("/health")
def health():
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    if not llm_client.check_health():
        raise HTTPException(status_code=503, detail="LLM service is not healthy")
    return {"status": "ok", "model": llm_client.model}

@router.post("/chat", response_model=GenerationResponse)
def chat_endpoint(request: GenerationRequest, user: User = Depends(current_user)) -> GenerationResponse:
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    start = time.perf_counter()
    response = llm_client.chat(
        messages=request.messages,
        tools=request.tools,
        temperature=request.temperature,
    )
    latency = (time.perf_counter() - start) * 1000

    return GenerationResponse(
        response=response.get("message", {}),
        info={"model": llm_client.model, "temperature": request.temperature},
        tools=response.get("tools", []),
        latency=latency,
    )
