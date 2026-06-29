from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import APIRouter, FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from echo.config import load_yaml
from echo.api.auth_api import current_user
from echo.store.session_store import Session, SessionStore
from echo.store.user_store import User
from echo.generation.agent import Agent
from echo.generation.tool_registry import ToolRegistry
from echo.generation.tools.document_search import (
    DOCUMENT_SEARCH_SCHEMA,
    make_document_search_handler,
)
from echo.generation.tools.web_search import WEB_SEARCH_SCHEMA, make_web_search_handler


agent_config = load_yaml("generation/agent.yaml").get("agent", {})
tools_config = agent_config.get("tools", {})
llm_api_url = agent_config.get("llm_api_url", "http://localhost:8100")
_storage_config = load_yaml("store/storage.yaml").get("storage", {})

router = APIRouter(prefix="/sessions", tags=["sessions"])

session_store: SessionStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_store
    session_store = SessionStore(
        db_path=_storage_config.get("db", "data/stores.db"),
    )
    yield


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message for the agent")

class MessageResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls_made: List[Dict[str, Any]]

def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    doc_cfg = tools_config.get("document_search", {})
    if doc_cfg.get("enabled"):
        registry.register(
            "search_documents",
            DOCUMENT_SEARCH_SCHEMA,
            make_document_search_handler(doc_cfg.get("retrieval_url", "http://localhost:8090")),
        )

    web_cfg = tools_config.get("web_search", {})
    if web_cfg.get("enabled"):
        registry.register(
            "web_search",
            WEB_SEARCH_SCHEMA,
            make_web_search_handler(web_cfg.get("max_results", 5)),
        )

    return registry

def build_agent(sess: Session, token: str) -> Agent:
    return Agent(
        llm_api_url=llm_api_url,
        token=token,
        tool_registry=build_tool_registry(),
        max_steps=agent_config.get("max_steps", 5),
        temperature=agent_config.get("temperature", 0.0),
        messages=sess.messages or None,
    )

def owned_session(session_id: str, user: User) -> Session:
    sess = session_store.load(session_id, user.id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@router.post("", status_code=201)
def create_session(user: User = Depends(current_user)):
    session_id = session_store.create(user_id=user.id, agent_name="research-agent")
    return {"session_id": session_id}


@router.get("")
def list_sessions(user: User = Depends(current_user)):
    return {"sessions": session_store.list_for_user(user.id)}


@router.get("/{session_id}")
def get_session(session_id: str, user: User = Depends(current_user)):
    sess = owned_session(session_id, user)
    return {"session_id": sess.id, "name": sess.name, "messages": sess.messages}


@router.delete("/{session_id}")
def delete_session(session_id: str, user: User = Depends(current_user)):
    owned_session(session_id, user)
    session_store.delete(session_id)
    return {"deleted": session_id}

@router.post("/{session_id}/message", response_model=MessageResponse)
def send_message(
    session_id: str,
    req: MessageRequest,
    authorization: str = Header(..., description="Bearer <token>"),
    user: User = Depends(current_user),
):
    sess = owned_session(session_id, user)
    is_first = not sess.messages  # name the session from its opening message
    _, _, token = authorization.partition(" ")
    agent = build_agent(sess, token)
    try:
        result = agent.send(req.message)
        session_store.save(session_id, agent.messages)
        if is_first:
            title = req.message.strip().replace("\n", " ")[:40]
            if title:
                session_store.set_name(session_id, title)
    finally:
        agent.close()

    return MessageResponse(
        session_id=session_id,
        answer=result["answer"],
        tool_calls_made=result["tool_calls_made"],
    )


