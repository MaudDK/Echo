import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from echo.config import load_yaml
from echo.store.user_store import User, UserStore, UserExistsError

logger = logging.getLogger(__name__)

api_config = load_yaml("auth/api.yaml").get("api", {})
storage_config = load_yaml("store/storage.yaml").get("storage", {})

if not storage_config:
    raise ValueError("sotre/storage.yaml missing required 'llm' section")

router = APIRouter(prefix="/auth", tags=["auth"])

user_store: UserStore | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global user_store
    user_store = UserStore(
        db_path=storage_config.get("users_db", "data/generation/users.db"),
        token_ttl_seconds=storage_config.get("token_ttl_seconds", 86400),
    )
    yield

def current_user(authorization: str = Header(..., description="Bearer <token>")) -> User:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Expected 'Authorization: Bearer <token>'")
    user = user_store.resolve_token(token)
    if user is None:
        logger.warning("Rejected request with invalid or expired token")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

class Credentials(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class RegisterRequest(Credentials):
    signup_secret: str = Field(
        default="",
        description="Shared signup secret (api.yaml:auth.signup_secret) required to create an account",
    )

class LoginResponse(BaseModel):
    token: str = Field(..., description="Opaque bearer token; send as 'Authorization: Bearer <token>'")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(..., description="Token lifetime in seconds")

@router.post("/register", status_code=201)
def register(req: RegisterRequest):
    expected = api_config.get("signup_secret", "")
    if not expected:
        raise HTTPException(status_code=403, detail="Registration is disabled")

    if not secrets.compare_digest(req.signup_secret, expected):
        raise HTTPException(status_code=403, detail="Invalid signup secret")
    try:
        user = user_store.register(req.username, req.password)
    except UserExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"username": user.username}

@router.post("/login", response_model=LoginResponse)
def login(creds: Credentials):
    token = user_store.login(creds.username, creds.password)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(token=token, expires_in=user_store.token_ttl_seconds)

@router.post("/logout")
def logout(authorization: str = Header(..., description="Bearer <token>")):
    _, _, token = authorization.partition(" ")
    revoked = user_store.logout(token) if token else False
    return {"revoked": revoked}
