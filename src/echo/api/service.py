import logging
from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from echo.config import load_yaml
from echo.api import auth_api, llm_api, sessions_api

api_config = load_yaml("generation/api.yaml").get("api", {})
if not api_config:
    raise ValueError("generation/api.yaml missing required 'api' section")

logging.basicConfig(level=getattr(logging, api_config.get("log_level", "INFO").upper()))
logger = logging.getLogger("echo.generation")

# Slices whose lifespans build state onto app.state, in startup order.
_SLICES = (auth_api, llm_api, sessions_api)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run every slice's lifespan; AsyncExitStack tears them down in reverse."""
    async with AsyncExitStack() as stack:
        for slice_module in _SLICES:
            await stack.enter_async_context(slice_module.lifespan(app))
        logger.info("Generation service state initialized")
        yield

app = FastAPI(title="Echo API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

for _slice in _SLICES:
    app.include_router(_slice.router)

def run():
    import uvicorn

    uvicorn.run(
        "echo.api.service:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8100),
        reload=not api_config.get("prod", False),
    )
