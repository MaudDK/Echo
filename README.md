# Echo

Echo is a self-hosted, modular **RAG (Retrieval-Augmented Generation)** system, built as a set of independently deployable FastAPI services that share one Python package (`echo`). The services communicate over HTTP, so each can be scaled, versioned, and swapped on its own.

A React chat UI sits in front of a tool-calling LLM agent that retrieves from a hybrid (dense + sparse) index — all running on your own hardware, with the LLM served by a local Ollama.

## Architecture

```
                 ┌───────────────┐
   Browser ─────▶│   Frontend    │  React + Vite (port 3000)
                 └──────┬────────┘
                        │ HTTP
                 ┌──────▼────────┐      ┌──────────────┐
                 │  Generation   │─────▶│    Ollama    │  (on the host)
                 │  (echo-llm)   │ /chat└──────────────┘
                 │   port 8100   │
                 │ auth·sessions │
                 │  ·agent·tools │
                 └──────┬────────┘
                        │ search_documents tool (HTTP)
                 ┌──────▼────────┐
                 │   Retrieval   │  FAISS dense + BM25 sparse, fused via RRF
                 │ (echo-retr.)  │
                 │   port 8090   │
                 └──────┬────────┘
                        │ embed query (HTTP)
                 ┌──────▼────────┐
                 │   Embedder    │  sentence-transformers
                 │ (echo-embed.) │
                 │   port 8080   │
                 └───────────────┘
```

| Service | Image / script | Port | Role |
|---|---|---|---|
| **Embedder** | `echo-embedder` | 8080 | Wraps `sentence-transformers`; applies per-model query/passage prefixes. |
| **Retrieval** | `echo-retrieval` | 8090 | Hybrid search — FAISS dense vectors + BM25 lexical, fused with Reciprocal Rank Fusion. Embeds queries via the embedder. |
| **Generation** | `echo-llm` | 8100 | Multi-router app: token auth, per-user chat sessions, and a tool-calling agent backed by a local Ollama model. |
| **Frontend** | — | 3000 | Streaming chat UI (login, sessions, live thinking/tool-call rendering). |

The LLM itself runs in **Ollama on the host**, not in a container — the generation service reaches it at `host.docker.internal:11434`.

## Quick start (Docker)

### Prerequisites

- Docker + Docker Compose (with the `buildx` plugin)
- [Ollama](https://ollama.com) running on the host, with a model pulled:
  ```bash
  ollama pull <your-model>      # e.g. a tool-calling capable model
  ```
- *(GPU only)* `nvidia-container-toolkit` installed on the host

### Run

```bash
cp .env.example .env            # set HF_CACHE_PATH, INDEX_PATH_DIR, ports, secret
docker compose up               # full CPU stack (embedder, retrieval, llm, frontend)
docker compose --profile gpu up # same stack with the GPU embedder
```

Only the **embedder** has CPU/GPU variants (different torch build); everything else is hardware-agnostic and reaches the embedder through the shared `embedder` network alias. `COMPOSE_PROFILES` in `.env` (default `cpu`) selects the variant for a bare `docker compose up`.

Then open the UI at **http://localhost:3000**, or hit the API directly — interactive docs are at **http://localhost:8100/docs**.

Before you can log in you must **register**, which requires the signup secret (see [Configuration](#configuration)):

```bash
curl -X POST http://localhost:8100/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"me","password":"secret","signup_secret":"<your-secret>"}'
```

## Local development (no Docker)

Each service is an editable install with its own extras:

```bash
pip install -e ".[embedder]"     # torch, sentence-transformers, numpy
pip install -e ".[retrieval]"    # faiss-cpu, pandas, httpx, rank-bm25, tqdm
pip install -e ".[generation]"   # httpx, bcrypt, ddgs
pip install -e ".[embedder,retrieval]"   # both, for local end-to-end work

echo-embedder      # serves the embedder on its api.yaml port
echo-retrieval     # requires a reachable embedder
echo-llm           # requires a reachable Ollama (and retrieval, if document_search is enabled)
```

When running outside Docker, point the `*_url` config values at `localhost` instead of the compose service names (each is documented inline in its YAML).

The frontend is a standard Vite app:

```bash
cd frontend && npm install && npm run dev
```

## Configuration

All runtime behavior is YAML-driven, resolved relative to `ECHO_CONFIGS_DIR` (defaults to `<repo>/configs`, set to `/app/configs` in the images). Loaded configs are cached per process, so changes need a restart to take effect.

| File | Owns |
|---|---|
| `configs/embedder/embedder.yaml` | Model, device, batch size, normalization |
| `configs/embedder/api.yaml` | Embedder port, max batch, prod flag, log level |
| `configs/embedder/prefixes.yaml` | Query/passage prefix conventions per model |
| `configs/vector_store/faiss.yaml` | FAISS variant + `dim` (must match the model's output dim) |
| `configs/sparse_store/bm25.yaml` | BM25 index path + parameters |
| `configs/retrieval/api.yaml` | Retrieval port, embedder URL, index path, hybrid settings |
| `configs/generation/api.yaml` | LLM-service host/port/CORS/logging |
| `configs/generation/llm.yaml` | Ollama base URL, model, timeout, retries |
| `configs/generation/agent.yaml` | Agent settings + tool toggles (`document_search`, `web_search`) |
| `configs/auth/api.yaml` | `signup_secret` (gates registration) |
| `configs/store/storage.yaml` | SQLite DB path + token TTL |
| `.env` | Machine-specific values: cache path, index dir, ports, compose profile |

**Secrets.** The registration gate reads `ECHO_SIGNUP_SECRET` from the environment if set, falling back to `configs/auth/api.yaml:signup_secret`. In production set the env var so the real secret never lands in git, and change the committed placeholder. An empty value disables registration.

**`prod: true`** in an `api.yaml` forces Hugging Face offline mode (`HF_HUB_OFFLINE=1`) and disables auto-reload — only set it once the target model is confirmed cached locally.

## Building an index

There is no indexing service — the retrieval service **loads** an index, it does not build one. Build offline with a standalone script that embeds your corpus through the embedder and calls `store.build(...)` / `store.save(path)`. The dense (FAISS) and sparse (BM25) stores must be built from the **same corpus in the same order**, since fusion keys on the shared `doc_id` (a document's position in the metadata list). Point `configs/retrieval/api.yaml:index_path` (and `configs/sparse_store/bm25.yaml:path`) at the results.

## Extending

- **New embedding model** — add it to `configs/embedder/embedder.yaml`; if it needs prefixes, add an entry to `configs/embedder/prefixes.yaml`. Update `dim` in `faiss.yaml` to match.
- **New FAISS variant** — subclass `BaseFAISSStore`, implement `_create_index()`, and register it in `indexing/vector_store/factory.py`.
- **New tool for the agent** — register a name → JSON schema + handler in the `ToolRegistry` and toggle it in `configs/generation/agent.yaml`.
- **New service** — a FastAPI app with a `lifespan`, a `run()` launcher, a `[project.scripts]` entry, and a config namespace under `configs/<svc>/`.

## License

Copyright © 2026 MaudDK. All rights reserved.

This source code is proprietary. No permission is granted to use, copy, modify, distribute, or create derivative works without the express written consent of the copyright holder. See [LICENSE](LICENSE).
