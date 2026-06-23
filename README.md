# Echo

Echo is a self-hosted, modular RAG (Retrieval-Augmented Generation) system. It currently implements a production-grade **embedding service**, with retrieval, reranking, and generation services to follow.

## Status

Early development — embedder service is functional. Retrieval, reranking, and generation layers are not yet built.

## Architecture

Echo is built as a set of independently deployable services, not a monolith. Each service is containerized separately so it can be scaled, versioned, and swapped independently.

```
[Ingestion Pipeline] ──┐
                        ├──> [Embedder Service] (this repo, currently)
[Retrieval API] ───────┘
        │
        ▼
[Vector Store] (planned)
        │
        ▼
[Reranker] → [Generation] (planned)
```

## Embedder Service

A FastAPI service wrapping `sentence-transformers`, with:

- **Config-driven model loading** — swap embedding models via YAML, no code changes
- **Query/passage prefix handling** — automatically applies the correct prefix convention for models that require it (E5, BGE, Qwen3, etc.), based on a maintained lookup table
- **CPU and GPU Docker builds** — separate Dockerfiles, CPU build uses CPU-only torch wheels to avoid bloated images
- **Offline-mode support** — can run without any Hugging Face network dependency once a model is cached
- **Health checks** — `/health` endpoint for container orchestration readiness checks

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns service status, loaded model, and device |
| `/embed` | POST | Embeds a list of texts, returns vectors |

**Example request:**

```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["hello world"], "is_query": true}'
```

## Project Structure

```
echo/
├── configs/
│   └── embedder/
│       ├── embedder.yaml      # model name, device, batch size, normalization
│       ├── api.yaml           # port, max batch size, prod flag, log level
│       └── prefixes.yaml      # known query/passage prefix conventions per model
├── src/
│   └── echo/
│       ├── api/
│       │   └── embedder_api.py
│       ├── indexing/
│       │   └── embedders/
│       │       ├── embedder.py
│       │       └── prefixes.py
│       └── config.py          # YAML config loader, shared across services
├── Dockerfile.embedder.cpu
├── Dockerfile.embedder.gpu
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

## Getting Started

### Prerequisites

- Python ≥ 3.10
- Docker + Docker Compose (with `buildx` plugin)
- (Optional, for GPU) `nvidia-container-toolkit` installed on the host

### Local setup (no Docker)

```bash
pip install -e .
cp .env.example .env   # adjust values as needed
echo-embedder
```

### Running with Docker

```bash
cp .env.example .env   # adjust HF_CACHE_PATH and ports as needed

# CPU
docker compose up embedder-cpu

# GPU (requires nvidia-container-toolkit)
docker compose up embedder-gpu
```

The container mounts your local Hugging Face cache (`HF_CACHE_PATH` in `.env`) so models aren't re-downloaded on every rebuild, and mounts `configs/` so config changes take effect on restart without rebuilding the image.

### Configuration

| File | Purpose |
|---|---|
| `configs/embedder/embedder.yaml` | Which model to load, device, batch size, normalization |
| `configs/embedder/api.yaml` | API port, max request batch size, prod/dev mode, log level |
| `configs/embedder/prefixes.yaml` | Query/passage prefix conventions for known models |
| `.env` | Local machine-specific values — Hugging Face cache path, exposed ports |

Setting `prod: true` in `api.yaml` forces offline mode (`HF_HUB_OFFLINE=1`) and disables auto-reload — only use this once your target model is confirmed cached, since offline mode has no network fallback.

## Adding a New Embedding Model

1. Add the model name to `configs/embedder/embedder.yaml`
2. If the model requires query/passage prefixes, add an entry to `configs/embedder/prefixes.yaml`:

```yaml
known_prefixes:
  "your-org/your-model":
    query: "query: "
    passage: "passage: "
```

If a model isn't listed, Echo logs a warning and falls back to no prefix — verify this is correct for your model before relying on it in production.

## Roadmap

- [ ] Vector store abstraction (FAISS, ChromaDB, Qdrant)
- [ ] Hybrid retrieval (dense + sparse/BM25)
- [ ] Reranking service
- [ ] Generation service (LLM integration)
- [ ] Evaluation harness (golden sets, retrieval metrics)
- [ ] Observability (tracing, latency metrics)

## License

TBD