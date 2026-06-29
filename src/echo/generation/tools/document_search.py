import logging

import httpx

logger = logging.getLogger(__name__)

DOCUMENT_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the private knowledge base for relevant documents. Use this for "
            "questions about the user's own indexed corpus. You can call it multiple "
            "times with different queries if one search isn't enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {"type": "integer", "description": "Number of results to retrieve (default 5)"},
            },
            "required": ["query"],
        },
    },
}


def make_document_search_handler(retrieval_url: str, timeout: float = 30.0):
    """Build a handler that queries the retrieval service over HTTP.

    Generation talks to retrieval over HTTP (not in-process) — same pattern as
    retrieval -> embedder — so this keeps FAISS/torch out of this service.
    """
    base_url = retrieval_url.rstrip("/")

    def handler(query: str, k: int = 5) -> str:
        try:
            resp = httpx.post(f"{base_url}/search", json={"query": query, "k": k}, timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Document search failed: {e}")
            return f"Error: document search is unavailable ({e})."

        results = resp.json().get("results", [])
        if not results:
            return "No results found in the knowledge base."
        return "\n".join(
            f"[{i+1}] {r['metadata'].get('text', r['metadata'])} (score: {r['score']:.3f})"
            for i, r in enumerate(results)
        )

    return handler
