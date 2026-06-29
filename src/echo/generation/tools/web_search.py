import logging

from ddgs import DDGS

logger = logging.getLogger(__name__)

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web for current or general information not in the private "
            "knowledge base — recent events, facts, definitions, anything online. Returns "
            "titles, snippets, and URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "How many results to return (default 5)"},
            },
            "required": ["query"],
        },
    },
}


def make_web_search_handler(default_max_results: int = 5):
    def handler(query: str, max_results: int = default_max_results) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Error: web search is unavailable ({e})."

        if not results:
            return "No web results found."
        return "\n".join(
            f"[{i+1}] {r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
            for i, r in enumerate(results)
        )

    return handler
