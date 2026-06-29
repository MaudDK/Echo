import logging
import time
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.max_retries = max_retries

    def check_health(self):
        try:
            response = self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {e}")
            return False

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, temperature: float = 0.7) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }

        if tools:
            payload["tools"] = tools

        last_exec = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post("/api/chat", json=payload)
                if response.status_code != 200:
                    logger.error(f"Failed to generate response: {response.json()}")
                response.raise_for_status()
                return response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exec = e
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait}s.")
                time.sleep(wait)
        
        raise RuntimeError(f"Ollama service unreachable after {self.max_retries} retries") from last_exec

    def close(self):
        self._client.close()