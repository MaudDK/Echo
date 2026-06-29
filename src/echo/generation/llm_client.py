import json
import logging
import time
from typing import List, Dict, Any, Iterator, Optional
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

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        think: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw Ollama chat chunks as they arrive.

        Each chunk carries a partial ``message`` (``content`` and/or ``thinking``
        deltas, and ``tool_calls`` once the model decides to call them). ``think``
        asks reasoning models to expose their reasoning; models that don't
        support it reject it with 400, so we retry once without it.
        """
        def _payload(with_think: bool) -> Dict[str, Any]:
            p = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            }
            if tools:
                p["tools"] = tools
            if with_think:
                p["think"] = True
            return p

        for attempt_think in (think, False) if think else (False,):
            try:
                with self._client.stream("POST", "/api/chat", json=_payload(attempt_think)) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            yield json.loads(line)
                return
            except httpx.HTTPStatusError as e:
                # A reasoning-unaware model rejects `think`; drop it and retry.
                if attempt_think and e.response.status_code == 400:
                    logger.warning("Model rejected think=true; retrying without it")
                    continue
                raise

    def close(self):
        self._client.close()