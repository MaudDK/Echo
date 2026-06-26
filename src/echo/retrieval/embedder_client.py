import logging
import time
from typing import List
import httpx
import numpy as np

logger = logging.getLogger(__name__)

class EmbedderClient:
    def __init__(self, base_url: str, timeout: float = 10.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.max_retries = max_retries

    def check_health(self):
        try:
            response = self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {e}")
            return False

    def embed(self, texts: List[str], is_query: bool = False, normalize: bool = True) -> np.ndarray:
        last_exec = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(
                    "/embed",
                    json={"inputs": texts, "is_query": is_query, "normalize": normalize},
                )
                if response.status_code != 200:
                    logger.error(f"Failed to get embeddings: {response.json()}")
                response.raise_for_status()
                return np.array(response.json()["embeddings"], dtype=np.float32)
            
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exec = e
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait}s.")
                time.sleep(wait)

        raise RuntimeError(f"Embedder service unreachable after {self.max_retries} retries") from last_exec
    
    def close(self):
        self._client.close()
