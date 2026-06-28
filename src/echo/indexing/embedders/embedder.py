import logging
import time
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from echo.config import load_yaml

logger = logging.getLogger(__name__)

prefix_config = load_yaml('embedder/prefixes.yaml')

class Embedder:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 32, normalize: bool = True, 
                 trust_remote_code: bool = True, warm_up: bool = True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.normalize = normalize

        model_prefix = prefix_config.get(model_name, {})
        if not model_prefix:
            logger.warning(
                f"No known prefix convention for '{model_name}' — using no prefix. "
                f"Verify this is correct, or add an entry to prefixes.yaml."
            )

        self.query_prefix = model_prefix.get("query", "")
        self.passage_prefix = model_prefix.get("passage", "")
        
        logger.info(f"Loading model '{model_name}' on device '{self.device}'")
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code, device=self.device)
        self.dim = self.model.get_embedding_dimension()
        
        if warm_up:
            self._warm_up()

    def _warm_up(self):
        _ = self.model.encode(["warmup"], normalize_embeddings=True)
        logger.info(f"Model successfully loaded and warmed up. Embedding dimension: {self.dim}")

    def encode(self, texts: List[str], is_query: bool = False, normalize: bool = None, batch_size: int = None, show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            raise ValueError("Embedder: No texts provided for encoding.")
        
        normalize = self.normalize if normalize is None else normalize
        batch_size = self.batch_size if batch_size is None else batch_size

        prefix = self.query_prefix if is_query else self.passage_prefix
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        embeds = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar= not is_query,
            convert_to_numpy=True
        )
        
        return embeds