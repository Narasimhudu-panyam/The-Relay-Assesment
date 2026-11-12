"""
Local embeddings module using sentence-transformers.
Provides deterministic, zero-cost, local vector embeddings.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class EmbeddingEngine:
    """Generates normalized vector embeddings using local SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            # Lazy load model on first usage
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings, returning L2-normalized float32 numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning 1D or (1, D) normalized vector."""
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()
