from typing import List, Optional
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        """
        Embedder using Hugging Face SentenceTransformer.

        Args:
            model_name (str, optional): Hugging Face model name.
                Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        """Returns embedding dimensionality."""
        return self._dim

    @lru_cache(maxsize=4096)
    def _encode_str_cached(self, text: str) -> tuple:
        vec = self.model.encode(text, convert_to_numpy=True)
        return tuple(vec)  # Cacheeee added bruv!

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into vectors.

        Uses cache for single strings.
        """
        if len(texts) == 1:
            return np.array([self._encode_str_cached(texts[0])], dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
