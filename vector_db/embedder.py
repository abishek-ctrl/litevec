from typing import List, Optional
import numpy as np
from functools import lru_cache

# Hugging‑Face
from sentence_transformers import SentenceTransformer

# Ollama
import ollama


class Embedder:
    def __init__(
        self,
        backend: str = "hf",
        model_name: Optional[str] = None
    ):
        """
        A pluggable embedder supporting:
          - Hugging‑Face SentenceTransformer (backend='hf')
          - Ollama (backend='ollama')

        Args:
            backend: 'hf' or 'ollama'.
            model_name: 
              - for 'hf': HF model name (default: 'all-MiniLM-L6-v2')
              - for 'ollama': Ollama model (default: 'nomic-embed-text:latest')
        """
        self.backend = backend.lower()
        if self.backend == "hf":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            self._dim = self.model.get_sentence_embedding_dimension()

            @lru_cache(maxsize=4096)
            def _hf_cache(text: str) -> tuple:
                vec = self.model.encode(text, convert_to_numpy=True)
                return tuple(vec)
            self._embed_cache = _hf_cache

        elif self.backend == "ollama":
            self.model_name = model_name or "nomic-embed-text:latest"
            # Defer dimension detection until first embed
            self._dim = None

            @lru_cache(maxsize=2048)
            def _ollama_cache(text: str) -> tuple:
                resp = ollama.embed(model=self.model_name, input=text)
                vec = resp.get("embeddings", [])
                if not vec:
                    raise ValueError(f"Ollama returned no embeddings for input: {text!r}")
                tup = tuple(vec[0])
                # set dim on first real call
                if self._dim is None:
                    self._dim = len(tup)
                return tup
            self._embed_cache = _ollama_cache

        else:
            raise ValueError(f"Unsupported backend '{backend}'")

    @property
    def dim(self) -> int:
        """Embedding dimensionality. For Ollama it's set on first embed."""
        if self._dim is None:
            raise ValueError("Embedding dimension unknown until after first embed call")
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into shape (len(texts), dim).
        Caches single-string calls for speed.
        """
        if self.backend == "hf":
            if len(texts) == 1:
                vec = np.array(self._embed_cache(texts[0]), dtype=np.float32)
                return vec.reshape(1, -1)
            return self.model.encode(texts, convert_to_numpy=True)

        # Ollama
        vecs = []
        for text in texts:
            tup = self._embed_cache(text)
            vecs.append(np.array(tup, dtype=np.float32))
        return np.vstack(vecs)

    def __repr__(self):
        return f"<Embedder backend={self.backend!r} model_name={self.model_name!r} dim={self._dim!r}>"
