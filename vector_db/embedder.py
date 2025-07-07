from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [np.array(vec) for vec in embeddings]
