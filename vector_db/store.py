import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

class VectorStore:
    def __init__(self):
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []

    def add(self, vector_id: str, vector: np.ndarray) -> None:
        assert vector.ndim == 1, "Vector must be 1D"
        self.ids.append(vector_id)
        self.vectors.append(vector)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []

        matrix = np.vstack(self.vectors)
        sims = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self.ids[i], float(sims[i])) for i in top_indices]
