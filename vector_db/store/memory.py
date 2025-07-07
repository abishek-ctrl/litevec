import numpy as np
from typing import List, Tuple
from vector_db.store.base import BaseVectorStore

class MemoryVectorStore(BaseVectorStore):
    def __init__(self, dim: int):
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.metadata: List[str] = []
        self.dim = dim

    def add(self, vector_id: str, vector: np.ndarray, metadata: str):
        assert vector.shape == (self.dim,), "Vector shape mismatch"
        self.vectors.append(vector)
        self.ids.append(vector_id)
        self.metadata.append(metadata)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float, str]]:
        sims = [np.dot(query, v) for v in self.vectors]
        idxs = np.argsort(sims)[::-1][:k]
        return [(self.ids[i], float(sims[i]), self.metadata[i]) for i in idxs]

    def save(self, path: str):
        from vector_db.store.persistence import save_memory_store
        save_memory_store(path, self.ids, self.vectors, self.metadata, self.dim)

    def load(self, path: str):
        from vector_db.store.persistence import load_memory_store
        self.ids, self.vectors, self.metadata, self.dim = load_memory_store(path)
