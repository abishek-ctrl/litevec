import numpy as np
from typing import List, Tuple, Dict, Any
from vector_db.store.base import BaseVectorStore
from vector_db.store.persistence import save_memory_store, load_memory_store

class MemoryVectorStore(BaseVectorStore):
    def __init__(self, dim: int):
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.dim = dim

    def add(self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        assert vector.shape == (self.dim,)
        self.vectors.append(vector)
        self.ids.append(vector_id)
        self.metadata.append(metadata)

    def add_many(self, vector_ids: List[str], vectors: List[np.ndarray], metadata: List[Dict[str, Any]]):
        for vid, vec, meta in zip(vector_ids, vectors, metadata):
            self.add(vid, vec, meta)

    def search(self, query: np.ndarray, k: int, filters: Dict[str, Any] = None):
        sims = [np.dot(query, v) for v in self.vectors]
        idxs = np.argsort(sims)[::-1]
        results = []
        for i in idxs:
            meta = self.metadata[i]
            if filters and not all(meta.get(k) == v for k, v in filters.items()):
                continue
            results.append((self.ids[i], float(sims[i]), meta))
            if len(results) == k:
                break
        return results

    def save(self, path: str):
        save_memory_store(path, self.ids, self.vectors, self.metadata, self.dim)

    def load(self, path: str):
        self.ids, self.vectors, self.metadata, self.dim = load_memory_store(path)
