import os
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
from vector_db.store.base import BaseVectorStore
from vector_db.store.persistence import save_faiss_metadata, load_faiss_metadata

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        assert vector.shape == (self.dim,)
        self.index.add(vector.reshape(1, -1))
        self.ids.append(vector_id)
        self.metadata.append(metadata)

    def add_many(self, vector_ids: List[str], vectors: List[np.ndarray], metadata: List[Dict[str, Any]]):
        self.index.add(np.vstack(vectors))
        self.ids.extend(vector_ids)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, k: int, filters: Dict[str, Any] = None):
        D, I = self.index.search(query.reshape(1, -1), len(self.ids))
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if filters and not all(meta.get(k) == v for k, v in filters.items()):
                continue
            results.append((self.ids[idx], float(score), meta))
            if len(results) == k:
                break
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        save_faiss_metadata(path, self.ids, self.metadata, self.dim)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        self.ids, self.metadata, self.dim = load_faiss_metadata(path)
