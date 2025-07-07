import faiss
import numpy as np
import json
import os
from typing import List, Tuple
from vector_db.store.base import BaseVectorStore

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Cosine sim over normalized vectors
        self.ids: List[str] = []
        self.metadata: List[str] = []

    def add(self, vector_id: str, vector: np.ndarray, metadata: str):
        assert vector.shape == (self.dim,), "Vector has incorrect dimension"
        self.index.add(vector.reshape(1, -1))
        self.ids.append(vector_id)
        self.metadata.append(metadata)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float, str]]:
        assert query.shape == (self.dim,), "Query vector dimension mismatch"
        D, I = self.index.search(query.reshape(1, -1), k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append((self.ids[idx], float(score), self.metadata[idx]))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "ids": self.ids,
                "metadata": self.metadata,
                "dim": self.dim
            }, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        self.ids = meta["ids"]
        self.metadata = meta["metadata"]
        self.dim = meta["dim"]
