import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Any
from vector_db.store.base import BaseVectorStore

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []
        self.vectors = []
        self.metadata = {}

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            raise ValueError(f"ID '{id}' already exists. Use upsert() to overwrite.")
        self._add_internal(id, vector, metadata)

    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            self.delete([id])
        self._add_internal(id, vector, metadata)

    def _add_internal(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        self.index.add(vector.reshape(1, -1))
        self.ids.append(id)
        self.vectors.append(vector)
        self.metadata[id] = metadata

    def delete(self, ids: List[str] = None, filter: Dict[str, Any] = None):
        to_delete = set()

        if ids:
            to_delete.update(ids)
        if filter:
            for id in self.ids:
                md = self.metadata.get(id, {})
                if all(md.get(k) == v for k, v in filter.items()):
                    to_delete.add(id)

        keep_ids = [id for id in self.ids if id not in to_delete]
        keep_vectors = [v for i, v in enumerate(self.vectors) if self.ids[i] not in to_delete]

        self.index = faiss.IndexFlatIP(self.dim)
        if keep_vectors:
            self.index.add(np.stack(keep_vectors))

        self.ids = keep_ids
        self.vectors = keep_vectors
        self.metadata = {id: self.metadata[id] for id in keep_ids}

    def search(self, vector: np.ndarray, k: int, filter: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.ids:
            return []

        vector = vector.reshape(1, -1)
        max_k = min(k * 2, len(self.ids))
        scores, indices = self.index.search(vector, max_k)

        results = []
        count = 0
        for j in range(max_k):
            i = indices[0][j]
            if i < 0 or i >= len(self.ids):
                continue

            id = self.ids[i]
            meta = self.metadata[id]

            if filter and not all(meta.get(k) == v for k, v in filter.items()):
                continue

            results.append((id, float(scores[0][j]), meta))
            count += 1
            if count == k:
                break

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump((self.ids, self.vectors, self.metadata), f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            self.ids, self.vectors, self.metadata = pickle.load(f)
        self.dim = self.vectors[0].shape[0] if self.vectors else self.dim

    def _validate_vector(self, vector: np.ndarray):
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected vector shape ({self.dim},), got {vector.shape}")
