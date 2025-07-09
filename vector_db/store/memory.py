import numpy as np
from typing import List, Tuple, Dict, Any
from vector_db.store.base import BaseVectorStore
from vector_db.store.persistence import save_memory_store, load_memory_store

class MemoryVectorStore(BaseVectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def _validate_vector(self, vector: np.ndarray):
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector shape must be ({self.dim},), got {vector.shape}")

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            raise ValueError(f"ID '{id}' exists; use upsert() to overwrite.")
        self._validate_vector(vector)
        self.ids.append(id)
        self.vectors.append(vector)
        self.metadata[id] = metadata

    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            self.delete(ids=[id])
        self.add(id, vector, metadata)

    def add_many(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ):
        for _id, vec, meta in zip(ids, vectors, metadata):
            self.upsert(_id, vec, meta)

    def delete(self, ids: List[str] = None, filter: Dict[str, Any] = None):
        to_remove = set(ids or [])
        if filter:
            for _id in list(self.ids):
                md = self.metadata.get(_id, {})
                if all(md.get(k) == v for k, v in filter.items()):
                    to_remove.add(_id)

        keep_ids = [i for i in self.ids if i not in to_remove]
        keep_vectors = [self.vectors[self.ids.index(i)] for i in keep_ids]
        keep_metadata = {i: self.metadata[i] for i in keep_ids}

        self.ids = keep_ids
        self.vectors = keep_vectors
        self.metadata = keep_metadata

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        self._validate_vector(vector)
        sims = [float(np.dot(vector, v)) for v in self.vectors]
        idxs = np.argsort(sims)[::-1]
        results = []
        for i in idxs:
            _id = self.ids[i]
            md = self.metadata[_id]
            if filter and not all(md.get(k) == v for k, v in filter.items()):
                continue
            results.append((_id, sims[i], md))
            if len(results) == k:
                break
        return results

    def save(self, path: str):
        save_memory_store(path, self.ids, self.vectors, self.metadata, self.dim)

    def load(self, path: str):
        ids, vectors, metadata, dim = load_memory_store(path)
        self.ids = ids
        self.vectors = vectors
        self.metadata = metadata
        self.dim = dim
