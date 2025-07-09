import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from vector_db.store.base import BaseVectorStore
from vector_db.store.persistence import save_faiss_metadata, load_faiss_metadata

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.vectors: List[np.ndarray] = []

    def _validate_vector(self, vector):
        vector = np.array(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector must be of shape ({self.dim},), got {vector.shape}")


    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            raise ValueError(f"ID '{id}' exists; use upsert() to overwrite.")
        self._validate_vector(vector)
        self.index.add(vector.reshape(1, -1))
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

        # rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        if keep_vectors:
            self.index.add(np.vstack(keep_vectors))

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
        if not self.ids:
            return []

        # request extra for filtering
        fetch_k = min(len(self.ids), k * 2)
        scores, indices = self.index.search(vector.reshape(1, -1), fetch_k)
        results = []
        count = 0
        for j, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            _id = self.ids[idx]
            md = self.metadata[_id]
            if filter and not all(md.get(k) == v for k, v in filter.items()):
                continue
            results.append((_id, float(scores[0][j]), md))
            count += 1
            if count == k:
                break
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        save_faiss_metadata(path, self.ids, self.metadata, self.dim)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        ids, metadata, dim = load_faiss_metadata(path)
        self.ids = ids
        self.metadata = metadata
        self.dim = dim
        self.vectors = [np.zeros(dim) for _ in self.ids]
