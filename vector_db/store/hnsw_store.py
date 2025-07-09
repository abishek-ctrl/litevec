import os
import pickle
import numpy as np
import hnswlib
from typing import List, Tuple, Dict, Any
from vector_db.store.base import BaseVectorStore

class HNSWVectorStore(BaseVectorStore):
    def __init__(
        self,
        dim: int,
        space: str = 'cosine',
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16
    ):
        self.dim = dim
        self.space = space
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M

        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef_construction)

        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def _validate_vector(self, vector):
        vector = np.array(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector must be of shape ({self.dim},), got {vector.shape}")

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            raise ValueError(f"ID '{id}' exists; use upsert() to overwrite.")
        self._validate_vector(vector)
        idx = len(self.ids)
        self.index.add_items(vector.reshape(1, -1), np.array([idx]))
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

        keep = [(i, _id) for i, _id in enumerate(self.ids) if _id not in to_remove]
        if keep:
            keep_idxs, keep_ids = zip(*keep)
        else:
            keep_idxs, keep_ids = [], []

        keep_vecs = [self.vectors[i] for i in keep_idxs]
        keep_meta = {i: self.metadata[i] for i in keep_ids}

        # rebuild
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.M)
        if keep_vecs:
            self.index.add_items(np.vstack(keep_vecs), np.arange(len(keep_vecs)))
        self.index.set_ef(self.ef_construction)

        self.ids = list(keep_ids)
        self.vectors = keep_vecs
        self.metadata = keep_meta

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform a k-NN search with metadata filtering.
        Clamps k to the number of items and ensures ef >= k for reliable results.
        """
        self._validate_vector(vector)
        n = len(self.ids)
        if n == 0:
            return []

        # How many neighbors to fetch before filtering
        fetch_k = min(k * 2, n)

        # Ensure the search ef (recall parameter) is at least fetch_k
        # ef parameter controls neighbors inspected; higher = more accurate
        self.index.set_ef(max(self.ef_construction, fetch_k))

        # Now perform the query
        labels, distances = self.index.knn_query(vector.reshape(1, -1), k=fetch_k)

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx < 0 or idx >= n:
                continue
            _id = self.ids[idx]
            md = self.metadata[_id]
            if filter and not all(md.get(field) == val for field, val in filter.items()):
                continue

            score = 1 - dist if self.space == 'cosine' else float(dist)
            results.append((_id, score, md))

            if len(results) == k:
                break

        return results


    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.index.save_index(os.path.join(path, "index.bin"))
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "vectors": self.vectors,
                "metadata": self.metadata,
                "dim": self.dim,
                "space": self.space,
                "max_elements": self.max_elements,
                "ef_construction": self.ef_construction,
                "M": self.M
            }, f)

    def load(self, path: str):
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        self.dim = data["dim"]
        self.space = data["space"]
        self.max_elements = data["max_elements"]
        self.ef_construction = data["ef_construction"]
        self.M = data["M"]
        self.ids = data["ids"]
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]

        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(os.path.join(path, "index.bin"))
        self.index.set_ef(self.ef_construction)
