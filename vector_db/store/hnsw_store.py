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
        """
        HNSW-backed vector store.

        Args:
            dim: Embedding dimensionality.
            space: 'l2', 'ip', or 'cosine'.
            max_elements: max number of elements to index.
            ef_construction: HNSW efConstruction parameter.
            M: HNSW M parameter.
        """
        self.dim = dim
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef_construction)
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[str, Any] = {}

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            raise ValueError(f"ID '{id}' already exists; use upsert() to overwrite.")
        self._add_internal(id, vector, metadata)

    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            self.delete(ids=[id])
        self._add_internal(id, vector, metadata)

    def _add_internal(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        assert vector.shape == (self.dim,), f"Vector must have shape ({self.dim},)"
        self.index.add_items(vector.reshape(1, -1), np.array([len(self.ids)]))
        self.ids.append(id)
        self.vectors.append(vector)
        self.metadata[id] = metadata

    def add_many(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ):
        idxs = np.arange(len(self.ids), len(self.ids) + len(ids))
        data = np.vstack(vectors)
        assert data.shape[1] == self.dim
        self.index.add_items(data, idxs)
        for id_, vec_, meta_ in zip(ids, vectors, metadata):
            self.ids.append(id_)
            self.vectors.append(vec_)
            self.metadata[id_] = meta_

    def delete(self, ids: List[str] = None, filter: Dict[str, Any] = None):
        to_remove = set()
        if ids:
            to_remove.update(ids)
        if filter:
            for id_ in self.ids:
                m = self.metadata.get(id_, {})
                if all(m.get(k) == v for k, v in filter.items()):
                    to_remove.add(id_)

        keep = [(i, id_) for i, id_ in enumerate(self.ids) if id_ not in to_remove]
        keep_idxs, keep_ids = zip(*keep) if keep else ([], [])
        keep_vecs = [self.vectors[i] for i in keep_idxs]
        keep_meta = {id_: self.metadata[id_] for id_ in keep_ids}

        # rebuild index
        self.index = hnswlib.Index(space=self.index.space, dim=self.dim)
        self.index.init_index(max_elements=len(keep_vecs), ef_construction=self.index.ef_construction, M=self.index.M)
        if keep_vecs:
            self.index.add_items(np.vstack(keep_vecs), np.arange(len(keep_vecs)))
        self.ids = list(keep_ids)
        self.vectors = keep_vecs
        self.metadata = keep_meta

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.ids:
            return []

        labels, distances = self.index.knn_query(vector.reshape(1, -1), k=k * 2)
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            id_ = self.ids[idx]
            meta = self.metadata[id_]
            if filter and not all(meta.get(k) == v for k, v in filter.items()):
                continue
            score = 1 - dist if self.index.space == 'cosine' else float(dist)
            results.append((id_, score, meta))
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
                "dim": self.dim
            }, f)

    def load(self, path: str):
        self.index = hnswlib.Index(space=self.index.space, dim=self.dim)
        self.index.load_index(os.path.join(path, "index.bin"))
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"]
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]
        self.dim = data["dim"]
