import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from annoy import AnnoyIndex
from vector_db.store.base import BaseVectorStore

class AnnoyVectorStore(BaseVectorStore):
    def __init__(self, dim: int, metric: str = 'angular', n_trees: int = 10):
        self.dim = dim
        self.metric = metric
        self.n_trees = n_trees
        self.index = AnnoyIndex(dim, metric)
        self.ids: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._built = False

    def _validate_vector(self, vector):
        # Accept np.ndarray or list
        if isinstance(vector, np.ndarray):
            if vector.shape != (self.dim,):
                raise ValueError(f"Expected vector shape ({self.dim},), got {vector.shape}")
            return vector.tolist()
        elif isinstance(vector, list):
            if len(vector) != self.dim:
                raise ValueError(f"Expected vector length {self.dim}, got {len(vector)}")
            return vector
        else:
            raise TypeError("Vector must be a list or numpy.ndarray")

    def add(self, id: str, vector, metadata: Dict[str, Any]):
        if id in self.ids:
            raise ValueError(f"ID '{id}' exists; use upsert() to overwrite.")
        vec_list = self._validate_vector(vector)
        idx = len(self.ids)
        self.index.add_item(idx, vec_list)
        self.ids.append(id)
        self.metadata[id] = metadata
        self._built = False

    def upsert(self, id: str, vector, metadata: Dict[str, Any]):
        if id in self.ids:
            self.delete(ids=[id])
        self.add(id, vector, metadata)

    def add_many(
        self,
        ids: List[str],
        vectors: List,
        metadata: List[Dict[str, Any]]
    ):
        for _id, vec, meta in zip(ids, vectors, metadata):
            self.add(_id, vec, meta)

    def delete(self, ids: List[str] = None, filter: Dict[str, Any] = None):
        to_remove = set(ids or [])
        if filter:
            for _id in list(self.ids):
                md = self.metadata.get(_id, {})
                if all(md.get(k) == v for k, v in filter.items()):
                    to_remove.add(_id)

        # Determine items to keep
        keep = [(i, _id) for i, _id in enumerate(self.ids) if _id not in to_remove]
        if keep:
            keep_idxs, keep_ids = zip(*keep)
        else:
            keep_idxs, keep_ids = [], []

        # Retrieve kept vectors
        keep_vecs = [self.index.get_item_vector(i) for i in keep_idxs]
        keep_meta = {id_: self.metadata[id_] for id_ in keep_ids}

        # Rebuild the Annoy index
        self.index = AnnoyIndex(self.dim, self.metric)
        for new_i, vec in enumerate(keep_vecs):
            self.index.add_item(new_i, vec)
        self.ids = list(keep_ids)
        self.metadata = keep_meta
        self._built = False

    def _ensure_built(self):
        if not self._built:
            self.index.build(self.n_trees)
            self._built = True

    def search(
        self,
        vector,
        k: int,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        vec_list = self._validate_vector(vector)
        if not self.ids:
            return []

        self._ensure_built()
        fetch_k = min(len(self.ids), k * 2)
        candidates, distances = self.index.get_nns_by_vector(
            vec_list, n=fetch_k, include_distances=True
        )

        results = []
        for idx, dist in zip(candidates, distances):
            if idx < 0 or idx >= len(self.ids):
                continue
            _id = self.ids[idx]
            md = self.metadata[_id]
            if filter and not all(md.get(k) == v for k, v in filter.items()):
                continue
            # Convert angular distance to cosine similarity
            score = 1 - (dist * dist) / 2
            results.append((_id, score, md))
            if len(results) == k:
                break

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._ensure_built()
        idx_path = os.path.join(path, "index.ann")
        self.index.save(idx_path)
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "metadata": self.metadata,
                "dim": self.dim,
                "metric": self.metric,
                "n_trees": self.n_trees
            }, f)

    def load(self, path: str):
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"]
        self.metadata = data["metadata"]
        self.dim = data["dim"]
        self.metric = data["metric"]
        self.n_trees = data["n_trees"]

        idx_path = os.path.join(path, "index.ann")
        self.index = AnnoyIndex(self.dim, self.metric)
        self.index.load(idx_path)
        self._built = True
