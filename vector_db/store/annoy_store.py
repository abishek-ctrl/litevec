import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from annoy import AnnoyIndex
from vector_db.store.base import BaseVectorStore

class AnnoyVectorStore(BaseVectorStore):
    def __init__(
        self,
        dim: int,
        metric: str = 'angular',
        n_trees: int = 10
    ):
        """
        Annoy-based vector store.

        Args:
            dim: Embedding dimensionality.
            metric: 'angular', 'euclidean', 'manhattan', etc.
            n_trees: Number of trees for the index (trade-off speed/accuracy).
        """
        self.dim = dim
        self.index = AnnoyIndex(dim, metric)
        self.n_trees = n_trees
        self.ids: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        if id in self.ids:
            raise ValueError(f"ID '{id}' exists; use upsert() to overwrite.")
        idx = len(self.ids)
        self.index.add_item(idx, vector.tolist())
        self.ids.append(id)
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
        for id_, vec_, meta_ in zip(ids, vectors, metadata):
            self.add(id_, vec_, meta_)

    def delete(self, ids: List[str] = None, filter: Dict[str, Any] = None):
        to_remove = set(ids or [])
        if filter:
            for id_ in self.ids:
                m = self.metadata.get(id_, {})
                if all(m.get(k) == v for k, v in filter.items()):
                    to_remove.add(id_)

        keep = [(i, id_) for i, id_ in enumerate(self.ids) if id_ not in to_remove]
        keep_idxs, keep_ids = zip(*keep) if keep else ([], [])
        keep_vecs = [self.index.get_item_vector(i) for i in keep_idxs]
        keep_meta = {id_: self.metadata[id_] for id_ in keep_ids}

        # rebuild index
        self.index = AnnoyIndex(self.dim, self.index.metric)
        for new_idx, vec in enumerate(keep_vecs):
            self.index.add_item(new_idx, vec)
        self.index.build(self.n_trees)

        self.ids = list(keep_ids)
        self.metadata = keep_meta

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        self.index.build(self.n_trees)
        candidates, distances = self.index.get_nns_by_vector(
            vector.tolist(), n=k * 2, include_distances=True
        )
        results = []
        for idx, dist in zip(candidates, distances):
            if idx < 0 or idx >= len(self.ids):
                continue
            id_ = self.ids[idx]
            meta = self.metadata[id_]
            if filter and not all(meta.get(k) == v for k, v in filter.items()):
                continue
            # Annoy’s “angular” distance: convert to similarity
            score = 1 - dist if self.index.metric == 'angular' else float(dist)
            results.append((id_, score, meta))
            if len(results) == k:
                break
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        idx_path = os.path.join(path, "index.ann")
        self.index.save(idx_path)
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "metadata": self.metadata,
                "dim": self.dim,
                "n_trees": self.n_trees,
                "metric": self.index.metric
            }, f)

    def load(self, path: str):
        idx_path = os.path.join(path, "index.ann")
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"]
        self.metadata = data["metadata"]
        self.dim = data["dim"]
        self.n_trees = data["n_trees"]
        metric = data["metric"]
        self.index = AnnoyIndex(self.dim, metric)
        self.index.load(idx_path)
