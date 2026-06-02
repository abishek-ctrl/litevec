"""Pure NumPy flat vector store implementing exact nearest neighbor search and structured pre-filtering."""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from vector_db.store.base import BaseVectorStore
from vector_db.exceptions import DimensionMismatchError, DuplicateIDError
from vector_db.filter import evaluate_filter
from vector_db.metrics import (
    cosine_similarity_matrix,
    l2_distance_matrix,
    inner_product_matrix
)

class FaissVectorStore(BaseVectorStore):
    """Pure NumPy implementation of a flat vector store (exact nearest neighbors).

    Retains the class name 'FaissVectorStore' for backwards compatibility.
    """

    def __init__(self, dim: int, metric: str = "cosine"):
        """Initialize the flat store.

        Args:
            dim: Dimension of the vectors.
            metric: The distance metric: 'cosine', 'l2', or 'ip'.
        """
        self.dim = dim
        self.metric = metric.lower()
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}

        if self.metric not in ("cosine", "l2", "ip"):
            raise ValueError(f"Unsupported metric: {metric}")

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        """Validate vector type and dimension."""
        vec = np.array(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise DimensionMismatchError(
                f"Expected vector shape ({self.dim},), got {vec.shape}"
            )
        return vec

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert a vector. Raises DuplicateIDError if ID exists."""
        if id in self.metadata:
            raise DuplicateIDError(f"ID '{id}' already exists in store.")
        vec = self._validate_vector(vector)
        self.ids.append(id)
        self.vectors.append(vec)
        self.metadata[id] = metadata

    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert or update existing vector and metadata."""
        vec = self._validate_vector(vector)
        if id in self.metadata:
            # Locate index
            idx = self.ids.index(id)
            self.vectors[idx] = vec
            self.metadata[id] = metadata
        else:
            self.add(id, vec, metadata)

    def add_many(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Insert multiple vectors."""
        for _id, vec, meta in zip(ids, vectors, metadata):
            self.upsert(_id, vec, meta)

    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """Delete vectors by ID or by metadata filter query."""
        to_remove = set(ids or [])
        if filter:
            for _id in self.ids:
                if evaluate_filter(self.metadata[_id], filter):
                    to_remove.add(_id)

        if not to_remove:
            return

        keep_indices = [idx for idx, _id in enumerate(self.ids) if _id not in to_remove]
        self.ids = [self.ids[idx] for idx in keep_indices]
        self.vectors = [self.vectors[idx] for idx in keep_indices]
        self.metadata = {k: v for k, v in self.metadata.items() if k not in to_remove}

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find top-k nearest vectors. Implements structured pre-filtering."""
        query_vec = self._validate_vector(vector)
        if not self.ids:
            return []

        # 1. Pre-filtering: identify candidate indices
        candidates: List[Tuple[int, str]] = []
        for idx, doc_id in enumerate(self.ids):
            if filter and not evaluate_filter(self.metadata[doc_id], filter):
                continue
            candidates.append((idx, doc_id))

        if not candidates:
            return []

        # 2. Extract vectors for candidates
        cand_indices = [idx for idx, _ in candidates]
        cand_vectors = np.vstack([self.vectors[idx] for idx in cand_indices])

        # 3. Compute distances
        if self.metric == "cosine":
            scores = cosine_similarity_matrix(query_vec, cand_vectors)
            # Higher cosine similarity is closer
            sorted_indices = np.argsort(scores)[::-1]
        elif self.metric == "ip":
            scores = inner_product_matrix(query_vec, cand_vectors)
            sorted_indices = np.argsort(scores)[::-1]
        else:  # L2 distance
            scores = l2_distance_matrix(query_vec, cand_vectors)
            # Lower L2 distance is closer
            sorted_indices = np.argsort(scores)

        # 4. Construct response
        results = []
        for idx in sorted_indices[:k]:
            cand_idx, doc_id = candidates[idx]
            results.append((doc_id, float(scores[idx]), self.metadata[doc_id]))
        return results

    def save(self, path: str) -> None:
        """Persist state to disk."""
        os.makedirs(path, exist_ok=True)
        # Save vectors as npy
        if self.vectors:
            np.save(os.path.join(path, "vectors.npy"), np.vstack(self.vectors))
        else:
            # save empty array
            np.save(os.path.join(path, "vectors.npy"), np.array([], dtype=np.float32))

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "ids": self.ids,
                "metadata": self.metadata,
                "dim": self.dim,
                "metric": self.metric
            }, f)

    def load(self, path: str) -> None:
        """Restore state from disk."""
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        self.ids = meta["ids"]
        self.metadata = meta["metadata"]
        self.dim = meta["dim"]
        self.metric = meta.get("metric", "cosine")

        vec_path = os.path.join(path, "vectors.npy")
        if os.path.exists(vec_path):
            arr = np.load(vec_path)
            if arr.size > 0:
                self.vectors = [arr[i] for i in range(arr.shape[0])]
            else:
                self.vectors = []
        else:
            self.vectors = [np.zeros(self.dim, dtype=np.float32) for _ in self.ids]
