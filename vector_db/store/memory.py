"""Pure Python/NumPy in-memory vector store utilizing LiteVec metrics and filter compilation."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from vector_db.store.base import BaseVectorStore
from vector_db.store.persistence import save_memory_store, load_memory_store
from vector_db.exceptions import DimensionMismatchError, DuplicateIDError
from vector_db.filter import evaluate_filter
from vector_db.metrics import (
    cosine_similarity_matrix,
    l2_distance_matrix,
    inner_product_matrix
)

class MemoryVectorStore(BaseVectorStore):
    """In-memory vector database store."""

    def __init__(self, dim: int, metric: str = "cosine"):
        """Initialize the store."""
        self.dim = dim
        self.metric = metric.lower()
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}

        if self.metric not in ("cosine", "l2", "ip"):
            raise ValueError(f"Unsupported metric: {metric}")

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        vec = np.array(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise DimensionMismatchError(
                f"Expected vector shape ({self.dim},), got {vec.shape}"
            )
        return vec

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add a single vector."""
        if id in self.metadata:
            raise DuplicateIDError(f"ID '{id}' already exists in store.")
        vec = self._validate_vector(vector)
        self.ids.append(id)
        self.vectors.append(vec)
        self.metadata[id] = metadata

    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert or update existing vector."""
        vec = self._validate_vector(vector)
        if id in self.metadata:
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
        """Delete items by ID or metadata filter."""
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
        """Query top-k nearest neighbors with pre-filtering."""
        query_vec = self._validate_vector(vector)
        if not self.ids:
            return []

        # 1. Pre-filtering
        candidates = []
        for idx, doc_id in enumerate(self.ids):
            if filter and not evaluate_filter(self.metadata[doc_id], filter):
                continue
            candidates.append((idx, doc_id))

        if not candidates:
            return []

        # 2. Score calculations
        cand_indices = [idx for idx, _ in candidates]
        cand_vectors = np.vstack([self.vectors[idx] for idx in cand_indices])

        if self.metric == "cosine":
            scores = cosine_similarity_matrix(query_vec, cand_vectors)
            sorted_indices = np.argsort(scores)[::-1]
        elif self.metric == "ip":
            scores = inner_product_matrix(query_vec, cand_vectors)
            sorted_indices = np.argsort(scores)[::-1]
        else:
            scores = l2_distance_matrix(query_vec, cand_vectors)
            sorted_indices = np.argsort(scores)

        # 3. Format output
        results = []
        for idx in sorted_indices[:k]:
            cand_idx, doc_id = candidates[idx]
            results.append((doc_id, float(scores[idx]), self.metadata[doc_id]))
        return results

    def save(self, path: str) -> None:
        save_memory_store(path, self.ids, self.vectors, self.metadata, self.dim)

    def load(self, path: str) -> None:
        ids, vectors, metadata, dim = load_memory_store(path)
        self.ids = ids
        self.vectors = vectors
        self.metadata = metadata
        self.dim = dim
