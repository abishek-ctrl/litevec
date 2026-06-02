"""Pure Python/NumPy implementation of a Random Projection Forest for Approximate Nearest Neighbor search."""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from vector_db.store.base import BaseVectorStore
from vector_db.exceptions import DimensionMismatchError, DuplicateIDError
from vector_db.filter import evaluate_filter
from vector_db.metrics import (
    cosine_similarity,
    l2_distance,
    inner_product
)

class RPTNode:
    """A node in a Random Projection Tree."""

    def __init__(self) -> None:
        self.is_leaf: bool = False
        self.indices: List[int] = []  # Leaf stores vector indices
        self.v: Optional[np.ndarray] = None  # Normal vector of hyperplane
        self.d: float = 0.0  # Median offset
        self.left: Optional['RPTNode'] = None
        self.right: Optional['RPTNode'] = None


class AnnoyVectorStore(BaseVectorStore):
    """Random Projection Forest vector store replacing the external Spotify Annoy library.

    Retains name 'AnnoyVectorStore' for backward compatibility.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        n_trees: int = 10,
        leaf_size: int = 32
    ):
        """Initialize the RP Forest vector store.

        Args:
            dim: Dimension of vectors.
            metric: Distance metric: 'cosine', 'l2', or 'ip'.
            n_trees: Number of projection trees in the forest.
            leaf_size: Maximum number of vectors inside a leaf node.
        """
        self.dim = dim
        self.metric = metric.lower()
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.trees: List[RPTNode] = []
        self._built = False

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
        self._built = False

    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert or update existing vector."""
        vec = self._validate_vector(vector)
        if id in self.metadata:
            idx = self.ids.index(id)
            self.vectors[idx] = vec
            self.metadata[id] = metadata
            self._built = False
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
        """Delete items by ID list or metadata filter."""
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
        self.trees = []
        self._built = False

    def _build_tree(self, indices: List[int]) -> RPTNode:
        """Recursively build a single Random Projection Tree."""
        node = RPTNode()
        if len(indices) <= self.leaf_size:
            node.is_leaf = True
            node.indices = indices
            return node

        # Extract vectors for these indices
        subset = [self.vectors[i] for i in indices]

        # Choose a random projection direction (hyperplane)
        # Select two points randomly, normal is the line connecting them
        idx1, idx2 = np.random.choice(len(indices), size=2, replace=False)
        v = subset[idx1] - subset[idx2]
        norm = np.linalg.norm(v)
        if norm == 0:
            # All points are identical
            node.is_leaf = True
            node.indices = indices
            return node
        v = v / norm

        # Project points
        projections = [float(np.dot(x, v)) for x in subset]
        median_val = float(np.median(projections))

        left_idx = []
        right_idx = []
        for i, proj in zip(indices, projections):
            if proj <= median_val:
                left_idx.append(i)
            else:
                right_idx.append(i)

        # Fallback if partition failed to split
        if not left_idx or not right_idx:
            node.is_leaf = True
            node.indices = indices
            return node

        node.v = v
        node.d = median_val
        node.left = self._build_tree(left_idx)
        node.right = self._build_tree(right_idx)
        return node

    def _ensure_built(self) -> None:
        """Ensure forest trees are populated if not already built."""
        if not self._built and self.ids:
            self.trees = []
            indices = list(range(len(self.ids)))
            for _ in range(self.n_trees):
                self.trees.append(self._build_tree(indices))
            self._built = True

    def _search_tree(self, node: RPTNode, query_vec: np.ndarray) -> List[int]:
        """Traverse down a tree to collect candidate leaf indices."""
        if node.is_leaf:
            return node.indices
        proj = float(np.dot(query_vec, node.v))
        if proj <= node.d:
            return self._search_tree(node.left, query_vec)
        else:
            return self._search_tree(node.right, query_vec)

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search approximately using the random projection forest."""
        query_vec = self._validate_vector(vector)
        if not self.ids:
            return []

        self._ensure_built()

        # 1. Gather index candidates from all trees
        candidates = set()
        for tree in self.trees:
            candidates.update(self._search_tree(tree, query_vec))

        # 2. Score candidates
        results = []
        for idx in candidates:
            doc_id = self.ids[idx]
            # Metadata pre-filtering
            if filter and not evaluate_filter(self.metadata[doc_id], filter):
                continue
            
            vec = self.vectors[idx]
            if self.metric == "cosine":
                score = cosine_similarity(query_vec, vec)
            elif self.metric == "ip":
                score = inner_product(query_vec, vec)
            else:
                score = l2_distance(query_vec, vec)
            
            results.append((doc_id, score, self.metadata[doc_id]))

        # 3. Sort results
        if self.metric in ("cosine", "ip"):
            # Higher score is closer
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            # Lower distance is closer
            results.sort(key=lambda x: x[1])

        return results[:k]

    def save(self, path: str) -> None:
        """Save state and forest structure."""
        os.makedirs(path, exist_ok=True)
        self._ensure_built()
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "vectors": self.vectors,
                "metadata": self.metadata,
                "dim": self.dim,
                "metric": self.metric,
                "n_trees": self.n_trees,
                "leaf_size": self.leaf_size,
                "trees": self.trees
            }, f)

    def load(self, path: str) -> None:
        """Load state and forest structure."""
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"]
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]
        self.dim = data["dim"]
        self.metric = data["metric"]
        self.n_trees = data["n_trees"]
        self.leaf_size = data["leaf_size"]
        self.trees = data["trees"]
        self._built = True
