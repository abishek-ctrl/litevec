"""Pure Python/NumPy Hierarchical Navigable Small World (HNSW) graph implementation for ANN search."""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Set, Any, Optional

from vector_db.store.base import BaseVectorStore
from vector_db.exceptions import DimensionMismatchError, DuplicateIDError
from vector_db.filter import evaluate_filter
from vector_db.metrics import (
    cosine_similarity,
    l2_distance,
    inner_product
)

class HNSWVectorStore(BaseVectorStore):
    """Hierarchical Navigable Small World (HNSW) vector store.

    Replaces the external hnswlib library. Retains 'HNSWVectorStore' for compatibility.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        max_elements: int = 10000,
        ef_construction: int = 200,
        ef_search: int = 50,
        M: int = 16
    ):
        """Initialize the HNSW graph store.

        Args:
            dim: Dimension of vectors.
            metric: Distance metric: 'cosine', 'l2', or 'ip'.
            max_elements: Maximum expected elements in the index.
            ef_construction: Candidate exploration size during graph building.
            ef_search: Candidate exploration size during querying.
            M: Number of bidirectional links created per node on insertion.
        """
        self.dim = dim
        self.metric = metric.lower()
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        self.M0 = 2 * M
        self.level_mult = 1.0 / np.log(M)

        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # graph[level][node_idx] = set of neighbor node_idx
        self.graphs: Dict[int, Dict[int, Set[int]]] = {}
        self.enter_node: Optional[int] = None
        self.max_level: int = -1

        if self.metric not in ("cosine", "l2", "ip"):
            raise ValueError(f"Unsupported metric: {metric}")

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        vec = np.array(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise DimensionMismatchError(
                f"Expected vector shape ({self.dim},), got {vec.shape}"
            )
        return vec

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate distance where smaller is closer.

        For Cosine/IP we invert similarity.
        """
        if self.metric == "cosine":
            return 1.0 - cosine_similarity(v1, v2)
        elif self.metric == "ip":
            return -inner_product(v1, v2)
        else:
            return l2_distance(v1, v2)

    def _search_layer(
        self,
        query: np.ndarray,
        enter_points: Set[int],
        ef: int,
        level: int
    ) -> List[Tuple[float, int]]:
        """Search a specific layer greedily with beam search."""
        # candidates sorted by distance (closest first)
        candidates = []
        for ep in enter_points:
            candidates.append((self._distance(query, self.vectors[ep]), ep))
        candidates.sort()

        visited = set(enter_points)
        # best results (max heap like behavior using sorted list)
        results = list(candidates)

        while candidates:
            curr_dist, curr_idx = candidates.pop(0)

            # Pruning threshold
            if curr_dist > results[-1][0]:
                break

            neighbors = self.graphs[level].get(curr_idx, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query, self.vectors[neighbor])
                    
                    if dist < results[-1][0] or len(results) < ef:
                        candidates.append((dist, neighbor))
                        results.append((dist, neighbor))
                        results.sort()
                        if len(results) > ef:
                            results.pop()
                            
            candidates.sort()

        return results

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert a vector into the HNSW graph."""
        if id in self.metadata:
            raise DuplicateIDError(f"ID '{id}' already exists in store.")
        vec = self._validate_vector(vector)

        new_idx = len(self.ids)
        self.ids.append(id)
        self.vectors.append(vec)
        self.metadata[id] = metadata

        # Determine level for new node
        level = int(floor_level := -np.log(np.random.uniform(1e-9, 1.0)) * self.level_mult)
        if level > 20: # cap level sanity
            level = 20

        # Initialize levels in graphs if needed
        for lvl in range(level + 1):
            if lvl not in self.graphs:
                self.graphs[lvl] = {}
            self.graphs[lvl][new_idx] = set()

        curr_eps = {self.enter_node} if self.enter_node is not None else set()

        # Step 1: Find entry point from top level down to level+1
        for lvl in range(self.max_level, level, -1):
            curr_eps = {self._search_layer(vec, curr_eps, 1, lvl)[0][1]}

        # Step 2: Insert into levels from min(level, max_level) down to 0
        for lvl in range(min(level, self.max_level), -1, -1):
            results = self._search_layer(vec, curr_eps, self.ef_construction, lvl)
            
            # Form connections
            neighbors = [idx for _, idx in results[:self.M]]
            self.graphs[lvl][new_idx] = set(neighbors)
            
            # Bidirectional links
            for nb in neighbors:
                self.graphs[lvl][nb].add(new_idx)
                # Shrink if connection count exceeds threshold
                max_conn = self.M0 if lvl == 0 else self.M
                if len(self.graphs[lvl][nb]) > max_conn:
                    sorted_nb = sorted(
                        self.graphs[lvl][nb],
                        key=lambda x: self._distance(self.vectors[nb], self.vectors[x])
                    )
                    self.graphs[lvl][nb] = set(sorted_nb[:max_conn])
                    
            curr_eps = set(idx for _, idx in results)

        # Update entry point if new node level exceeds max level
        if level > self.max_level:
            self.max_level = level
            self.enter_node = new_idx

    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert or update existing vector."""
        vec = self._validate_vector(vector)
        if id in self.metadata:
            self.delete(ids=[id])
        self.add(id, vec, metadata)

    def add_many(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> None:
        for _id, vec, meta in zip(ids, vectors, metadata):
            self.upsert(_id, vec, meta)

    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """Delete items. Rebuilds the graph to guarantee graph connectivity."""
        to_remove = set(ids or [])
        if filter:
            for _id in self.ids:
                if evaluate_filter(self.metadata[_id], filter):
                    to_remove.add(_id)

        if not to_remove:
            return

        # Keep lists
        keep_indices = [idx for idx, _id in enumerate(self.ids) if _id not in to_remove]
        
        old_ids = list(self.ids)
        old_vectors = list(self.vectors)
        old_metadata = dict(self.metadata)

        # Clear state
        self.ids = []
        self.vectors = []
        self.metadata = {}
        self.graphs = {}
        self.enter_node = None
        self.max_level = -1

        # Re-insert kept elements to ensure correct connectivity
        for idx in keep_indices:
            self.add(old_ids[idx], old_vectors[idx], old_metadata[old_ids[idx]])

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Query top-k approximate nearest neighbors with metadata pre-filtering."""
        query_vec = self._validate_vector(vector)
        if not self.ids or self.enter_node is None:
            return []

        curr_eps = {self.enter_node}
        
        # Traverse down layers
        for lvl in range(self.max_level, 0, -1):
            curr_eps = {self._search_layer(query_vec, curr_eps, 1, lvl)[0][1]}

        # Search bottom layer
        candidates = self._search_layer(query_vec, curr_eps, max(self.ef_search, k), 0)

        # Filter and sort
        results = []
        for dist, idx in candidates:
            doc_id = self.ids[idx]
            if filter and not evaluate_filter(self.metadata[doc_id], filter):
                continue
            
            vec = self.vectors[idx]
            if self.metric == "cosine":
                score = cosine_similarity(query_vec, vec)
            elif self.metric == "ip":
                score = inner_product(query_vec, vec)
            else:
                score = dist
            
            results.append((doc_id, score, self.metadata[doc_id]))

        if self.metric in ("cosine", "ip"):
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            results.sort(key=lambda x: x[1])

        return results[:k]

    def save(self, path: str) -> None:
        """Save HNSW state and graph structures."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "vectors": self.vectors,
                "metadata": self.metadata,
                "graphs": self.graphs,
                "enter_node": self.enter_node,
                "max_level": self.max_level,
                "dim": self.dim,
                "metric": self.metric,
                "max_elements": self.max_elements,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "M": self.M
            }, f)

    def load(self, path: str) -> None:
        """Load HNSW state and graph structures."""
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"]
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]
        self.graphs = data["graphs"]
        self.enter_node = data["enter_node"]
        self.max_level = data["max_level"]
        self.dim = data["dim"]
        self.metric = data["metric"]
        self.max_elements = data["max_elements"]
        self.ef_construction = data["ef_construction"]
        self.ef_search = data["ef_search"]
        self.M = data["M"]
