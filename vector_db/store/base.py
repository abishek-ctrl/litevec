from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        """Insert a new vector. Error if ID exists."""
        pass

    @abstractmethod
    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        """Insert or update (overwrite) existing ID."""
        pass

    @abstractmethod
    def add_many(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ):
        """Batch insert vectors; may overwrite or error on duplicates depending on implementation."""
        pass

    @abstractmethod
    def delete(self, ids: List[str] = None, filter: Dict[str, Any] = None):
        """
        Delete by explicit list of IDs and/or by matching metadata filter.
        If both provided, union of deletions.
        """
        pass

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Find top-k nearest vectors, optionally filtering by metadata.
        Returns list of (ID, score, metadata).
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Persist the index and metadata to disk at `path`."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the index and metadata from disk at `path`."""
        pass
