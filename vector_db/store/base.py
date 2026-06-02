"""Abstract base class interface for LiteVec vector stores."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class BaseVectorStore(ABC):
    """Abstract Base Class defining the interface for all vector stores."""

    @abstractmethod
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert a new vector and its associated metadata.

        Args:
            id: Unique identifier for the vector.
            vector: 1D numpy array representing the embedding.
            metadata: Structured dictionary containing metadata.

        Raises:
            DuplicateIDError: If the ID already exists in the store.
            DimensionMismatchError: If the vector dimensions do not match the store's dimension.
        """
        pass

    @abstractmethod
    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert a new vector or overwrite an existing one if the ID already exists.

        Args:
            id: Unique identifier.
            vector: 1D numpy array.
            metadata: Associated metadata dictionary.

        Raises:
            DimensionMismatchError: If the vector dimensions do not match the store's dimension.
        """
        pass

    @abstractmethod
    def add_many(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Batch insert or update vectors and metadata.

        Args:
            ids: List of unique identifiers.
            vectors: List of 1D numpy arrays.
            metadata: List of metadata dictionaries.
        """
        pass

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """Delete vectors from the store by explicit IDs or by metadata query filter.

        If both are specified, the union of matching vectors is deleted.

        Args:
            ids: Optional list of document IDs to delete.
            filter: Optional structured query filter.
        """
        pass

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Query top-k nearest neighbors, with optional structured metadata query filtering.

        Args:
            vector: 1D query vector.
            k: Number of neighbors to return.
            filter: Optional query filter specification.

        Returns:
            A list of tuples: (id, similarity_score, metadata) ordered by similarity descending.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the store state and indexes to the local disk.

        Args:
            path: Directory path where files will be written.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the store state and indexes from the local disk.

        Args:
            path: Directory path where state files are saved.
        """
        pass
