from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]): ...

    @abstractmethod
    def add_many(self, vector_ids: List[str], vectors: List[np.ndarray], metadata: List[Dict[str, Any]]): ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int, filters: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]: ...

    @abstractmethod
    def save(self, path: str): ...

    @abstractmethod
    def load(self, path: str): ...
