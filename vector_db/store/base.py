from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any

class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        pass

    @abstractmethod
    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        pass

    @abstractmethod
    def delete(self, ids: List[str] = None, filter: Dict[str, Any] = None):
        pass

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
