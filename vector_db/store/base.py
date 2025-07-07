from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, vector_id: str, vector: np.ndarray, metadata: str): ...
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float, str]]: ...
    
    @abstractmethod
    def save(self, path: str): ...
    
    @abstractmethod
    def load(self, path: str): ...
