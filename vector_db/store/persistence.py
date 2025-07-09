import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any

def save_memory_store(
    path: str,
    ids: List[str],
    vectors: List[np.ndarray],
    metadata: Dict[str, Dict[str, Any]],
    dim: int
):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "vectors.npy"), np.vstack(vectors))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({
            "ids": ids,
            "metadata": metadata,
            "dim": dim
        }, f)

def load_memory_store(
    path: str
) -> Tuple[List[str], List[np.ndarray], Dict[str, Dict[str, Any]], int]:
    vectors = np.load(os.path.join(path, "vectors.npy"))
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    ids = meta["ids"]
    metadata = meta["metadata"]
    dim = meta["dim"]
    return ids, [vec for vec in vectors], metadata, dim

def save_faiss_metadata(
    path: str,
    ids: List[str],
    metadata: Dict[str, Dict[str, Any]],
    dim: int
):
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({
            "ids": ids,
            "metadata": metadata,
            "dim": dim
        }, f)

def load_faiss_metadata(
    path: str
) -> Tuple[List[str], Dict[str, Dict[str, Any]], int]:
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    return meta["ids"], meta["metadata"], meta["dim"]
