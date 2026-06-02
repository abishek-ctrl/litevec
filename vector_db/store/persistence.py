"""Serialization and deserialization helpers for LiteVec store states."""

import os
import json
from typing import List, Tuple, Dict, Any
import numpy as np

def save_memory_store(
    path: str,
    ids: List[str],
    vectors: List[np.ndarray],
    metadata: Dict[str, Dict[str, Any]],
    dim: int
) -> None:
    """Save memory store arrays and metadata dictionaries to path.

    Args:
        path: Output directory path.
        ids: List of document IDs.
        vectors: List of 1D numpy arrays.
        metadata: Dictionary of metadata objects.
        dim: Dimension of vectors.
    """
    os.makedirs(path, exist_ok=True)
    if vectors:
        np.save(os.path.join(path, "vectors.npy"), np.vstack(vectors))
    else:
        np.save(os.path.join(path, "vectors.npy"), np.array([], dtype=np.float32))

    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({
            "ids": ids,
            "metadata": metadata,
            "dim": dim
        }, f)

def load_memory_store(
    path: str
) -> Tuple[List[str], List[np.ndarray], Dict[str, Dict[str, Any]], int]:
    """Load memory store arrays and metadata from disk path.

    Args:
        path: Input directory path containing vectors.npy and meta.json.

    Returns:
        A tuple of (ids, list of vectors, metadata dictionary, dimensions).
    """
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    ids = meta["ids"]
    metadata = meta["metadata"]
    dim = meta["dim"]

    vec_path = os.path.join(path, "vectors.npy")
    if os.path.exists(vec_path):
        arr = np.load(vec_path)
        if arr.size > 0:
            vectors = [arr[i] for i in range(arr.shape[0])]
        else:
            vectors = []
    else:
        vectors = [np.zeros(dim, dtype=np.float32) for _ in ids]

    return ids, vectors, metadata, dim

def save_faiss_metadata(
    path: str,
    ids: List[str],
    metadata: Dict[str, Dict[str, Any]],
    dim: int
) -> None:
    """Save FAISS metadata schema directly to path."""
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({
            "ids": ids,
            "metadata": metadata,
            "dim": dim
        }, f)

def load_faiss_metadata(
    path: str
) -> Tuple[List[str], Dict[str, Dict[str, Any]], int]:
    """Load FAISS metadata schema from path."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    return meta["ids"], meta["metadata"], meta["dim"]
