"""Vector distance and similarity metrics implemented using NumPy."""

import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors.

    Args:
        v1: 1D numpy array.
        v2: 1D numpy array.

    Returns:
        The cosine similarity score (between -1 and 1).
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def l2_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two 1D vectors.

    Args:
        v1: 1D numpy array.
        v2: 1D numpy array.

    Returns:
        The L2 distance.
    """
    return float(np.linalg.norm(v1 - v2))

def inner_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute inner product between two 1D vectors.

    Args:
        v1: 1D numpy array.
        v2: 1D numpy array.

    Returns:
        The dot product value.
    """
    return float(np.dot(v1, v2))

def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of vectors.

    Args:
        query: 1D numpy array of shape (dim,).
        matrix: 2D numpy array of shape (num_vectors, dim).

    Returns:
        1D numpy array of shape (num_vectors,) containing similarity scores.
    """
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)
    
    # Avoid division by zero
    q_norm = 1.0 if q_norm == 0 else q_norm
    m_norms[m_norms == 0] = 1.0
    
    dot_products = np.dot(matrix, query)
    return dot_products / (q_norm * m_norms)

def l2_distance_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute L2 distance between a query vector and a matrix of vectors.

    Args:
        query: 1D numpy array of shape (dim,).
        matrix: 2D numpy array of shape (num_vectors, dim).

    Returns:
        1D numpy array of shape (num_vectors,) containing L2 distances.
    """
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    return np.linalg.norm(matrix - query, axis=1)

def inner_product_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute inner product between a query vector and a matrix of vectors.

    Args:
        query: 1D numpy array of shape (dim,).
        matrix: 2D numpy array of shape (num_vectors, dim).

    Returns:
        1D numpy array of shape (num_vectors,) containing dot products.
    """
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    return np.dot(matrix, query)
