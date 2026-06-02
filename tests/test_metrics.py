"""Unit tests for the LiteVec distance and similarity metrics."""

import numpy as np
import pytest
from vector_db.metrics import (
    cosine_similarity,
    l2_distance,
    inner_product,
    cosine_similarity_matrix,
    l2_distance_matrix,
    inner_product_matrix,
)

def test_cosine_similarity_identical_vectors():
    """Verify that identical vectors yield a cosine similarity of 1.0."""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(v, v)) == 1.0

def test_cosine_similarity_orthogonal_vectors():
    """Verify that orthogonal vectors yield a cosine similarity of 0.0."""
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(v1, v2)) == 0.0

def test_cosine_similarity_opposite_vectors():
    """Verify that opposite vectors yield a cosine similarity of -1.0."""
    v1 = np.array([1.0, -1.0], dtype=np.float32)
    v2 = np.array([-1.0, 1.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(v1, v2)) == -1.0

def test_l2_distance_calculation():
    """Verify the Euclidean distance computation between two known vectors."""
    v1 = np.array([0.0, 0.0], dtype=np.float32)
    v2 = np.array([3.0, 4.0], dtype=np.float32)
    assert pytest.approx(l2_distance(v1, v2)) == 5.0

def test_inner_product_calculation():
    """Verify the dot product calculation."""
    v1 = np.array([1.0, 2.0], dtype=np.float32)
    v2 = np.array([3.0, 4.0], dtype=np.float32)
    assert pytest.approx(inner_product(v1, v2)) == 11.0

def test_cosine_similarity_matrix():
    """Verify vectorized matrix cosine similarity matches sequential evaluation."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ], dtype=np.float32)
    expected = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    np.testing.assert_allclose(cosine_similarity_matrix(query, matrix), expected, atol=1e-6)

def test_l2_distance_matrix():
    """Verify vectorized L2 distance calculations."""
    query = np.array([0.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [3.0, 4.0],
        [0.0, 1.0]
    ], dtype=np.float32)
    expected = np.array([5.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(l2_distance_matrix(query, matrix), expected, atol=1e-6)

def test_inner_product_matrix():
    """Verify vectorized Inner Product calculations."""
    query = np.array([1.0, 2.0], dtype=np.float32)
    matrix = np.array([
        [3.0, 4.0],
        [5.0, 6.0]
    ], dtype=np.float32)
    expected = np.array([11.0, 17.0], dtype=np.float32)
    np.testing.assert_allclose(inner_product_matrix(query, matrix), expected, atol=1e-6)
