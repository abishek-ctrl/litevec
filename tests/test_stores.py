"""Integration tests validating standard CRUD operations across all custom vector stores."""

import shutil
import tempfile
import os
import pytest
import numpy as np

from vector_db.store.faiss_store import FaissVectorStore
from vector_db.store.hnsw_store import HNSWVectorStore
from vector_db.store.annoy_store import AnnoyVectorStore
from vector_db.store.memory import MemoryVectorStore
from vector_db.exceptions import DimensionMismatchError, DuplicateIDError

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

# Parametrize to run all tests against every backend
@pytest.mark.parametrize("store_class", [
    FaissVectorStore,
    HNSWVectorStore,
    AnnoyVectorStore,
    MemoryVectorStore
])
def test_store_crud_operations(store_class, temp_dir):
    """Verify basic add, search, duplicate checks, dimension checks, and deletion works."""
    dim = 4
    store = store_class(dim=dim)

    # 1. Add elements
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    vec3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    store.add("doc1", vec1, {"category": "a", "tag": 1})
    store.add("doc2", vec2, {"category": "b", "tag": 2})
    store.add("doc3", vec3, {"category": "a", "tag": 3})

    # 2. Test duplicate ID error
    with pytest.raises(DuplicateIDError):
        store.add("doc1", vec1, {})

    # 3. Test dimension mismatch error
    with pytest.raises(DimensionMismatchError):
        store.add("doc4", np.array([1.0, 0.0], dtype=np.float32), {})

    # 4. Search elements
    # Query closest to vec1
    results = store.search(vec1, k=2)
    assert len(results) >= 1
    # Check that doc1 is first and closest
    assert results[0][0] == "doc1"

    # 5. Search with structured pre-filtering
    filtered_results = store.search(vec1, k=5, filter={"category": "a"})
    # Only doc1 and doc3 should match category 'a'
    matched_ids = [item[0] for item in filtered_results]
    assert "doc1" in matched_ids
    assert "doc3" in matched_ids
    assert "doc2" not in matched_ids

    # 6. Delete elements by ID
    store.delete(ids=["doc1"])
    post_delete_results = store.search(vec1, k=5)
    matched_post_ids = [item[0] for item in post_delete_results]
    assert "doc1" not in matched_post_ids

    # 7. Delete elements by filter
    store.delete(filter={"category": "b"})  # should delete doc2
    final_results = store.search(vec1, k=5)
    final_ids = [item[0] for item in final_results]
    assert "doc2" not in final_ids
    assert "doc3" in final_ids  # doc3 is still there

@pytest.mark.parametrize("store_class", [
    FaissVectorStore,
    HNSWVectorStore,
    AnnoyVectorStore,
    MemoryVectorStore
])
def test_store_persistence(store_class, temp_dir):
    """Verify that saving and reloading does not cause data loss or vector corruption."""
    dim = 3
    store = store_class(dim=dim)
    
    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    store.add("doc1", vec1, {"label": "x"})
    store.add("doc2", vec2, {"label": "y"})
    
    # Save to temp path
    save_path = os.path.join(temp_dir, "db_state")
    store.save(save_path)

    # Load into a new store instance
    loaded_store = store_class(dim=dim)
    loaded_store.load(save_path)

    # Assert structural equality
    assert loaded_store.ids == ["doc1", "doc2"]
    assert loaded_store.metadata["doc1"] == {"label": "x"}
    
    # Ensure vectors loaded match
    np.testing.assert_allclose(loaded_store.vectors[0], vec1)
    np.testing.assert_allclose(loaded_store.vectors[1], vec2)

    # Verify search after loading
    results = loaded_store.search(vec1, k=1)
    assert results[0][0] == "doc1"

    # Make a deletion on the loaded store to ensure no corruption (Bug 1 fix verification)
    loaded_store.delete(ids=["doc1"])
    remaining = loaded_store.search(vec2, k=1)
    assert len(remaining) == 1
    assert remaining[0][0] == "doc2"
    # Ensure the actual vector wasn't corrupted to zero
    np.testing.assert_allclose(loaded_store.vectors[0], vec2)
