"""Unit tests for the LiteVecClient orchestrator."""

import os
import shutil
import tempfile
import pytest
from vector_db.client import LiteVecClient

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

def test_client_orchestration(temp_dir):
    """Test LiteVecClient end-to-end with the local TF-IDF embedder."""
    client = LiteVecClient(
        backend="flat",
        metric="cosine",
        dim=8,
        embed_backend="tfidf"
    )

    # Add items using text directly
    client.add("doc1", "The quick brown fox jumps over the lazy dog.", {"type": "animal"})
    client.add("doc2", "Artificial intelligence and machine learning technologies.", {"type": "tech"})

    # Search
    results = client.search("machine learning", k=1)
    assert len(results) == 1
    # Check it matched the tech doc
    assert results[0][0] == "doc2"
    assert results[0][2]["type"] == "tech"

    # Pre-filtering
    filtered = client.search("fox", k=5, filter={"type": "tech"})
    assert len(filtered) == 1
    assert filtered[0][0] == "doc2" # fox is not in tech doc but only tech category was queried

    # Save and Load
    save_path = os.path.join(temp_dir, "client_db")
    client.save(save_path)

    new_client = LiteVecClient(
        backend="flat",
        metric="cosine",
        dim=8,
        embed_backend="tfidf"
    )
    new_client.load(save_path)

    loaded_results = new_client.search("fox", k=1)
    assert loaded_results[0][0] == "doc1"
