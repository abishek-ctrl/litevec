"""Unit tests for LiteVec embedders (local TF-IDF and mock HTTP remote clients)."""

import json
import pytest
from unittest.mock import patch, MagicMock
import urllib.error
import numpy as np

from vector_db.embedder import Embedder, LocalTFIDFEmbedder
from vector_db.exceptions import EmbeddingError

def test_local_tfidf_embedder():
    """Verify that the local TF-IDF embedder fits and encodes documents into normalized vectors."""
    corpus = [
        "The sky is blue and beautiful today.",
        "Neural networks are a kind of machine learning model.",
        "I love hiking in the mountains."
    ]
    embedder = LocalTFIDFEmbedder(dim=10)
    embedder.fit(corpus)
    
    vectors = embedder.transform(["blue sky neural networks"])
    assert vectors.shape == (1, 10)
    # Norm of vectors should be 1.0 (L2 normalized)
    assert pytest.approx(np.linalg.norm(vectors[0])) == 1.0

def test_embedder_tfidf_backend():
    """Verify Embedder orchestration with tfidf backend."""
    embedder = Embedder(backend="tfidf", dim=32)
    assert embedder.dim == 32
    res = embedder.encode("test string")
    assert res.shape == (1, 32)
    assert pytest.approx(np.linalg.norm(res[0])) == 1.0

@patch("urllib.request.urlopen")
def test_embedder_ollama_backend_success(mock_urlopen):
    """Verify Ollama REST API response parsing on success."""
    # Mock HTTP response
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    embedder = Embedder(backend="ollama", model_name="nomic-embed-text")
    res = embedder.encode("hello")
    assert res.shape == (1, 3)
    assert embedder.dim == 3
    np.testing.assert_allclose(res[0], [0.1, 0.2, 0.3], atol=1e-6)

@patch("urllib.request.urlopen")
def test_embedder_ollama_backend_failure(mock_urlopen):
    """Verify Ollama remote errors raise EmbeddingError."""
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    embedder = Embedder(backend="ollama")
    
    with pytest.raises(EmbeddingError):
        embedder.encode("hello")

@patch("urllib.request.urlopen")
def test_embedder_hf_api_backend_success(mock_urlopen):
    """Verify HF serverless feature extraction endpoint query."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps([[0.5, 0.6, 0.7]]).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    embedder = Embedder(backend="hf_api", model_name="sentence-transformers/all-MiniLM-L6-v2")
    res = embedder.encode("world")
    assert res.shape == (1, 3)
    assert embedder.dim == 3
    np.testing.assert_allclose(res[0], [0.5, 0.6, 0.7], atol=1e-6)
