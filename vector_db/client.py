"""Unified client orchestrator for the LiteVec package."""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

from vector_db.embedder import Embedder
from vector_db.store.base import BaseVectorStore
from vector_db.store.faiss_store import FaissVectorStore
from vector_db.store.hnsw_store import HNSWVectorStore
from vector_db.store.annoy_store import AnnoyVectorStore
from vector_db.store.memory import MemoryVectorStore

class LiteVecClient:
    """Orchestrates embedding generation and index operations under a single unified interface."""

    def __init__(
        self,
        backend: str = "flat",
        metric: str = "cosine",
        dim: Optional[int] = None,
        embed_backend: str = "tfidf",
        embed_model: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """Initialize the client.

        Args:
            backend: The storage backend: 'flat', 'hnsw', 'annoy', or 'memory'.
            metric: The distance metric: 'cosine', 'l2', or 'ip'.
            dim: Dimension of vector embeddings (optional if determined by embedder).
            embed_backend: Embedding model backend: 'tfidf', 'hf_api', 'ollama', or 'hf'.
            embed_model: Model name for remote or local embedders.
            api_token: Bearer token for remote Inference APIs.
        """
        self.embedder = Embedder(
            backend=embed_backend,
            model_name=embed_model,
            api_token=api_token,
            dim=dim or 384
        )
        
        # Try to infer dimension if embedder supports it
        try:
            self.dim = self.embedder.dim
        except ValueError:
            self.dim = dim or 384

        self.backend_name = backend.lower()
        self.metric = metric.lower()
        self.store: BaseVectorStore

        if self.backend_name == "flat":
            self.store = FaissVectorStore(dim=self.dim, metric=self.metric)
        elif self.backend_name == "hnsw":
            self.store = HNSWVectorStore(dim=self.dim, metric=self.metric)
        elif self.backend_name == "annoy":
            self.store = AnnoyVectorStore(dim=self.dim, metric=self.metric)
        elif self.backend_name == "memory":
            self.store = MemoryVectorStore(dim=self.dim, metric=self.metric)
        else:
            raise ValueError(f"Unknown storage backend: {backend}")

    def fit(self, texts: List[str]) -> None:
        """Fit the client's embedder on a corpus of text (useful for TF-IDF).

        Args:
            texts: List of strings to learn the vocabulary from.
        """
        self.embedder.fit(texts)

    def add(self, id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Embed text and insert it into the store.

        Args:
            id: Unique document identifier.
            text: Text content to embed and index.
            metadata: Associated metadata dictionary.
        """
        meta = metadata or {}
        meta["text"] = text
        vector = self.embedder.encode(text)[0]
        self.store.add(id, vector, meta)

    def add_many(self, ids: List[str], texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Embed a batch of texts and insert them into the store.

        Args:
            ids: List of unique document identifiers.
            texts: List of text contents.
            metadatas: Optional list of metadata dictionaries.
        """
        vectors = [v for v in self.embedder.encode(texts)]
        meta_list = metadatas or [{} for _ in ids]
        for m, t in zip(meta_list, texts):
            m["text"] = t
        self.store.add_many(ids, vectors, meta_list)

    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """Delete items from the database by ID list or metadata filter query.

        Args:
            ids: List of document IDs.
            filter: Structured query filter dictionary.
        """
        self.store.delete(ids=ids, filter=filter)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search nearest text items.

        Args:
            query: Query string.
            k: Top-k matches to retrieve.
            filter: Structured query filter dictionary.

        Returns:
            List of search results: (id, score, metadata) ordered by similarity descending.
        """
        q_vector = self.embedder.encode(query)[0]
        return self.store.search(q_vector, k=k, filter=filter)

    def save(self, path: str) -> None:
        """Save the store state to disk."""
        self.store.save(path)

    def load(self, path: str) -> None:
        """Load the store state from disk."""
        self.store.load(path)
