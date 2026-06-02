"""Pluggable text embedding module with standard-library HTTP clients and custom TF-IDF fallback."""

import json
import re
import urllib.request
import urllib.error
from typing import List, Optional, Union, Dict, Any
import numpy as np

from vector_db.exceptions import EmbeddingError

class LocalTFIDFEmbedder:
    """A pure Python/NumPy TF-IDF vectorizer to generate document embeddings locally without deep learning libraries."""

    def __init__(self, dim: int = 384):
        """Initialize the TF-IDF embedder.

        Args:
            dim: Maximum vocabulary size (dimension of output embedding).
        """
        self.dim_size = dim
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text."""
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF model on a corpus of text.

        Args:
            texts: List of documents to learn vocabulary and IDF from.
        """
        if not texts:
            return

        # Count document frequencies
        doc_counts: Dict[str, int] = {}
        total_docs = len(texts)

        for text in texts:
            unique_words = set(self._tokenize(text))
            for word in unique_words:
                doc_counts[word] = doc_counts.get(word, 0) + 1

        # Select top words up to dim_size
        sorted_words = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.dim_size]

        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        
        # Calculate IDF
        self.idf = {}
        for word, count in top_words:
            self.idf[word] = float(np.log((1 + total_docs) / (1 + count)) + 1)
        
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform a list of texts into TF-IDF vector embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            A 2D numpy array of shape (len(texts), dim).
        """
        num_docs = len(texts)
        vectors = np.zeros((num_docs, self.dim_size), dtype=np.float32)

        if not self._fitted:
            # Fit on the fly if not already fitted
            self.fit(texts)

        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            
            # Compute term frequencies
            tf: Dict[str, int] = {}
            for token in tokens:
                if token in self.vocabulary:
                    tf[token] = tf.get(token, 0) + 1
            
            total_tokens = len(tokens)
            for token, count in tf.items():
                idx = self.vocabulary[token]
                vectors[i, idx] = (count / total_tokens) * self.idf[token]
                
            # Normalize vector (L2 norm)
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] = vectors[i] / norm

        return vectors


class Embedder:
    """Orchestrates different text embedding generators."""

    def __init__(
        self,
        backend: str = "tfidf",
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        dim: int = 384
    ):
        """Initialize the Embedder.

        Args:
            backend: The embedding backend: 'tfidf', 'ollama', 'hf_api', or 'hf' (local).
            model_name: Name of the embedding model to use.
            api_url: Optional custom HTTP endpoint for Ollama or Hugging Face.
            api_token: Optional authorization token (e.g. Hugging Face Inference API).
            dim: Dimension size (primarily used for local TF-IDF vectorizer).
        """
        self.backend = backend.lower()
        self.model_name = model_name
        self.api_url = api_url
        self.api_token = api_token
        self._dim = dim

        if self.backend == "tfidf":
            self.model_name = model_name or "local-tfidf"
            self.local_model = LocalTFIDFEmbedder(dim=self._dim)
            self._dim = dim

        elif self.backend == "ollama":
            self.model_name = model_name or "nomic-embed-text"
            self.api_url = api_url or "http://localhost:11434/api/embeddings"
            self._dim = None  # Detected on first run

        elif self.backend == "hf_api":
            self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.api_url = api_url or f"https://api-inference.huggingface.co/models/{self.model_name}"
            self._dim = None

        elif self.backend == "hf":
            # Optional local SentenceTransformers support
            self.model_name = model_name or "all-MiniLM-L6-v2"
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer(self.model_name)
                self._dim = self.local_model.get_sentence_embedding_dimension()
            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Use backend='tfidf' or install sentence-transformers."
                ) from e
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def fit(self, texts: List[str]) -> None:
        """Fit the underlying model if the backend requires it (e.g. TF-IDF).

        Args:
            texts: Corpus of texts to fit on.
        """
        if self.backend == "tfidf":
            self.local_model.fit(texts)

    @property
    def dim(self) -> int:
        """Embedding dimension size."""
        if self._dim is None:
            raise ValueError("Embedding dimension unknown until first encode call.")
        return self._dim

    def _query_ollama(self, text: str) -> List[float]:
        """Perform HTTP POST request to Ollama's local embedding endpoint."""
        payload = json.dumps({"model": self.model_name, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                vec = result.get("embedding")
                if not vec:
                    raise EmbeddingError("Ollama response did not contain 'embedding' key")
                return vec
        except Exception as e:
            raise EmbeddingError(f"Ollama API request failed: {e}") from e

    def _query_hf_api(self, texts: List[str]) -> List[List[float]]:
        """Perform HTTP POST request to Hugging Face Inference API."""
        payload = json.dumps({"inputs": texts}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        req = urllib.request.Request(self.api_url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
                # HF Inference API returns a list of floats (for a single input) or list of lists
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], list):
                        return result
                    elif len(result) > 0 and isinstance(result[0], float):
                        return [result]
                raise EmbeddingError("Hugging Face API returned invalid feature structure")
        except Exception as e:
            raise EmbeddingError(f"Hugging Face Inference API request failed: {e}") from e

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode a single text or list of texts into normalized embedding vectors.

        Args:
            texts: String or List of strings to encode.

        Returns:
            A 2D numpy array of shape (len(texts), dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.backend == "tfidf":
            return self.local_model.transform(texts)

        if self.backend == "hf":
            return self.local_model.encode(texts, convert_to_numpy=True)

        if self.backend == "ollama":
            vectors = []
            for text in texts:
                vec = self._query_ollama(text)
                vectors.append(vec)
            res = np.array(vectors, dtype=np.float32)
            if self._dim is None and res.ndim > 1:
                self._dim = res.shape[1]
            return res

        if self.backend == "hf_api":
            vecs = self._query_hf_api(texts)
            res = np.array(vecs, dtype=np.float32)
            if self._dim is None and res.ndim > 1:
                self._dim = res.shape[1]
            return res

        raise ValueError("Invalid backend state configuration")
