# LiteVec: A Minimal, Extensible Vector Database

---

## Overview

**LiteVec** is a lightweight, local vector database implementation designed for semantic search and similarity retrieval of high-dimensional vector embeddings. It provides a robust and modular foundation for storing, indexing, searching, and persisting vector data with associated metadata — all built on top of the FAISS indexing library and Hugging Face embeddings.

This is just a simple implementation of vectordbs that I worked on to get more indepth on similar databases and building them. 

---

## Features

- **Efficient Vector Indexing:** Uses Facebook AI Similarity Search (FAISS) for fast approximate nearest neighbor search.
- **Pluggable Embedding Models:** Built-in support for Hugging Face SentenceTransformer models; easily extendable.
- **Metadata Support:** Attach custom metadata to vectors and filter search results based on it.
- **Persistence:** Save and load the vector index and metadata to/from disk.
- **Insertion & Deletion:** Add new vectors with unique IDs and remove vectors logically.
- **Command-Line Interface:** Simple CLI tool to insert, search, save, and load vector data.

---

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/vector-db.git
    cd vector-db
    ```

2. (Recommended) Create and activate a Python virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate
    # On Windows: 
    venv\Scripts\activate
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

Dependencies include:
- faiss-cpu
- sentence-transformers
- numpy
- PyPDF2
- argparse

---

## Usage

### CLI Tool

The CLI tool supports basic vector database operations:

```
python -m vector_db.cli.main insert <id> "<text>"
python -m vector_db.cli.main search "<query_text>" --k 5
python -m vector_db.cli.main save <path>
python -m vector_db.cli.main load <path>
```

### Python API Example

```python
from vector_db.embedder import Embedder
from vector_db.store.faiss_store import FaissVectorStore

# Initialize embedder and store
embedder = Embedder()
store = FaissVectorStore(dim=embedder.dim)

# Insert data
text = "Machine learning enables computers to learn from data."
vec = embedder.encode([text])[0]
store.add("doc1", vec, {"source": "example"})

# Search
query = "What is machine learning?"
qvec = embedder.encode([query])[0]
results = store.search(qvec, k=3)
print(results)
```

### Example: PDF Semantic Search
The examples/pdf_search.py script demonstrates ingesting a PDF resume, chunking text, indexing embeddings, and querying semantic search.

Run:
```
python examples/pdf_search.py
```

Follow the prompt to enter queries.


---

## Comparison with Other Vector Databases

| Feature          | Vector-DB | FAISS | Qdrant | Weaviate | Chroma |
|------------------|-----------|-------|--------|----------|--------|
| Backend Index    | FAISS     | FAISS | HNSW   | HNSW     | FAISS  |
| Metadata Support | Yes       | No    | Yes    | Yes      | Yes    |
| Persistence      | Yes       | Partial | Yes  | Yes      | Yes    |
| Embedding Support| Built-in  | N/A   | Yes    | Yes      | Yes    |
| CLI Interface    | Yes       | No    | No     | No       | No     |
| Local Deployment | Yes       | Yes   | Yes    | Yes      | Yes    |
| Cloud Support    | No        | No    | Yes    | Yes      | Yes    |

---

---

## Getting Started Guide

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Basic Usage**:

```python
from vector_db import FaissVectorStore, Embedder

# Initialize components
embedder = Embedder()
store = FaissVectorStore(dim=embedder.dim)

# Add some data
store.add("doc1", embedder.encode("Sample text")[0], {"type": "example"})

# Search
results = store.search(embedder.encode("Search query")[0], k=3)
```
3. **Working with Documents**:

```python
from utils.pdf_loader import load_and_chunk_pdf

# Load PDF and process
chunks = load_and_chunk_pdf("document.pdf", chunk_size=500)
for i, chunk in enumerate(chunks):
    store.add(f"chunk_{i}", embedder.encode(chunk)[0], {"source": "document.pdf"})
```
