Markdown

# 🧠 Vector Database

This project is a  **vector database engine** built entirely from scratch — without relying on any third-party vector DB like FAISS, Pinecone, or Chroma. It offers real **semantic search** using transformer-based embeddings, efficient vector indexing (brute-force or ANN), and exposes a clean **REST API** for integration.

> ✅ Built as an original end-to-end project to demonstrate vector search engineering, architecture, and applied ML skills.

---

## 🚀 Features

| Component            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Custom Vector Store** | In-memory + optional FAISS-based ANN backend                             |
| **Sentence Embeddings** | Uses `sentence-transformers` (`MiniLM-L6`) for high-quality embeddings  |
| **Cosine Similarity Search** | Both brute-force and ANN via normalized inner product               |
| **Full Metadata Support** | Store and retrieve associated metadata for every document              |
| **Persistence** | Save/load vectors + metadata from JSON                                    |
| **REST API** | Built using FastAPI for `/insert` and `/search` endpoints                  |
| **Modular Design** | Swap out storage/index/encoder layers with ease                           |

---


## 📊 Updates

### Phase 1 Release - July 06, 2025

**Description:**
Implemented the foundational in-memory vector database with real semantic embeddings and a command-line interface (CLI) for basic insert and search operations. Focused on establishing the core concept and modular structure.

**Results (Demo):**
The demo script execution showcased the performance characteristics of initial setup, document ingestion, and query processing.

Initialization complete in 25.3287 seconds.

Processing query: 'Tell me about machine learning'
Query embedding took 0.0211 seconds.
Searching for top 3 matches in the store...
Search operation took 0.0064 seconds.

Top Results:

ai2 — score: 0.4732

ai1 — score: 0.4276

life2 — score: 0.0863

*Note: The initial "Initialization" time (e.g., 25.3287 seconds) often includes the first-time download and loading of the `sentence-transformers` model. Subsequent runs will typically see much faster initialization if the model is cached.*

## 🔧 Built With

* **Sentence-Transformers** – Embedding model (`MiniLM-L6-v2`)
* **FAISS (optional)** – High-speed Approximate Nearest Neighbor (ANN) engine
* **FastAPI** – Modern, fast (high-performance) web framework for building APIs
* **numpy**, **scikit-learn** – Essential libraries for efficient vector mathematics and cosine similarity calculations

*Note: While FAISS is used optionally to speed up search, the core database infrastructure and i