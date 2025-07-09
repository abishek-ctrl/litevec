import time
import numpy as np
from vector_db.embedder import Embedder
from vector_db.store.faiss_store import FaissVectorStore
from vector_db.store.hnsw_store import HNSWVectorStore
from vector_db.store.annoy_store import AnnoyVectorStore
from utils.pdf_loader import extract_text_from_pdf

def chunk_text(text: str, chunk_size=50, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def build_index(store, embedder, chunks):
    vectors = embedder.encode(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadata = [{"text": c} for c in chunks]

    if isinstance(store, AnnoyVectorStore):
        vs = vectors.tolist()
    else:
        vs = vectors

    store.add_many(ids, vs, metadata)
    return store

def benchmark_search(store, embedder, query, top_k=3):
    vec = embedder.encode([query])[0]
    start = time.time()
    results = store.search(vec, k=top_k)
    duration = time.time() - start
    return results, duration

def display_results(name, results, duration):
    print(f"\n{name} (Search Time: {duration:.4f}s)")
    print("-" * 60)
    for i, (cid, score, meta) in enumerate(results, 1):
        print(f"[{i}] {score:.4f} → {meta['text'][:120]}...")
    print("-" * 60)

def main():
    text = extract_text_from_pdf("data/Grades.pdf")
    chunks = chunk_text(text)

    print(f"\nTotal chunks: {len(chunks)}\n")
    for idx, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {idx} ---\n{chunk[:100]}...\n")

    embedder = Embedder(backend="ollama", model_name="nomic-embed-text:latest")
    # Warm up to detect dim
    _ = embedder.encode([chunks[0]])
    dim = embedder.dim

    print("Building FAISS index...")
    faiss_store = build_index(FaissVectorStore(dim), embedder, chunks)

    print("Building HNSW index...")
    hnsw_store = build_index(HNSWVectorStore(dim), embedder, chunks)

    print("Building Annoy index...")
    annoy_store = build_index(AnnoyVectorStore(dim), embedder, chunks)

    stores = [
        ("FAISS", faiss_store),
        ("HNSW", hnsw_store),
        ("Annoy", annoy_store),
    ]

    print("\nReady. Type your query or 'exit' to quit.\n")
    while True:
        query = input("🔍 Query > ").strip()
        if query.lower() == "exit":
            break
        for name, store in stores:
            results, duration = benchmark_search(store, embedder, query)
            display_results(name, results, duration)

if __name__ == "__main__":
    main()
