from vector_db.store.faiss_store import FaissVectorStore
from vector_db.embedder import Embedder

store = FaissVectorStore(dim=384)  # default for all-MiniLM-L6-v2 is 384
embedder = Embedder()

docs = {
    "doc1": "The sky is blue and beautiful today.",
    "doc2": "Artificial intelligence is changing the world.",
    "doc3": "I love hiking in the mountains.",
    "doc4": "Neural networks are a kind of machine learning model.",
}


for doc_id, text in docs.items():
    store.add(doc_id, embedder.encode(text)[0], {"text": text})

query = "What are neural networks?"
query_vec = embedder.encode(query)[0]
results = store.search(query_vec, k=2)

print(f"\nQuery: {query}")
for rid, score, meta in results:
    print(f"{rid} — score: {score:.4f}")
    print(f"    → {meta.get('text', '')}")
