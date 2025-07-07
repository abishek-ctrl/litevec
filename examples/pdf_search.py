import sys
import numpy as np
from vector_db.store.faiss_store import FaissVectorStore
from vector_db.embedder import Embedder
from utils.pdf_loader import extract_text_from_pdf

# CONFIG
PDF_PATH = r"data\Abishek_M.pdf"
CHUNK_SIZE = 300
DIM = 384

store = FaissVectorStore(dim=DIM)
embedder = Embedder()

# Ingest PDF
full_text = extract_text_from_pdf(PDF_PATH)
chunks = [full_text[i:i+CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]

for i, chunk in enumerate(chunks):
    vec = embedder.encode([chunk])[0]
    store.add(f"chunk_{i}", vec, {"text": chunk})

print(f"[✓] Inserted {len(chunks)} chunks from {PDF_PATH}")

query = input("Enter query: ")
qvec = embedder.encode([query])[0]
results = store.search(qvec, k=5)

print("\nTop Matches:")
for i, (id, score, meta) in enumerate(results, 1):
    print(f"{i}. {id} | Score: {score:.4f}")
    print(f"   {meta['text'][:150]}...\n")
