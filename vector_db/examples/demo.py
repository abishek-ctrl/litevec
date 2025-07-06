import time
from vector_db.embedder import Embedder
from vector_db.store import VectorStore
import numpy as np

print("Initializing Embedder and VectorStore...")
start_init_time = time.time()
embedder = Embedder()
store = VectorStore()
end_init_time = time.time()
print(f"Initialization complete in {end_init_time - start_init_time:.4f} seconds.")

data = {
    "ai1": "Artificial intelligence is transforming industries.",
    "ai2": "Deep learning and neural networks are powerful tools.",
    "life1": "I went hiking and enjoyed the mountains.",
    "life2": "Nature walks help me clear my mind."
}

print(f"\nAdding {len(data)} documents to the store...")
total_add_time = 0
for key, text in data.items():
    print(f"  Processing '{key}': '{text}'")
    start_embed_time = time.time()
    vector = embedder.encode(text)
    end_embed_time = time.time()
    embed_duration = end_embed_time - start_embed_time
    print(f"    Embedding took {embed_duration:.4f} seconds.")

    start_add_to_store_time = time.time()
    store.add(key, vector)
    end_add_to_store_time = time.time()
    add_to_store_duration = end_add_to_store_time - start_add_to_store_time
    print(f"    Adding to store took {add_to_store_duration:.4f} seconds.")
    total_add_time += (embed_duration + add_to_store_duration)

print(f"Total time to add {len(data)} documents (including embedding) : {total_add_time:.4f} seconds.")


query = "Tell me about machine learning"
print(f"\nProcessing query: '{query}'")

start_query_embed_time = time.time()
query_vector = embedder.encode(query)
end_query_embed_time = time.time()
print(f"  Query embedding took {end_query_embed_time - start_query_embed_time:.4f} seconds.")

print(f"  Searching for top 3 matches in the store...")
start_search_time = time.time()
results = store.search(query_vector, k=3)
end_search_time = time.time()
print(f"  Search operation took {end_search_time - start_search_time:.4f} seconds.")

print("\nTop Results:")
for rank, (doc_id, score) in enumerate(results, 1):
    print(f"{rank}. {doc_id} — score: {score:.4f}")
print("\nDemo complete.")