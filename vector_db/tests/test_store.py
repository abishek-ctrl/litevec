import numpy as np
from vector_db.store import VectorStore

def test_add_and_search():
    store = VectorStore()
    store.add("vec1", np.array([1.0, 0.0]))
    store.add("vec2", np.array([0.0, 1.0]))

    query = np.array([0.9, 0.1])
    results = store.search(query, k=1)

    assert results[0][0] == "vec1", f"Expected vec1, got {results[0][0]}"
    assert 0 <= results[0][1] <= 1.0
