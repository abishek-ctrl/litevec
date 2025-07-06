import argparse
from vector_db.embedder import Embedder
from vector_db.store import VectorStore

store = VectorStore()
embedder = Embedder()

def insert(vector_id: str, text: str) -> None:
    vector = embedder.encode(text)
    store.add(vector_id, vector)
    print(f"[✓] Inserted: {vector_id}")

def search(text: str, k: int = 5) -> None:
    vector = embedder.encode(text)
    results = store.search(vector, k)
    print(f"\nTop {k} matches for query: '{text}'")
    for idx, (rid, score) in enumerate(results):
        print(f"{idx+1}. {rid} — score: {score:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Vector DB CLI")
    subparsers = parser.add_subparsers(dest="command")

    parser_insert = subparsers.add_parser("insert")
    parser_insert.add_argument("id", type=str)
    parser_insert.add_argument("text", type=str)

    parser_search = subparsers.add_parser("search")
    parser_search.add_argument("text", type=str)
    parser_search.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "insert":
        insert(args.id, args.text)
    elif args.command == "search":
        search(args.text, args.k)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
