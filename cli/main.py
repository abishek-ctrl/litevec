import argparse
from vector_db.embedder import Embedder
from vector_db.store.faiss_store import FaissVectorStore

DIM = 384
store = FaissVectorStore(dim=DIM)
embedder = Embedder()

def insert(id: str, text: str):
    vec = embedder.encode(text)
    store.add(id, vec, text)
    print(f"[+] Inserted '{id}'")

def search(text: str, k: int):
    vec = embedder.encode(text)
    results = store.search(vec, k)
    print(f"\nTop {k} results:")
    for i, (rid, score, original) in enumerate(results, 1):
        print(f"{i}. {rid} - score: {score:.4f}\n   Text: {original}")

def save(path: str):
    store.save(path)
    print(f"[✓] Saved index to {path}")

def load(path: str):
    store.load(path)
    print(f"[✓] Loaded index from {path}")

def upsert(id: str, text: str):
    vec = embedder.encode([text])[0]
    store.upsert(id, vec, {"text": text})
    print(f"[+] Upserted '{id}'")

def delete(ids: List[str], filter: Dict[str, str]):
    store.delete(ids=ids, filter=filter)
    print(f"[✓] Deleted items matching IDs or filter")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("save").add_argument("path")
    subparsers.add_parser("load").add_argument("path")

    insert_parser = subparsers.add_parser("insert")
    insert_parser.add_argument("id")
    insert_parser.add_argument("text")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("text")
    search_parser.add_argument("--k", type=int, default=5)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--ids", nargs='*', default=[])
    delete_parser.add_argument("--filter", nargs='*', help="key=value filters", default=[])

    upsert_parser = subparsers.add_parser("upsert")
    upsert_parser.add_argument("id")
    upsert_parser.add_argument("text")

    args = parser.parse_args()

    if args.command == "insert":
        insert(args.id, args.text)
    elif args.command == "search":
        search(args.text, args.k)
    elif args.command == "save":
        save(args.path)
    elif args.command == "load":
        load(args.path)
    elif args.command == "delete":
        filters = dict(f.split("=", 1) for f in args.filter)
        delete(ids=args.ids, filter=filters)
    elif args.command == "upsert":
        upsert(args.id, args.text)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
