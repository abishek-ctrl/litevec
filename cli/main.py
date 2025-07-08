import argparse
import sys
from vector_db.embedder import Embedder
from vector_db.store.faiss_store import FaissVectorStore
from vector_db.store.hnsw_store import HNSWVectorStore
from vector_db.store.annoy_store import AnnoyVectorStore

def get_store(backend: str, dim: int):
    if backend == "faiss":
        return FaissVectorStore(dim=dim)
    elif backend == "hnsw":
        return HNSWVectorStore(dim=dim)
    elif backend == "annoy":
        return AnnoyVectorStore(dim=dim)
    else:
        raise ValueError(f"Unknown backend '{backend}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="faiss",
                        choices=["faiss", "hnsw", "annoy"], help="Index backend")
    parser.add_argument("--dim", type=int, default=None,
                        help="Embedding dimension (required if not loading)")
    subparsers = parser.add_subparsers(dest="command")

    parser_save = subparsers.add_parser("save")
    parser_save.add_argument("path")

    parser_load = subparsers.add_parser("load")
    parser_load.add_argument("path")

    parser_insert = subparsers.add_parser("insert")
    parser_insert.add_argument("id")
    parser_insert.add_argument("text")

    parser_upsert = subparsers.add_parser("upsert")
    parser_upsert.add_argument("id")
    parser_upsert.add_argument("text")

    parser_delete = subparsers.add_parser("delete")
    parser_delete.add_argument("--ids", nargs="*", default=[])
    parser_delete.add_argument("--filter", nargs="*", default=[])

    parser_search = subparsers.add_parser("search")
    parser_search.add_argument("text")
    parser_search.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    embedder = Embedder()
    dim = args.dim or embedder.dim
    store = get_store(args.backend, dim)

    if args.command in ["load", "search", "insert", "upsert", "delete"]:
        try:
            store.load("data")
        except Exception:
            pass

    if args.command == "insert":
        vec = embedder.encode([args.text])[0]
        store.add(args.id, vec, {"text": args.text})
    elif args.command == "upsert":
        vec = embedder.encode([args.text])[0]
        store.upsert(args.id, vec, {"text": args.text})
    elif args.command == "delete":
        filt = dict(f.split("=",1) for f in args.filter) if args.filter else {}
        store.delete(ids=args.ids or None, filter=filt or None)
    elif args.command == "search":
        vec = embedder.encode([args.text])[0]
        results = store.search(vec, k=args.k)
        for rid, score, meta in results:
            print(f"{rid}\t{score:.4f}\t{meta.get('text','')}")
    elif args.command == "save":
        store.save(args.path)
    elif args.command == "load":
        store.load(args.path)
    else:
        parser.print_help()
        sys.exit(1)

    if args.command in ["insert", "upsert", "delete"]:
        store.save("data")

if __name__ == "__main__":
    main()
