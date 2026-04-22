import pickle
import json
from pathlib import Path

def load_config():
    with open("config.json") as f:
        return json.load(f)

def inspect_bm25(config: dict):
    path = Path(config["retrieval"]["bm25_index_path"])
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "rb") as f:
        bm25 = pickle.load(f)

    print(f"Type          : {type(bm25)}")
    print(f"Docs indexed  : {len(bm25.docs)}")
    print(f"Top-k default : {bm25.k}")
    print(f"File size     : {path.stat().st_size / 1024:.1f} KB")

    print("\nSample doc 1:")
    doc = bm25.docs[0]
    print(f"  page_content : {doc.page_content[:150].strip()!r}")
    print(f"  metadata     : {doc.metadata}")

    print("\nSample doc 2:")
    doc = bm25.docs[1]
    print(f"  page_content : {doc.page_content[:150].strip()!r}")
    print(f"  metadata     : {doc.metadata}")

    print("\nTest query: 'RMSE error analysis'")
    results = bm25.invoke("RMSE error analysis")
    print(f"  Results returned: {len(results)}")
    for i, r in enumerate(results[:3], 1):
        print(f"  [{i}] chunk_id={r.metadata.get('chunk_id')}  preview={r.page_content[:80].strip()!r}")

if __name__ == "__main__":
    config = load_config()
    inspect_bm25(config)