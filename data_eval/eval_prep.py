import json
import csv
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


def prepare_eval_dataset(config: dict) -> list[dict]:
    meta_dir = Path(config["evaluation"]["meta_dir"])

    # 1. Load all three source files
    queries = json.loads((meta_dir / "queries.json").read_text())
    qrels   = json.loads((meta_dir / "qrels.json").read_text())
    answers = json.loads((meta_dir / "answers.json").read_text())

    # 2. Get doc_ids currently indexed in ChromaDB
    client     = chromadb.PersistentClient(
        path=config["vector_store"]["persist_dir"],
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(config["vector_store"]["collection_name"])
    all_meta   = collection.get(include=["metadatas"])
    indexed_docs = set(m["doc_id"] for m in all_meta["metadatas"])

    logger.info("Unique docs indexed in ChromaDB: %d", len(indexed_docs))

    # 3. Filter: only keep query UUIDs where relevant doc is in the store
    rows = []
    skipped = 0

    for uuid, qrel in qrels.items():
        relevant_doc = qrel["doc_id"]       # e.g. "2410.14077v2"
        section_id   = qrel["section_id"]

        if relevant_doc not in indexed_docs:
            skipped += 1
            continue

        # UUID is the join key across all three files
        if uuid not in queries:
            skipped += 1
            continue

        rows.append({
            "query_id":       uuid,
            "query":          queries[uuid]["query"],
            "type":           queries[uuid]["type"],        # extractive / abstractive
            "source":         queries[uuid]["source"],      # text / text-image / etc.
            "relevant_doc_id": relevant_doc,
            "section_id":     section_id,
            "answer":         answers.get(uuid, ""),
        })

    logger.info("Evaluable queries: %d | Skipped: %d", len(rows), skipped)

    # 4. Save as CSV (for inspection) and JSON (for the evaluator to load)
    out_dir = Path(config["evaluation"]["metrics_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "eval_dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "eval_dataset.json"
    json_path.write_text(json.dumps(rows, indent=2))

    logger.info("Saved eval_dataset.csv  → %s", csv_path)
    logger.info("Saved eval_dataset.json → %s", json_path)

    # 5. Print a quick breakdown so you know what you're working with
    types   = {}
    sources = {}
    for r in rows:
        types[r["type"]]     = types.get(r["type"], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1

    logger.info("Query type breakdown  : %s", types)
    logger.info("Query source breakdown: %s", sources)

    return rows

if __name__ == "__main__":
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Always find config.json relative to this file's location
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    config = json.load(open(config_path))
    rows   = prepare_eval_dataset(config)
    print(f"\nDone — {len(rows)} evaluable queries ready for evaluation")