import json
import sys
import logging
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_injest.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.test_vector_store")


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def test_collection_exists(store: VectorStore):
    logger.info("TEST 1: Collection exists and is accessible")
    count = store.count()
    assert count > 0, "Collection is empty — run the ingestion pipeline first"
    logger.info(f"  PASSED | total chunks = {count}\n")
    return count


def test_collection_info(store: VectorStore):
    logger.info("TEST 2: Collection metadata")
    logger.info(f"  Name        : {store.collection.name}")
    logger.info(f"  Total docs  : {store.count()}")
    logger.info(f"  Persist dir : {store.persist_dir}")
    logger.info(f"  Distance fn : cosine (hnsw:space)")
    logger.info("  PASSED\n")


def test_sample_records(store: VectorStore):
    logger.info("TEST 3: Sample records — inspect stored data")
    results = store.collection.get(limit=3, include=["documents", "metadatas", "embeddings"])

    for i in range(len(results["ids"])):
        meta   = results["metadatas"][i]
        text   = results["documents"][i]
        vector = results["embeddings"][i]
        logger.info(f"  Record {i + 1}")
        logger.info(f"    chunk_id       : {results['ids'][i]}")
        logger.info(f"    doc_id         : {meta.get('doc_id')}")
        logger.info(f"    source_file    : {meta.get('source_file')}")
        logger.info(f"    arxiv_id       : {meta.get('arxiv_id')}")
        logger.info(f"    page_number    : {meta.get('page_number')}")
        logger.info(f"    section_title  : {meta.get('section_title') or '(none)'}")
        logger.info(f"    chunk_index    : {meta.get('chunk_index')}")
        logger.info(f"    char_count     : {meta.get('char_count')}")
        logger.info(f"    is_first_chunk : {meta.get('is_first_chunk')}")
        logger.info(f"    is_last_chunk  : {meta.get('is_last_chunk')}")
        logger.info(f"    embedding_model: {meta.get('embedding_model')}")
        logger.info(f"    chunk_size_cfg : {meta.get('chunk_size_config')}")
        logger.info(f"    overlap_cfg    : {meta.get('overlap_config')}")
        logger.info(f"    embedded_at    : {meta.get('embedded_at')}")
        logger.info(f"    vector dims    : {len(vector)}")
        logger.info(f"    vector preview : {[round(v, 4) for v in vector[:5]]}...")
        logger.info(f"    text preview   : {text[:120].strip()!r}")
        logger.info("")
    logger.info("  PASSED\n")


def test_vector_stats(store: VectorStore):
    logger.info("TEST 4: Vector statistics")
    total   = store.count()
    batch   = min(total, 500)
    results = store.collection.get(limit=batch, include=["embeddings"])
    vectors = np.array(results["embeddings"])

    norms     = np.linalg.norm(vectors, axis=1)
    zero_vecs = int(np.sum(norms < 0.01))

    logger.info(f"  Sampled       : {batch} / {total} vectors")
    logger.info(f"  Dimensions    : {vectors.shape[1]}")
    logger.info(f"  Norm  min     : {norms.min():.4f}")
    logger.info(f"  Norm  max     : {norms.max():.4f}")
    logger.info(f"  Norm  mean    : {norms.mean():.4f}")
    logger.info(f"  Zero vectors  : {zero_vecs}  {'WARNING' if zero_vecs > 0 else '(clean)'}")
    logger.info(f"  {'WARN — zero vectors found' if zero_vecs > 0 else 'PASSED'}\n")


def test_metadata_coverage(store: VectorStore):
    logger.info("TEST 5: Metadata coverage — check for missing values")
    total   = store.count()
    batch   = min(total, 500)
    results = store.collection.get(limit=batch, include=["metadatas"])

    fields  = ["doc_id", "source_file", "arxiv_id", "embedding_model", "chunk_size_config", "page_number"]
    missing = {f: 0 for f in fields}

    for meta in results["metadatas"]:
        for f in fields:
            if meta.get(f) is None:
                missing[f] += 1

    for f, count in missing.items():
        status = "OK" if count == 0 else f"MISSING in {count} records"
        logger.info(f"  {f:<22}: {status}")
    logger.info("  PASSED\n")


def test_unique_documents(store: VectorStore):
    logger.info("TEST 6: Unique documents in store")
    total   = store.count()
    results = store.collection.get(limit=total, include=["metadatas"])

    doc_ids     = [m["doc_id"] for m in results["metadatas"]]
    chunks_per  = Counter(doc_ids)
    unique_docs = len(chunks_per)

    logger.info(f"  Unique documents : {unique_docs}")
    logger.info(f"  Total chunks     : {total}")
    logger.info(f"  Avg chunks/doc   : {total / max(unique_docs, 1):.1f}")
    logger.info(f"  Max chunks/doc   : {max(chunks_per.values())}")
    logger.info(f"  Min chunks/doc   : {min(chunks_per.values())}")
    logger.info("  Top 5 documents by chunk count:")
    for doc_id, count in chunks_per.most_common(5):
        logger.info(f"    {doc_id:<40} {count} chunks")
    logger.info("  PASSED\n")


def test_page_numbers(store: VectorStore):
    logger.info("TEST 7: Page numbers are stored and span multiple pages")
    total   = store.count()
    results = store.collection.get(limit=total, include=["metadatas"])

    page_numbers = [m.get("page_number") for m in results["metadatas"]]
    none_count   = sum(1 for p in page_numbers if p is None)
    zero_count   = sum(1 for p in page_numbers if p == 0)
    unique_pages = sorted(set(p for p in page_numbers if p is not None))

    logger.info(f"  Total chunks         : {total}")
    logger.info(f"  Missing page_number  : {none_count}")
    logger.info(f"  Chunks on page 0     : {zero_count}")
    logger.info(f"  Unique page numbers  : {len(unique_pages)}")
    logger.info(f"  Page range           : {min(unique_pages)} → {max(unique_pages)}")

    by_doc: dict = {}
    for m in results["metadatas"]:
        doc = m.get("doc_id", "unknown")
        page = m.get("page_number", 0)
        by_doc.setdefault(doc, set()).add(page)

    logger.info("  Pages per document (sample of 5):")
    for doc_id, pages in list(by_doc.items())[:5]:
        logger.info(f"    {doc_id:<40} pages={sorted(pages)}")

    status = "PASSED" if none_count == 0 else f"WARN — {none_count} chunks missing page_number"
    logger.info(f"  {status}\n")


def test_query_returns_results(store: VectorStore):
    logger.info("TEST 8: Query returns results")
    sample       = store.collection.get(limit=1, include=["embeddings"])
    query_vector = sample["embeddings"][0]

    hits = store.query(query_vector, top_k=5)
    assert len(hits) > 0, "Query returned no results"

    logger.info(f"  Query returned {len(hits)} hits")
    for i, hit in enumerate(hits):
        logger.info(f"  Hit {i + 1}")
        logger.info(f"    chunk_id    : {hit['chunk_id']}")
        logger.info(f"    doc_id      : {hit['doc_id']}")
        logger.info(f"    page_number : {hit.get('page_number')}")
        logger.info(f"    distance    : {hit['distance']:.6f}")
        logger.info(f"    source_file : {hit['source_file']}")
        logger.info(f"    section     : {hit['section_title'] or '(none)'}")
        logger.info(f"    text        : {hit['text'][:100].strip()!r}")
    logger.info("  PASSED\n")


def test_metadata_filter(store: VectorStore):
    logger.info("TEST 9: Metadata filtering by doc_id")
    sample       = store.collection.get(limit=1, include=["metadatas", "embeddings"])
    meta         = sample["metadatas"][0]
    query_vector = sample["embeddings"][0]
    target_doc   = meta["doc_id"]

    hits      = store.query(query_vector, top_k=5, where={"doc_id": {"$eq": target_doc}})
    all_match = all(h["doc_id"] == target_doc for h in hits)

    logger.info(f"  Filtered to doc_id='{target_doc}'")
    logger.info(f"  Results returned : {len(hits)}")
    logger.info(f"  All match filter : {all_match}")
    logger.info(f"  {'PASSED' if all_match else 'FAILED — results contain wrong doc_ids'}\n")


def test_page_filter(store: VectorStore):
    logger.info("TEST 10: Metadata filtering by page_number")
    total   = store.count()
    results = store.collection.get(limit=total, include=["metadatas"])

    all_pages   = [m.get("page_number", 0) for m in results["metadatas"]]
    target_page = max(set(all_pages), key=all_pages.count)

    sample       = store.collection.get(limit=1, include=["embeddings"])
    query_vector = sample["embeddings"][0]

    hits      = store.query(query_vector, top_k=5, where={"page_number": {"$eq": target_page}})
    all_match = all(h.get("page_number") == target_page for h in hits)

    logger.info(f"  Filtered to page_number={target_page}")
    logger.info(f"  Results returned : {len(hits)}")
    logger.info(f"  All match filter : {all_match}")
    logger.info(f"  {'PASSED' if all_match else 'FAILED — results contain wrong page numbers'}\n")


def test_char_count_filter(store: VectorStore):
    logger.info("TEST 11: Filter by char_count >= 100")
    sample       = store.collection.get(limit=1, include=["embeddings"])
    query_vector = sample["embeddings"][0]

    hits      = store.query(query_vector, top_k=5, where={"char_count": {"$gte": 100}})
    all_large = all(h["char_count"] >= 100 for h in hits)

    logger.info(f"  Results returned : {len(hits)}")
    logger.info(f"  All >= 100 chars : {all_large}")
    logger.info(f"  {'PASSED' if all_large else 'FAILED'}\n")


def test_persist_directory(store: VectorStore):
    logger.info("TEST 12: Persist directory exists and has data")
    p = Path(store.persist_dir)
    assert p.exists(), f"Persist dir not found: {p}"

    files      = list(p.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    logger.info(f"  Path       : {p.resolve()}")
    logger.info(f"  Files      : {len(files)}")
    logger.info(f"  Total size : {total_size / 1024 / 1024:.2f} MB")
    for f in files:
        if f.is_file():
            logger.info(f"    {f.relative_to(p)}  ({f.stat().st_size / 1024:.1f} KB)")
    logger.info("  PASSED\n")


def run_all_tests(store: VectorStore):
    logger.info("=" * 55)
    logger.info("VECTOR STORE TEST SUITE")
    logger.info("=" * 55 + "\n")

    test_collection_exists(store)
    test_collection_info(store)
    test_sample_records(store)
    test_vector_stats(store)
    test_metadata_coverage(store)
    test_unique_documents(store)
    test_page_numbers(store)
    test_query_returns_results(store)
    test_metadata_filter(store)
    test_page_filter(store)
    test_char_count_filter(store)
    test_persist_directory(store)

    logger.info("=" * 55)
    logger.info("ALL TESTS COMPLETE")
    logger.info("=" * 55)


if __name__ == "__main__":
    config = load_config("config.json")
    store  = VectorStore(config)
    run_all_tests(store)