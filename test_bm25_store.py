import pickle
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.test_bm25")

def load_config():
    with open("config.json") as f:
        return json.load(f)

def load_bm25(config):
    path = Path(config["retrieval"]["bm25_index_path"])
    assert path.exists(), f"BM25 index not found at {path}"
    with open(path, "rb") as f:
        return pickle.load(f)


def test_file_exists(config):
    logger.info("TEST 1: File exists")
    path = Path(config["retrieval"]["bm25_index_path"])
    assert path.exists(), f"Not found: {path}"
    size_kb = path.stat().st_size / 1024
    logger.info(f"  Path      : {path}")
    logger.info(f"  Size      : {size_kb:.1f} KB")
    logger.info("  PASSED\n")


def test_loads_correctly(bm25):
    logger.info("TEST 2: Loads without error")
    assert bm25 is not None
    logger.info(f"  Type : {type(bm25).__name__}")
    logger.info("  PASSED\n")


def test_doc_count(bm25):
    logger.info("TEST 3: Document count > 0")
    count = len(bm25.docs)
    assert count > 0, "No documents in BM25 index"
    logger.info(f"  Docs indexed : {count}")
    logger.info("  PASSED\n")


def test_sample_documents(bm25):
    logger.info("TEST 4: Sample documents have expected fields")
    for i in range(min(3, len(bm25.docs))):
        doc = bm25.docs[i]
        meta = doc.metadata
        assert doc.page_content, f"Doc {i} has empty page_content"
        logger.info(f"  Doc {i+1}")
        logger.info(f"    chunk_id     : {meta.get('chunk_id')}")
        logger.info(f"    doc_id       : {meta.get('doc_id')}")
        logger.info(f"    arxiv_id     : {meta.get('arxiv_id')}")
        logger.info(f"    page_number  : {meta.get('page_number')}")
        logger.info(f"    section_title: {meta.get('section_title') or '(none)'}")
        logger.info(f"    text preview : {doc.page_content[:100].strip()!r}")
    logger.info("  PASSED\n")


def test_metadata_coverage(bm25):
    logger.info("TEST 5: Metadata coverage across all docs")
    fields = ["chunk_id", "doc_id", "arxiv_id", "source_file", "page_number"]
    missing = {f: 0 for f in fields}
    for doc in bm25.docs:
        for f in fields:
            if doc.metadata.get(f) is None:
                missing[f] += 1
    for f, count in missing.items():
        status = "OK" if count == 0 else f"MISSING in {count} docs"
        logger.info(f"  {f:<20}: {status}")
    logger.info("  PASSED\n")


def test_keyword_query(bm25):
    logger.info("TEST 6: Keyword query returns results")
    results = bm25.invoke("neural network training loss")
    assert len(results) > 0, "No results returned"
    logger.info(f"  Results returned : {len(results)}")
    for i, r in enumerate(results[:3], 1):
        logger.info(f"  [{i}] chunk_id={r.metadata.get('chunk_id')}")
        logger.info(f"       preview : {r.page_content[:100].strip()!r}")
    logger.info("  PASSED\n")


def test_exact_term_query(bm25):
    logger.info("TEST 7: Exact term match — BM25 strength")
    results = bm25.invoke("RMSE")
    logger.info(f"  Query : 'RMSE'")
    logger.info(f"  Results: {len(results)}")
    for i, r in enumerate(results[:3], 1):
        hit = "RMSE" in r.page_content
        logger.info(f"  [{i}] contains 'RMSE': {hit} | preview: {r.page_content[:100].strip()!r}")
    logger.info("  PASSED\n")


def test_unique_docs_returned(bm25):
    logger.info("TEST 8: Query returns unique chunk_ids")
    results = bm25.invoke("attention mechanism transformer")
    ids = [r.metadata.get("chunk_id") for r in results]
    unique = len(set(ids))
    logger.info(f"  Total returned : {len(ids)}")
    logger.info(f"  Unique ids     : {unique}")
    assert unique == len(ids), "Duplicate chunk_ids in results"
    logger.info("  PASSED\n")


def run_all_tests():
    logger.info("=" * 55)
    logger.info("BM25 INDEX TEST SUITE")
    logger.info("=" * 55 + "\n")

    config = load_config()
    test_file_exists(config)
    bm25 = load_bm25(config)
    test_loads_correctly(bm25)
    test_doc_count(bm25)
    test_sample_documents(bm25)
    test_metadata_coverage(bm25)
    test_keyword_query(bm25)
    test_exact_term_query(bm25)
    test_unique_docs_returned(bm25)

    logger.info("=" * 55)
    logger.info("ALL TESTS COMPLETE")
    logger.info("=" * 55)

if __name__ == "__main__":
    run_all_tests()