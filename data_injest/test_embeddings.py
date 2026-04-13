import json
import sys
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_injest.embedder import Embedder, Chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("rag.test_embeddings")


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def test_model_loads(embedder: Embedder):
    logger.info("TEST 1: Model loads and returns correct dimensions")
    logger.info(f"  Model      : {embedder.model_name}")
    logger.info(f"  Dimensions : {embedder.dimensions}")
    assert embedder.dimensions > 0, "Dimensions should be > 0"
    logger.info("  PASSED\n")


def test_similar_texts_are_close(embedder: Embedder):
    logger.info("TEST 2: Similar texts have high cosine similarity")

    pairs = [
        ("inflation is caused by rising demand", "prices increase due to higher demand"),
        ("neural networks learn from data",      "deep learning models train on datasets"),
        ("the mitochondria powers the cell",     "mitochondria is the energy source of cells"),
    ]

    for text_a, text_b in pairs:
        vec_a = embedder.model.embed_query(text_a)
        vec_b = embedder.model.embed_query(text_b)
        score = cosine_similarity([vec_a], [vec_b])[0][0]
        status = "PASSED" if score > 0.6 else "FAILED"
        logger.info(f"  [{status}] score={score:.3f}")
        logger.info(f"    A: {text_a}")
        logger.info(f"    B: {text_b}")

    logger.info("")


def test_different_texts_are_far(embedder: Embedder):
    logger.info("TEST 3: Unrelated texts have low cosine similarity")

    pairs = [
        ("inflation is caused by rising demand", "the mitochondria powers the cell"),
        ("neural networks learn from data",      "the french revolution began in 1789"),
    ]

    for text_a, text_b in pairs:
        vec_a = embedder.model.embed_query(text_a)
        vec_b = embedder.model.embed_query(text_b)
        score = cosine_similarity([vec_a], [vec_b])[0][0]
        status = "PASSED" if score < 0.6 else "FAILED"
        logger.info(f"  [{status}] score={score:.3f}")
        logger.info(f"    A: {text_a}")
        logger.info(f"    B: {text_b}")

    logger.info("")


def test_vector_norms(embedder: Embedder):
    logger.info("TEST 4: Vector norms are healthy (no zero/near-zero vectors)")

    texts = [
        "inflation and monetary policy",
        "x",
        "",
        "deep learning with transformers for nlp tasks",
        "   ",
    ]

    for text in texts:
        try:
            vec = embedder.model.embed_query(text if text.strip() else "empty")
            norm = float(np.linalg.norm(vec))
            status = "PASSED" if norm > 0.1 else "WARN  "
            logger.info(f"  [{status}] norm={norm:.4f} | input={repr(text)}")
        except Exception as e:
            logger.warning(f"  [ERROR] input={repr(text)} | {e}")

    logger.info("")


def test_batch_vs_single(embedder: Embedder):
    logger.info("TEST 5: Batch embedding matches one-by-one embedding")

    texts = [
        "inflation rises due to demand",
        "neural networks are powerful",
        "the mitochondria is the powerhouse",
    ]

    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id="test", text=t, chunk_index=i, char_count=len(t))
        for i, t in enumerate(texts)
    ]

    single_results = embedder.embed_one_by_one(chunks)
    batch_results  = embedder.embed_batch(chunks)

    all_match = True
    for s, b in zip(single_results, batch_results):
        score = cosine_similarity([s.vector], [b.vector])[0][0]
        match = score > 0.999
        if not match:
            all_match = False
        logger.info(f"  similarity={score:.6f} | {'PASSED' if match else 'FAILED'} | {s.text[:40]}")

    logger.info(f"  Overall: {'PASSED' if all_match else 'FAILED'}\n")


def test_query_vs_relevant_chunk(embedder: Embedder):
    logger.info("TEST 6: Query vector is close to its ground-truth chunk")

    query_chunk_pairs = [
        (
            "What causes inflation?",
            "Inflation is primarily driven by excess demand in the economy, supply chain disruptions, and monetary expansion."
        ),
        (
            "How do transformers work?",
            "Transformers use self-attention mechanisms to weigh the importance of different tokens in a sequence."
        ),
    ]

    for query, chunk_text in query_chunk_pairs:
        q_vec   = embedder.model.embed_query(query)
        c_vec   = embedder.model.embed_query(chunk_text)
        score   = cosine_similarity([q_vec], [c_vec])[0][0]
        status  = "PASSED" if score > 0.5 else "FAILED"
        logger.info(f"  [{status}] score={score:.3f}")
        logger.info(f"    Query : {query}")
        logger.info(f"    Chunk : {chunk_text[:60]}...")

    logger.info("")


def run_all_tests(embedder: Embedder):
    logger.info("=" * 55)
    logger.info("EMBEDDING TEST SUITE")
    logger.info("=" * 55 + "\n")

    test_model_loads(embedder)
    test_similar_texts_are_close(embedder)
    test_different_texts_are_far(embedder)
    test_vector_norms(embedder)
    test_batch_vs_single(embedder)
    test_query_vs_relevant_chunk(embedder)

    logger.info("=" * 55)
    logger.info("ALL TESTS COMPLETE")
    logger.info("=" * 55)


if __name__ == "__main__":
    config = load_config("config.json")
    embedder = Embedder(config)
    run_all_tests(embedder)
