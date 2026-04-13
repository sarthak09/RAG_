import json
import logging
import time
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings

from data_injest.models import EmbeddedChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("rag.vector_store")


class VectorStore:
    def __init__(self, config: dict):
        cfg = config["vector_store"]

        self.persist_dir     = cfg["persist_dir"]
        self.collection_name = cfg["collection_name"]
        self.log_dir         = config["ingestion"]["log_dir"]

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"VectorStore ready | collection={self.collection_name} | persist_dir={self.persist_dir} | existing_docs={self.collection.count()}")

    def is_populated(self) -> bool:
        return self.collection.count() > 0

    def add(self, chunks: list[EmbeddedChunk], batch_size: int = 100) -> None:
        if not chunks:
            logger.warning("No chunks to add")
            return

        detected = len(chunks[0].vector)
        logger.info(f"Inserting vectors | dimensions={detected} | model={chunks[0].embedding_model}")
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")

        start  = time.time()
        failed = 0

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start: batch_start + batch_size]
            try:
                self.collection.add(
                    ids=[c.chunk_id for c in batch],
                    embeddings=[c.vector for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[
                        {
                            "doc_id":              c.doc_id,
                            "source_file":         c.source_file,
                            "arxiv_id":            c.arxiv_id,
                            "page_number":         c.page_number,
                            "section_title":       c.section_title,
                            "chunk_index":         c.chunk_index,
                            "char_count":          c.char_count,
                            "total_chunks_in_doc": c.total_chunks_in_doc,
                            "is_first_chunk":      c.is_first_chunk,
                            "is_last_chunk":       c.is_last_chunk,
                            "embedding_model":     c.embedding_model,
                            "chunk_size_config":   c.chunk_size_config,
                            "overlap_config":      c.overlap_config,
                            "embedded_at":         c.embedded_at,
                        }
                        for c in batch
                    ]
                )
                logger.info(f"Batch {batch_start // batch_size + 1} inserted | size={len(batch)}")
            except Exception as e:
                failed += len(batch)
                logger.error(f"Batch at {batch_start} failed: {e}")

        elapsed = round(time.time() - start, 2)
        logger.info(f"Insert complete | added={len(chunks) - failed} | failed={failed} | total_in_store={self.collection.count()} | time={elapsed}s")
        self._save_stats("add", len(chunks), len(chunks) - failed, failed, elapsed)

    def query(self, query_vector: list[float], top_k: int = 5, where: dict = None) -> list[dict]:
        try:
            kwargs = {
                "query_embeddings": [query_vector],
                "n_results":        top_k,
                "include":          ["documents", "metadatas", "distances"]
            }
            if where:
                kwargs["where"] = where

            results = self.collection.query(**kwargs)

            return [
                {
                    "chunk_id": results["ids"][0][i],
                    "text":     results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    **results["metadatas"][0][i]
                }
                for i in range(len(results["ids"][0]))
            ]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{self.collection_name}' deleted and recreated")

    def count(self) -> int:
        return self.collection.count()

    def _save_stats(self, operation: str, total: int, succeeded: int, failed: int, elapsed: float):
        stats = {
            "timestamp":      datetime.now().isoformat(),
            "operation":      operation,
            "collection":     self.collection_name,
            "persist_dir":    self.persist_dir,
            "total":          total,
            "succeeded":      succeeded,
            "failed":         failed,
            "elapsed_seconds": elapsed,
            "total_in_store": self.collection.count()
        }
        filepath = Path(self.log_dir) / f"vector_store_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Stats saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")