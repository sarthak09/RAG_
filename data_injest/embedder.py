import os
import json
import logging
import time
from datetime import datetime
from langchain_ollama import OllamaEmbeddings

from data_injest.models import Chunk, EmbeddedChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("rag.embedder")


class Embedder:
    def __init__(self, config: dict):
        cfg = config["embeddings"]

        self.model_name = cfg["model_name"]
        self.base_url = os.getenv("OLLAMA_BASE_URL", cfg["base_url"])
        self.batch_size = cfg.get("batch_size", 32)
        self.chunk_size_config = config["chunking"]["chunk_size"]
        self.overlap_config = config["chunking"]["chunk_overlap"]
        self.log_dir = config["ingestion"]["log_dir"]

        self.model = OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url
        )

        self.dimensions = self._detect_dimensions()
        logger.info(f"Embedder ready | model={self.model_name} | dimensions={self.dimensions}")

    def _detect_dimensions(self) -> int:
        try:
            vector = self.model.embed_query("test")
            logger.info(f"Auto-detected dimensions={len(vector)} for model={self.model_name}")
            return len(vector)
        except Exception as e:
            logger.error(f"Could not load model '{self.model_name}': {e}")
            raise

    def _to_embedded(self, chunk: Chunk, vector: list[float]) -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            char_count=chunk.char_count,
            vector=vector,
            source_file=chunk.source_file,
            arxiv_id=chunk.arxiv_id,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            total_chunks_in_doc=chunk.total_chunks_in_doc,
            is_first_chunk=chunk.is_first_chunk,
            is_last_chunk=chunk.is_last_chunk,
            embedding_model=self.model_name,
            chunk_size_config=self.chunk_size_config,
            overlap_config=self.overlap_config,
        )

    def embed_one_by_one(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        logger.info(f"Starting one-by-one embedding | total={len(chunks)}")
        embedded, failed, start = [], 0, time.time()

        for i, chunk in enumerate(chunks):
            try:
                vector = self.model.embed_query(chunk.text)
                embedded.append(self._to_embedded(chunk, vector))
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{len(chunks)}")
            except Exception as e:
                failed += 1
                logger.error(f"Failed chunk {chunk.chunk_id}: {e}")

        elapsed = round(time.time() - start, 2)
        logger.info(f"One-by-one done | embedded={len(embedded)} | failed={failed} | time={elapsed}s")
        self._save_stats("one_by_one", len(chunks), len(embedded), failed, elapsed)
        return embedded

    def embed_batch(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        logger.info(f"Starting batch embedding | total={len(chunks)} | batch_size={self.batch_size}")
        embedded, failed, start = [], 0, time.time()

        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start: batch_start + self.batch_size]
            try:
                vectors = self.model.embed_documents([c.text for c in batch])
                for chunk, vector in zip(batch, vectors):
                    embedded.append(self._to_embedded(chunk, vector))
                logger.info(f"Batch {batch_start // self.batch_size + 1} done | size={len(batch)}")
            except Exception as e:
                failed += len(batch)
                logger.error(f"Batch at {batch_start} failed: {e}")

        elapsed = round(time.time() - start, 2)
        logger.info(f"Batch done | embedded={len(embedded)} | failed={failed} | time={elapsed}s")
        self._save_stats("batch", len(chunks), len(embedded), failed, elapsed)
        return embedded

    def _save_stats(self, method: str, total: int, succeeded: int, failed: int, elapsed: float):
        stats = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "model": self.model_name,
            "dimensions": self.dimensions,
            "total_chunks": total,
            "succeeded": succeeded,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "chunks_per_second": round(succeeded / elapsed, 2) if elapsed > 0 else 0
        }
        filename = f"embedder_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"{self.log_dir}/{filename}"
        try:
            with open(filepath, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Stats saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
