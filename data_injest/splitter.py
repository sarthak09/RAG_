import logging
import json
import time
import httpx
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from data_injest.loader import ParsedDocument
from data_injest.models import Chunk


class OllamaEmbeddingsWithCtx(Embeddings):
    def __init__(self, model: str, base_url: str, num_ctx: int):
        self.model = model
        self.num_ctx = num_ctx
        self.url = f"{base_url.rstrip('/')}/api/embed"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            self.url,
            json={
                "model": self.model,
                "input": texts,
                "options": {"num_ctx": self.num_ctx}
            },
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"Data_splitter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("rag.splitter")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


@dataclass
class SplitterStats:
    strategy: str = ""
    total_documents: int = 0
    total_chunks: int = 0
    total_time_seconds: float = 0.0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    start_time: str = ""
    end_time: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "avg_chunks_per_doc": round(self.total_chunks / max(self.total_documents, 1), 1),
            "avg_chunk_size_chars": round(self.avg_chunk_size, 1),
            "min_chunk_size_chars": self.min_chunk_size,
            "max_chunk_size_chars": self.max_chunk_size,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


def _extract_section_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## ") or line.startswith("### "):
            return line.lstrip("#").strip()
        if line.startswith("**") and line.endswith("**") and len(line) < 100:
            return line.strip("*").strip()
    return ""


class TextSplitter:
    def __init__(self, config: dict):
        cfg = config["chunking"]
        self.strategy      = cfg.get("strategy", "recursive")
        self.chunk_size    = cfg["chunk_size"]
        self.chunk_overlap = cfg["chunk_overlap"]
        self.log_dir       = config["ingestion"]["log_dir"]
        self.stats_dir     = Path(config["ingestion"]["stats_dir"])
        self.logger = setup_logger(self.log_dir)
        self.stats  = SplitterStats()

        if self.strategy == "semantic":
            embed_cfg = config["embeddings"]
            embeddings = OllamaEmbeddingsWithCtx(
                model=embed_cfg["model_name"],
                base_url=embed_cfg["base_url"],
                num_ctx=embed_cfg.get("num_ctx", 8192)
            )
            self.splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=cfg.get("breakpoint_threshold_type", "percentile"),
                breakpoint_threshold_amount=cfg.get("breakpoint_threshold_amount", 95)
            )
            self.pre_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=0,
                separators=["\n\n", "\n", " "]
            )
            self.logger.info(f"TextSplitter strategy=semantic | threshold_type={cfg.get('breakpoint_threshold_type')} | amount={cfg.get('breakpoint_threshold_amount')}")
        else:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n## ", "\n### ", "\n**", "\n\n", "\n", " "],
                length_function=len
            )
            self.pre_splitter = None
            self.logger.info(f"TextSplitter strategy=recursive | chunk_size={self.chunk_size} | chunk_overlap={self.chunk_overlap}")

    def _split_document(self, doc: ParsedDocument) -> list[Chunk]:
        all_chunks = []
        chunk_index = 0
        self.logger.info(f"Data splitting of {doc.doc_id}")
        for page_number, page_text in doc.pages:
            segments = self.pre_splitter.split_text(page_text) if self.pre_splitter else [page_text]
            for segment in segments:
                raw_chunks = self.splitter.split_text(segment)
                for text in raw_chunks:
                    text = text.strip()
                    if not text:
                        continue
                    all_chunks.append((chunk_index, page_number, text))
                    chunk_index += 1
        total = len(all_chunks)
        chunks = []
        for i, (idx, page_number, text) in enumerate(all_chunks):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{idx}",
                doc_id=doc.doc_id,
                text=text,
                chunk_index=idx,
                char_count=len(text),
                source_file=Path(doc.source_path).name,
                arxiv_id=doc.doc_id,
                page_number=page_number,
                section_title=_extract_section_title(text),
                total_chunks_in_doc=total,
                is_first_chunk=(i == 0),
                is_last_chunk=(i == total - 1),
            ))
        return chunks

    def split_documents(self, documents: list[ParsedDocument]) -> list[Chunk]:
        if not documents:
            self.logger.warning("No documents to split")
            return []
        self.stats.total_documents = len(documents)
        self.stats.strategy = self.strategy
        self.stats.start_time = datetime.now().isoformat()
        run_start = time.time()
        self.logger.info(f"Starting data splitting for {len(documents)} documents")
        all_chunks = []
        for i, doc in enumerate(documents, start=1):
            self.logger.info(f"[{i}/{len(documents)}] Splitting: {doc.doc_id}")
            all_chunks.extend(self._split_document(doc))
        self.stats.total_time_seconds = time.time() - run_start
        self.stats.end_time = datetime.now().isoformat()
        self.stats.total_chunks = len(all_chunks)
        if all_chunks:
            sizes = [c.char_count for c in all_chunks]
            self.stats.avg_chunk_size = sum(sizes) / len(sizes)
            self.stats.min_chunk_size = min(sizes)
            self.stats.max_chunk_size = max(sizes)
        self._log_summary()
        self._save_stats()
        return all_chunks

    def _log_summary(self):
        s = self.stats
        self.logger.info("Data splitting Summary:")
        self.logger.info(f"  Strategy     : {s.strategy}")
        self.logger.info(f"  Documents    : {s.total_documents}")
        self.logger.info(f"  Total chunks : {s.total_chunks}")
        self.logger.info(f"  Avg/doc      : {s.total_chunks / max(s.total_documents, 1):.1f}")
        self.logger.info(f"  Avg size     : {s.avg_chunk_size:.0f} chars")
        self.logger.info(f"  Min size     : {s.min_chunk_size} chars")
        self.logger.info(f"  Max size     : {s.max_chunk_size} chars")
        self.logger.info(f"  Time         : {s.total_time_seconds:.1f}s")

    def _save_stats(self):
        stats_file = self.stats_dir / f"Data_splitter_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        self.logger.info(f"Stats: {stats_file}")