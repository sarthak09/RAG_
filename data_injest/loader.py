import logging
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from langchain_community.document_loaders import PyMuPDFLoader


def setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("rag.loader")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
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
class LoaderStats:
    total_pdfs: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_chars: int = 0
    total_time_seconds: float = 0.0
    failed_files: list = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""

    def success_rate(self) -> float:
        if self.total_pdfs == 0:
            return 0.0
        return round((self.successful / self.total_pdfs) * 100, 2)

    def to_dict(self) -> dict:
        return {
            "total_pdfs": self.total_pdfs,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate_pct": self.success_rate(),
            "total_chars_extracted": self.total_chars,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_pdf_seconds": round(
                self.total_time_seconds / max(self.successful, 1), 2
            ),
            "failed_files": self.failed_files,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class ParsedDocument:
    doc_id: str
    text: str
    source_path: str
    char_count: int = 0
    parse_time_seconds: float = 0.0
    page_count: int = 0


class PDFLoader:
    def __init__(self, config: dict):
        cfg = config["ingestion"]

        self.pdf_dir   = Path(cfg["pdf_dir"])
        self.log_dir   = cfg["log_dir"]
        self.stats_dir = Path(cfg["stats_dir"])
        self.limit     = cfg["max_pdfs"]

        self.logger = setup_logger(self.log_dir)
        self.stats  = LoaderStats()

        self.stats_dir.mkdir(parents=True, exist_ok=True)

        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")

    def _parse_pdf(self, pdf_path: Path) -> Optional[ParsedDocument]:
        self.logger.debug(f"Parsing: {pdf_path.name}")
        start = time.time()

        try:
            pages   = PyMuPDFLoader(str(pdf_path)).load()
            elapsed = time.time() - start

            if not pages:
                self.logger.warning(f"Empty output: {pdf_path.name} — skipping")
                self.stats.skipped += 1
                return None

            text = "\n\n".join(p.page_content for p in pages if p.page_content.strip())

            if not text.strip():
                self.logger.warning(f"No text extracted: {pdf_path.name} — skipping")
                self.stats.skipped += 1
                return None

            self.logger.debug(f"{pdf_path.name} | {len(text):,} chars | {len(pages)} pages | {elapsed:.2f}s")

            return ParsedDocument(
                doc_id=pdf_path.stem,
                text=text,
                source_path=str(pdf_path),
                char_count=len(text),
                parse_time_seconds=round(elapsed, 3),
                page_count=len(pages),
            )

        except Exception as e:
            elapsed = time.time() - start
            self.logger.error(f"Failed: {pdf_path.name} | {e} ({elapsed:.2f}s)")
            self.stats.failed_files.append({"file": pdf_path.name, "error": str(e)})
            return None

    def load_all(self) -> list[ParsedDocument]:
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            self.logger.warning(f"No PDFs found in {self.pdf_dir}")
            return []

        if self.limit:
            pdf_files = pdf_files[: self.limit]
            self.logger.info(f"Limit active — processing {self.limit} PDFs")

        self.stats.total_pdfs = len(pdf_files)
        self.stats.start_time = datetime.now().isoformat()
        run_start             = time.time()

        self.logger.info(f"Starting ingestion | {len(pdf_files)} PDFs | {self.pdf_dir}")

        documents = []

        for i, pdf_path in enumerate(pdf_files, start=1):
            self.logger.info(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
            doc = self._parse_pdf(pdf_path)
            if doc:
                documents.append(doc)
                self.stats.successful  += 1
                self.stats.total_chars += doc.char_count
            else:
                self.stats.failed += 1

        self.stats.total_time_seconds = time.time() - run_start
        self.stats.end_time           = datetime.now().isoformat()

        self._log_summary()
        self._save_stats()

        return documents

    def _log_summary(self):
        s = self.stats
        self.logger.info("=" * 55)
        self.logger.info("INGESTION COMPLETE")
        self.logger.info(f"  Total    : {s.total_pdfs}")
        self.logger.info(f"  Success  : {s.successful}  ({s.success_rate()}%)")
        self.logger.info(f"  Failed   : {s.failed}")
        self.logger.info(f"  Skipped  : {s.skipped}")
        self.logger.info(f"  Chars    : {s.total_chars:,}")
        self.logger.info(f"  Time     : {s.total_time_seconds:.1f}s")
        self.logger.info(f"  Avg/PDF  : {s.total_time_seconds / max(s.successful, 1):.2f}s")
        self.logger.info("=" * 55)
        if s.failed_files:
            self.logger.warning(f"Failed: {[f['file'] for f in s.failed_files]}")

    def _save_stats(self):
        stats_file = self.stats_dir / f"loader_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        self.logger.info(f"Stats → {stats_file}")