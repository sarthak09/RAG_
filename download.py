import os
import json
import requests
import time
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    HUGGINGFACE_DATASET = "vectara/open_ragbench"
    ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf/"
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw" / "pdfs"
        self.metadata_dir = self.base_dir / "raw" / "metadata"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def download_metadata(self) -> None:
        files = ["pdf_urls.json", "queries.json", "qrels.json", "answers.json"]
        for filename in files:
            dest = self.metadata_dir / filename
            if dest.exists():
                logger.info(f"Already exists: {filename}")
                continue
            logger.info(f"Downloading {filename}...")
            path = hf_hub_download(
                repo_id=self.HUGGINGFACE_DATASET,
                filename=f"pdf/arxiv/{filename}",
                repo_type="dataset",
                local_dir=str(self.metadata_dir))
            logger.info(f"Saved → {path}")

    def load_pdf_urls(self) -> dict:
        path = self.metadata_dir / "pdf" / "arxiv" / "pdf_urls.json"
        with open(path) as f:
            return json.load(f)
        
    def download_pdfs(self, pdf_urls: dict, limit: int = None, delay: float = 1.5) -> dict:
        items = list(pdf_urls.items())
        if limit:
            items = items[:limit]
        downloaded = {}
        failed = []
        logger.info(f"Downloading {len(items)} PDFs...")
        for paper_id, url in tqdm(items, desc="Downloading PDFs"):
            dest = self.raw_dir / f"{paper_id}.pdf"
            if dest.exists():
                logger.debug(f"Already exists: {paper_id}")
                downloaded[paper_id] = str(dest)
                continue
            try:
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded[paper_id] = str(dest)
                time.sleep(delay)  
            except requests.RequestException as e:
                logger.warning(f"Failed to download {paper_id}: {e}")
                failed.append(paper_id)
        manifest = {
            "downloaded": downloaded,
            "failed": failed,
            "total": len(items),
            "success_count": len(downloaded),
            "fail_count": len(failed)
        }
        manifest_path = self.metadata_dir / "download_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            f"Done. {len(downloaded)} downloaded, {len(failed)} failed. "
            f"Manifest saved → {manifest_path}")
        return downloaded

if __name__ == "__main__":
    downloader = DatasetDownloader(base_dir="data")
    downloader.download_metadata()
    pdf_urls = downloader.load_pdf_urls()
    downloader.download_pdfs(pdf_urls, limit=None, delay=1.5)