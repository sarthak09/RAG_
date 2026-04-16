import json
import logging
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_injest.loader import PDFLoader
from data_injest.splitter import TextSplitter
from data_injest.embedder import Embedder
from data_injest.vector_store import VectorStore

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"Data_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.FileHandler(log_file),logging.StreamHandler(sys.stdout)])
    return logging.getLogger("rag.main")

def load_config(config_path: str = "config.json") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return json.load(f)

def run_ingestion(config: dict) -> list:
    return PDFLoader(config).load_all()

def run_splitting(config: dict, documents: list) -> list:
    return TextSplitter(config).split_documents(documents)

def run_embedding(config: dict, chunks: list) -> tuple:
    embedder = Embedder(config)
    return embedder, embedder.embed_batch(chunks)

def run_vector_store(logger, config: dict, embedded_chunks: list) -> VectorStore:
    store = VectorStore(config)
    if store.is_populated():
        logger.info(f"Vector store already populated | chunks={store.count()} — skipping insert")
        return store
    store.add(embedded_chunks)
    return store

def main():
    config = load_config("config.json")
    logger = setup_logging(config["ingestion"]["log_dir"])

    logger.info(f"RAG data ingestion pipeline starting")
    logger.info(f"Setup:")
    logger.info(f"Max PDFs       : {config['ingestion']['max_pdfs'] or 'all'}")
    logger.info(f"Embedding model: {config['embeddings']['model_name']}")
    logger.info(f"Batch size     : {config['embeddings']['batch_size']}")
    logger.info(f"Strategy       : {config['chunking']['strategy']}")
    logger.info(f"Chunk size     : {config['chunking']['chunk_size']}")
    logger.info(f"Chunk overlap  : {config['chunking']['chunk_overlap']}")
    logger.info(f"Vector store   : {config['vector_store']['persist_dir']}")
    logger.info(f"Collection     : {config['vector_store']['collection_name']}")
    logger.info(f"Batch size     : {config['vector_store']['batch_size']}")

    logger.info(f"── Stage 1: Ingestion ──")
    documents = run_ingestion(config)
    logger.info(f"Ingestion done : {len(documents)} documents")

    logger.info("── Stage 2: Splitting ──")
    chunks = run_splitting(config, documents)
    logger.info(f"Splitting done : {len(chunks)} chunks")

    logger.info("── Stage 3: Embedding ──")
    embedder, embedded_chunks = run_embedding(config, chunks)
    logger.info(f"Embedding done : {len(embedded_chunks)}")
    
    logger.info("── Stage 4: Vector Store ──")
    store = run_vector_store(logger, config, embedded_chunks)
    logger.info(f"Vector store done : {store.count()}")

    logger.info("Pipeline complete")

    logger.info(f"Summary:")
    logger.info(f"  Documents    : {len(documents)}")
    logger.info(f"  Chunks       : {len(chunks)}")
    logger.info(f"  Embedded     : {len(embedded_chunks)}")
    logger.info(f"  Vector store : {store.count()}")

if __name__ == "__main__":
    main()
