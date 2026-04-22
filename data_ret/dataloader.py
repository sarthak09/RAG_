import pickle
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

class VectorStoreLoader:
    def __init__(self, config: dict, ollama_base_url: str, embed_model: str):
        self.persist_dir = config["vector_store"]["persist_dir"]
        self.collection_name = config["vector_store"]["collection_name"]
        self.embed_model = embed_model
        self.ollama_base_url = ollama_base_url
        self.store = None

    def load(self) -> Chroma:
        try:
            embeddings = OllamaEmbeddings(model=self.embed_model, base_url=self.ollama_base_url)
            self.store = Chroma(collection_name=self.collection_name, persist_directory=self.persist_dir, embedding_function=embeddings)
            count = self.store._collection.count()
            return self.store
        except Exception as e:
            raise

class BM25Loader:
    def __init__(self, config: dict):
        self.bm25_index_path = Path(config["retrieval"]["bm25_index_path"])

    def load(self) -> BM25Retriever:
        if not self.bm25_index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.bm25_index_path}. "
                "Run ingestion with use_hybrid: true to build it.")
        try:
            with open(self.bm25_index_path, "rb") as f:
                bm25 = pickle.load(f)
            return bm25
        except Exception as e:
            raise RuntimeError(f"Failed to load BM25 index: {e}") from e