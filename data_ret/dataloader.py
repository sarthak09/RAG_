from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

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