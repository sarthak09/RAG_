import os
import json
import logging
from dotenv import load_dotenv

from data_ret.dataloader import VectorStoreLoader
from data_ret.retriever import SimpleRetriever
from data_ret.llm import LLMGenerator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = json.load(f)
    logger.info("Config loaded from %s", path)
    return config


def get_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required env variable: {key}")
    return value


def main():
    logger.info("Starting RAG pipeline")

    config          = load_config()
    ollama_base_url = get_env("OLLAMA_BASE_URL")
    embed_model     = config["embeddings"]["model_name"]

    store = VectorStoreLoader(
        config=config,
        ollama_base_url=ollama_base_url,
        embed_model=embed_model
    ).load()

    retriever = SimpleRetriever(config=config, store=store)
    generator = LLMGenerator(config=config, ollama_base_url=ollama_base_url)

    query   = "How do SPX smiles vary with different maturities in financial modeling?"
    chunks  = retriever.retrieve(query)
    result  = generator.generate(query, chunks)

    chunks = retriever.retrieve(query)
    print(f"\nDEBUG — first chunk metadata: {chunks[0]}")
    
    print("\n" + "=" * 60)
    print(f"Query   : {result.query}")
    print(f"Success : {result.success}")
    print(f"Answer  : {result.answer}")
    print(f"Model   : {result.model}")

    print(f"\nDEBUG — citations count: {len(result.citations)}")
    if hasattr(result, 'citations'):
        for c in result.citations:
            print(f"  citation chunk_id: {c.chunk_id}")

    print(f"\nDEBUG — retrieved chunk_ids from retriever:")
    for c in chunks:
        print(f"  chunk_id: {c['chunk_id']}")

    if result.success and result.citations:
        print(f"\nCitations ({len(result.citations)}):")
        for i, c in enumerate(result.citations, 1):
            print(f"  {i}. doc_id={c.doc_id} | chunk={c.chunk_id}")
            print(f"     snippet: {c.snippet[:120]}")

    if not result.success:
        print(f"Error   : {result.error}")

    print("=" * 60)


if __name__ == "__main__":
    main()