import os
import sys
import json
import logging
from dotenv import load_dotenv
from data_ret.dataloader import VectorStoreLoader, BM25Loader
from agentic_ai.agent_workflow import AgenticRAGWorkflow
from data_ret.models import RAGResponse, FailedRAGResponse
from pathlib import Path
from agentic_ai.agent_workflow2 import CorrectiveRAGWorkflow

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.main_agent")

def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)

def print_store_info(store, config: dict) -> None:
    vs_cfg = config["vector_store"]
    emb_cfg = config["embeddings"]
    count = store._collection.count()
    print("\n" + "=" * 70)
    print("  VECTOR STORE")
    print(f"    persist_dir     : {vs_cfg['persist_dir']}")
    print(f"    collection      : {vs_cfg['collection_name']}")
    print(f"    embedding_model : {emb_cfg['model_name']}")
    print(f"    total_chunks    : {count:,}")
    print("=" * 70 + "\n")

def print_response(response: RAGResponse | FailedRAGResponse, latency: dict) -> None:
    print("\n" + "=" * 70)
    if isinstance(response, FailedRAGResponse):
        print(f"  ERROR: {response.error}")
        print("=" * 70)
        return
    print(f"  QUERY   : {response.query}")
    print(f"  MODEL   : {response.model}")
    print(f"  CHUNKS  : {response.total_chunks_used}")
    print("=" * 70)
    print(f"\n  ANSWER:\n  {response.answer}\n")
    if response.citations:
        print("  CITATIONS:")
        for i, c in enumerate(response.citations, 1):
            print(f"    [{i}] chunk_id     : {c.chunk_id}")
            print(f"        doc_id       : {c.doc_id}")
            if c.arxiv_id:
                print(f"        arxiv_id     : {c.arxiv_id}")
            if c.source_file:
                print(f"        source_file  : {c.source_file}")
            if c.page_number:
                print(f"        page         : {c.page_number}")
            if c.section_title:
                print(f"        section      : {c.section_title}")
            print(f"        snippet      : {c.snippet[:120]}...")
            print()
    if latency:
        print("  LATENCY:")
        print(f"    prep_llm  : {latency.get('prep_llm_ms', 0):.1f} ms  (HyDE + rewriting in parallel)")
        print(f"    retrieval : {latency.get('retrieval_ms', 0):.1f} ms  (all queries in parallel)")
        print(f"    reranking : {latency.get('reranking_ms', 0):.1f} ms")
        print(f"    generation: {latency.get('generation_ms', 0):.1f} ms")
        print(f"    ─────────────────────────────────")
        print(f"    total     : {latency.get('total_ms', 0):.1f} ms")
        print(f"    hyde      : {'yes' if latency.get('hyde_used') else 'no'}")
        print(f"    rewrites  : {latency.get('rewrites_used', 0)}")
        print(f"    merged    : {latency.get('chunks_merged', 0)} → final {latency.get('chunks_final', 0)}")
    print("=" * 70 + "\n")

def main():
    use_crag = "--crag" in sys.argv
    clean_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    config  = load_config()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", config["llm"]["base_url"])
    embed_model = config["embeddings"]["model_name"]
    store = VectorStoreLoader(config, ollama_base_url, embed_model).load()
    print_store_info(store, config)
    bm25_retriever = None
    if config["retrieval"].get("use_hybrid", False):
        try:
            bm25_retriever = BM25Loader(config).load()
            logger.info(f"BM25 index loaded | path={config['retrieval']['bm25_index_path']}")
        except Exception as e:
            logger.warning(f"BM25 load failed, running without it: {e}")
    if use_crag:
        workflow = CorrectiveRAGWorkflow(config=config, ollama_base_url=ollama_base_url,store=store, bm25_retriever=bm25_retriever,)
    else:
        workflow = AgenticRAGWorkflow(config=config, ollama_base_url=ollama_base_url,store=store, bm25_retriever=bm25_retriever,)
    print("\n  AGENT GRAPH:")
    graph_name = "crag_graph.png" if use_crag else "agent_graph.png"
    graph_path = Path(f"logs/{graph_name}")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_bytes(workflow.graph.get_graph(xray=True).draw_mermaid_png())
    print(f"  Agent graph saved → {graph_path}")
    query = " ".join(clean_args) if clean_args else None
    if not query:
        query = input("Enter query: ").strip()
    response, latency = workflow.run(query)
    print_response(response, latency)
 
if __name__ == "__main__":
    main()
