import os
import sys
import json
import re
import time
import statistics
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ret.dataloader import VectorStoreLoader, BM25Loader
from data_ret.retriever import SimpleRetriever, HybridRetriever
from data_ret.prompts import build_system_prompt, build_user_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.benchmark")

DEFAULT_QUERIES = [
    "What is RAG?",
    "Why is the concept of matching classes important in category theory?",
    "How does the MLMM approach affect the analysis of Root Mean Squared Error (RMSE)?",
    "What is unambiguously efficient allocation? What challenges do large position holders face when executing trades in financial markets?",
    "How do SPX smiles vary with different maturities in financial modeling?",
]


def load_config(path: str = "config.json") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config.json not found at {path}")
    with open(p) as f:
        return json.load(f)


def get_strategy(config: dict) -> str:
    use_hybrid = config["retrieval"].get("use_hybrid", False)
    use_rerank = config["retrieval"].get("use_rerank", False)
    if use_hybrid and use_rerank:
        return "hybrid+rerank"
    if use_hybrid:
        return "hybrid"
    return "dense"


def build_retriever(config: dict, ollama_base_url: str):
    store = VectorStoreLoader(
        config=config,
        ollama_base_url=ollama_base_url,
        embed_model=config["embeddings"]["model_name"]
    ).load()
    if config["retrieval"].get("use_hybrid", False):
        bm25 = BM25Loader(config).load()
        return HybridRetriever(config=config, store=store, bm25_retriever=bm25)
    return SimpleRetriever(config=config, store=store)


def build_llm(config: dict, ollama_base_url: str) -> ChatOllama:
    cfg = config["llm"]
    return ChatOllama(
        model=cfg["model_name"],
        base_url=ollama_base_url,
        temperature=cfg["temperature"],
        num_predict=cfg["max_tokens"],
        num_ctx=cfg.get("num_ctx", 2048),
        reasoning=False,
        format="json",
        keep_alive="60m",
    )


@dataclass
class RunResult:
    retrieval_ms:     float = 0.0
    ttft_ms:          float = 0.0
    generation_ms:    float = 0.0
    total_ms:         float = 0.0
    chunks_retrieved: int   = 0
    chunks_to_llm:    int   = 0
    prompt_chars:     int   = 0
    prompt_tokens:    int   = 0
    output_tokens:    int   = 0
    tokens_per_sec:   float = 0.0
    avg_chunk_len:    float = 0.0
    success:          bool  = True
    error:            str   = ""


def run_once(query: str, retriever, llm: ChatOllama) -> RunResult:
    r = RunResult()
    wall_start = time.perf_counter()

    try:
        t0 = time.perf_counter()
        chunks, _ = retriever.retrieve(query)
        r.retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
        r.chunks_retrieved = len(chunks)
    except Exception as e:
        r.success  = False
        r.error    = f"Retrieval failed: {e}"
        r.total_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        return r

    if not chunks:
        r.success  = False
        r.error    = "No chunks retrieved"
        r.total_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        return r

    r.chunks_to_llm = len(chunks)
    system_prompt   = build_system_prompt()
    user_prompt     = build_user_prompt(query, chunks)
    r.prompt_chars  = len(system_prompt) + len(user_prompt)
    r.prompt_tokens = r.prompt_chars // 4
    r.avg_chunk_len = round(sum(len(c["text"]) for c in chunks) / len(chunks), 1)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    raw_response  = ""
    ttft_recorded = False
    t_llm_start   = time.perf_counter()

    try:
        for chunk in llm.stream(messages):
            token = chunk.content
            if token and not ttft_recorded:
                r.ttft_ms     = round((time.perf_counter() - t_llm_start) * 1000, 2)
                ttft_recorded = True
            raw_response += token
    except Exception as e:
        r.success  = False
        r.error    = f"LLM generation failed: {e}"
        r.total_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        return r

    r.generation_ms  = round((time.perf_counter() - t_llm_start) * 1000, 2)
    r.output_tokens  = len(raw_response.split())
    r.tokens_per_sec = round(r.output_tokens / (r.generation_ms / 1000), 2) if r.generation_ms > 0 else 0.0
    r.total_ms       = round((time.perf_counter() - wall_start) * 1000, 2)
    return r


def summarise(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "mean":  round(statistics.mean(s), 2),
        "min":   round(s[0], 2),
        "max":   round(s[-1], 2),
        "p50":   round(s[int(n * 0.50)], 2),
        "p95":   round(s[min(int(n * 0.95), n - 1)], 2),
        "p99":   round(s[min(int(n * 0.99), n - 1)], 2),
        "stdev": round(statistics.stdev(s), 2) if n > 1 else 0.0,
    }


def benchmark_query(query: str, retriever, llm: ChatOllama, runs: int) -> dict:
    logger.info(f"  [warmup]")
    run_once(query, retriever, llm)

    results = []
    for i in range(runs):
        r = run_once(query, retriever, llm)
        results.append(r)
        status = "OK" if r.success else f"FAIL: {r.error}"
        logger.info(
            f"  [run {i+1}/{runs}]  "
            f"total={r.total_ms:.0f}ms  "
            f"retrieval={r.retrieval_ms:.0f}ms  "
            f"ttft={r.ttft_ms:.0f}ms  "
            f"gen={r.generation_ms:.0f}ms  "
            f"tok/s={r.tokens_per_sec:.1f}  "
            f"chunks={r.chunks_to_llm}  [{status}]"
        )

    good = [r for r in results if r.success]
    if not good:
        return {"error": "all runs failed"}

    return {
        "successful_runs": len(good),
        "raw_runs": [
            {
                "retrieval_ms":   r.retrieval_ms,
                "ttft_ms":        r.ttft_ms,
                "generation_ms":  r.generation_ms,
                "total_ms":       r.total_ms,
                "tokens_per_sec": r.tokens_per_sec,
                "output_tokens":  r.output_tokens,
                "success":        r.success,
            }
            for r in results
        ],
        "total_ms":        summarise([r.total_ms         for r in good]),
        "retrieval_ms":    summarise([r.retrieval_ms      for r in good]),
        "ttft_ms":         summarise([r.ttft_ms           for r in good]),
        "generation_ms":   summarise([r.generation_ms     for r in good]),
        "tokens_per_sec":  summarise([r.tokens_per_sec    for r in good]),
        "output_tokens":   summarise([r.output_tokens     for r in good]),
        "chunks_retrieved":summarise([r.chunks_retrieved  for r in good]),
        "chunks_to_llm":   summarise([r.chunks_to_llm     for r in good]),
        "prompt_chars":    summarise([r.prompt_chars       for r in good]),
        "prompt_tokens":   summarise([r.prompt_tokens      for r in good]),
        "avg_chunk_len":   summarise([r.avg_chunk_len      for r in good]),
    }


def build_global_summary(queries_report: dict) -> dict:
    all_total      = []
    all_retrieval  = []
    all_ttft       = []
    all_generation = []
    all_tok_per_s  = []

    for stats in queries_report.values():
        if "error" in stats or "raw_runs" not in stats:
            continue
        for run in stats["raw_runs"]:
            if run["success"]:
                all_total.append(run["total_ms"])
                all_retrieval.append(run["retrieval_ms"])
                all_ttft.append(run["ttft_ms"])
                all_generation.append(run["generation_ms"])
                all_tok_per_s.append(run["tokens_per_sec"])

    if not all_total:
        return {}

    mean_total = statistics.mean(all_total)
    return {
        "total_ms":       summarise(all_total),
        "retrieval_ms":   summarise(all_retrieval),
        "ttft_ms":        summarise(all_ttft),
        "generation_ms":  summarise(all_generation),
        "tokens_per_sec": summarise(all_tok_per_s),
        "retrieval_pct":  round(statistics.mean(all_retrieval)  / mean_total * 100, 1),
        "generation_pct": round(statistics.mean(all_generation) / mean_total * 100, 1),
    }


def print_summary(report: dict):
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"  Strategy  : {report['strategy']}")
    print(f"  LLM       : {report['llm_model']}")
    print(f"  Embed     : {report['embed_model']}")
    print(f"  Rerank    : {report['use_rerank']}")
    print(f"  Runs      : {report['runs_per_query']} per query + 1 warmup")

    g = report.get("global_summary", {})
    if g:
        print(f"\n  {'Metric':<22} {'Mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-' * 72}")
        for key in ["total_ms", "retrieval_ms", "ttft_ms", "generation_ms"]:
            s = g.get(key, {})
            if s:
                print(
                    f"  {key:<22} "
                    f"{s['mean']:>8.0f}  "
                    f"{s['p50']:>8.0f}  "
                    f"{s['p95']:>8.0f}  "
                    f"{s['p99']:>8.0f}  "
                    f"{s['min']:>8.0f}  "
                    f"{s['max']:>8.0f}"
                )
        tps = g.get("tokens_per_sec", {})
        if tps:
            print(f"\n  Tokens/sec (mean / p95)  : {tps['mean']:.1f} / {tps['p95']:.1f}")
        print(f"\n  Retrieval % of total     : {g.get('retrieval_pct', 0):.1f}%")
        print(f"  Generation % of total    : {g.get('generation_pct', 0):.1f}%")
        print(f"\n  → The largest % above is your biggest bottleneck.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Latency Benchmark")
    parser.add_argument("--runs",    type=int, default=3)
    parser.add_argument("--config",  default="config.json")
    parser.add_argument("--queries", nargs="+")
    args = parser.parse_args()

    config          = load_config(args.config)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    metrics_dir     = Path(config["evaluation"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    strategy        = get_strategy(config)
    queries         = args.queries or DEFAULT_QUERIES

    logger.info(f"Building retriever | strategy={strategy}")
    retriever = build_retriever(config, ollama_base_url)
    llm       = build_llm(config, ollama_base_url)

    print("=" * 75)
    print("RAG PIPELINE BENCHMARK")
    print("=" * 75)
    print(f"  Strategy  : {strategy}")
    print(f"  LLM       : {config['llm']['model_name']}")
    print(f"  Embed     : {config['embeddings']['model_name']}")
    print(f"  top_k     : {config['retrieval']['top_k']}")
    print(f"  Rerank    : {config['retrieval'].get('use_rerank', False)}")
    print(f"  Queries   : {len(queries)}")
    print(f"  Runs      : {args.runs} per query + 1 warmup")
    print("=" * 75 + "\n")

    queries_report = {}
    for query in queries:
        logger.info(f"\nQuery: {query!r}")
        queries_report[query] = benchmark_query(query, retriever, llm, runs=args.runs)

    report = {
        "timestamp":      datetime.now().isoformat(),
        "strategy":       strategy,
        "llm_model":      config["llm"]["model_name"],
        "embed_model":    config["embeddings"]["model_name"],
        "top_k":          config["retrieval"]["top_k"],
        "use_rerank":     config["retrieval"].get("use_rerank", False),
        "candidate_k":    config["retrieval"].get("candidate_k", "n/a"),
        "rerank_top_k":   config["retrieval"].get("rerank_top_k", "n/a"),
        "runs_per_query": args.runs,
        "queries":        queries_report,
        "global_summary": build_global_summary(queries_report),
    }

    print_summary(report)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = metrics_dir / f"benchmark_{strategy}_{ts}.json"
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report saved → {output_path}")


if __name__ == "__main__":
    main()