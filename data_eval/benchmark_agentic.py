import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_ret.dataloader import VectorStoreLoader, BM25Loader
from agentic_ai.agent_workflow2 import CorrectiveRAGWorkflow
from agentic_ai.agent_workflow import AgenticRAGWorkflow
load_dotenv()

DEFAULT_QUERIES = [
    "What is RAG?",
    "Why is the concept of matching classes important in category theory?",
    "How does the MLMM approach affect the analysis of Root Mean Squared Error (RMSE)?",
    "What is unambiguously efficient allocation?",
    "What challenges do large position holders face when executing trades in financial markets?",
    "How do SPX smiles vary with different maturities in financial modeling?",
]


@dataclass
class AgenticStageTimings:
    prep_llm_s:   float = 0.0
    retrieval_s:  float = 0.0
    reranking_s:  float = 0.0
    grading_s:    float = 0.0
    hyde_s:       float = 0.0
    generation_s: float = 0.0
    end_to_end_s: float = 0.0
    retries:      int   = 0
    chunks_merged: int  = 0
    chunks_final:  int  = 0
    success:      bool  = True
    error:        str   = ""


@dataclass
class AgenticQueryResult:
    query: str
    runs: list[AgenticStageTimings] = field(default_factory=list)

    def stats(self) -> dict:
        good = [r for r in self.runs if r.success]
        if not good:
            return {"error": "all runs failed"}

        def summarise(values: list[float]) -> dict:
            return {
                "mean":  round(statistics.mean(values), 3),
                "min":   round(min(values), 3),
                "max":   round(max(values), 3),
                "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            }

        return {
            "successful_runs": len(good),
            "prep_llm_s":      summarise([r.prep_llm_s   for r in good]),
            "retrieval_s":     summarise([r.retrieval_s  for r in good]),
            "reranking_s":     summarise([r.reranking_s  for r in good]),
            "grading_s":       summarise([r.grading_s    for r in good]),
            "hyde_s":          summarise([r.hyde_s        for r in good]),
            "generation_s":    summarise([r.generation_s for r in good]),
            "end_to_end_s":    summarise([r.end_to_end_s for r in good]),
            "retries":         summarise([float(r.retries) for r in good]),
            "chunks_merged":   summarise([float(r.chunks_merged) for r in good]),
            "chunks_final":    summarise([float(r.chunks_final)  for r in good]),
        }


class AgenticBenchmark:
    def __init__(self, config: dict, ollama_base_url: str, use_crag: bool = False):
        self.config      = config
        self.ollama_url  = ollama_base_url
        self.use_crag    = use_crag
        self.workflow    = None

    def load_workflow(self) -> float:
        print("Loading vector store + workflow...", flush=True)
        t0 = time.perf_counter()

        embed_model = self.config["embeddings"]["model_name"]
        store       = VectorStoreLoader(self.config, self.ollama_url, embed_model).load()

        bm25 = None
        if self.config["retrieval"].get("use_hybrid", False):
            try:
                bm25 = BM25Loader(self.config).load()
            except Exception as e:
                print(f"  [warn] BM25 load failed: {e}")

        if self.use_crag:
            self.workflow = CorrectiveRAGWorkflow(
                config=self.config, ollama_base_url=self.ollama_url,
                store=store, bm25_retriever=bm25,
            )
        else:
            self.workflow = AgenticRAGWorkflow(
                config=self.config, ollama_base_url=self.ollama_url,
                store=store, bm25_retriever=bm25,
            )

        elapsed = time.perf_counter() - t0
        print(f"  Workflow loaded in {elapsed:.2f}s\n", flush=True)
        return elapsed

    def run_once(self, query: str) -> AgenticStageTimings:
        t = AgenticStageTimings()
        t_start = time.perf_counter()
        try:
            _, latency = self.workflow.run(query)
            t.prep_llm_s   = round(latency.get("prep_llm_ms",  0.0) / 1000, 3)
            t.retrieval_s  = round(latency.get("retrieval_ms", 0.0) / 1000, 3)
            t.reranking_s  = round(latency.get("reranking_ms", 0.0) / 1000, 3)
            t.grading_s    = round(latency.get("grading_ms",   0.0) / 1000, 3)
            t.hyde_s       = round(latency.get("hyde_ms",      0.0) / 1000, 3)
            t.generation_s = round(latency.get("generation_ms",0.0) / 1000, 3)
            t.retries      = latency.get("retries", 0)
            t.chunks_merged = latency.get("chunks_merged", 0)
            t.chunks_final  = latency.get("chunks_final",  0)
        except Exception as e:
            t.success = False
            t.error   = str(e)
        t.end_to_end_s = round(time.perf_counter() - t_start, 3)
        return t

    def benchmark_query(self, query: str, runs: int, warmup: bool = True) -> AgenticQueryResult:
        result = AgenticQueryResult(query=query)
        if warmup:
            print(f"  [warmup] ", end="", flush=True)
            self.run_once(query)
            print("done", flush=True)
        for i in range(runs):
            print(f"  [run {i+1}/{runs}] ", end="", flush=True)
            timing = self.run_once(query)
            result.runs.append(timing)
            status = "OK" if timing.success else f"FAIL: {timing.error}"
            print(
                f"e2e={timing.end_to_end_s:.2f}s  "
                f"retrieval={timing.retrieval_s:.3f}s  "
                f"rerank={timing.reranking_s:.3f}s  "
                f"gen={timing.generation_s:.3f}s  "
                f"retries={timing.retries}  "
                f"[{status}]",
                flush=True,
            )
        return result

    def run(self, queries: list[str], runs: int) -> dict:
        wf_label = "CRAG" if self.use_crag else "AGENTIC"
        print("=" * 65)
        print(f"AGENTIC RAG PIPELINE BENCHMARK — {wf_label}")
        print("=" * 65)
        print(f"Model   : {self.config['llm']['model_name']}")
        print(f"Embed   : {self.config['embeddings']['model_name']}")
        print(f"top_k   : {self.config['agentic'].get('final_top_k', 5)}")
        print(f"Queries : {len(queries)}")
        print(f"Runs    : {runs} per query (+ 1 warmup)")
        print("=" * 65 + "\n")

        load_s = self.load_workflow()
        all_results: dict[str, AgenticQueryResult] = {}
        for q in queries:
            print(f"\nQuery: {q!r}")
            print("-" * 55)
            all_results[q] = self.benchmark_query(q, runs=runs, warmup=True)

        return self._build_report(load_s, all_results)

    def _build_report(self, load_s: float, results: dict[str, AgenticQueryResult]) -> dict:
        wf_label = "crag" if self.use_crag else "agentic"
        report   = {
            "meta": {
                "timestamp":   datetime.now().isoformat(),
                "workflow":    wf_label,
                "model":       self.config["llm"]["model_name"],
                "embed_model": self.config["embeddings"]["model_name"],
                "final_top_k": self.config["agentic"].get("final_top_k", 5),
            },
            "store_load_s": round(load_s, 3),
            "queries":      {},
        }

        all_e2e  = []
        all_ret  = []
        all_rank = []
        all_gen  = []

        for query, result in results.items():
            stats = result.stats()
            report["queries"][query] = stats
            if "error" not in stats:
                good = [r for r in result.runs if r.success]
                all_e2e.extend( [r.end_to_end_s for r in good])
                all_ret.extend( [r.retrieval_s  for r in good])
                all_rank.extend([r.reranking_s  for r in good])
                all_gen.extend( [r.generation_s for r in good])

        if all_e2e:
            def s(vals):
                return {
                    "mean": round(statistics.mean(vals), 3),
                    "min":  round(min(vals), 3),
                    "max":  round(max(vals), 3),
                }
            report["global_summary"] = {
                "end_to_end_s":          s(all_e2e),
                "retrieval_s":           s(all_ret),
                "reranking_s":           s(all_rank),
                "generation_s":          s(all_gen),
                "retrieval_pct_of_e2e":  round(statistics.mean(all_ret)  / statistics.mean(all_e2e) * 100, 1),
                "reranking_pct_of_e2e":  round(statistics.mean(all_rank) / statistics.mean(all_e2e) * 100, 1),
                "generation_pct_of_e2e": round(statistics.mean(all_gen)  / statistics.mean(all_e2e) * 100, 1),
            }

        self._print_summary(report)
        return report

    def _print_summary(self, report: dict):
        wf = report["meta"]["workflow"].upper()
        print("\n" + "=" * 65)
        print(f"BENCHMARK SUMMARY — {wf}")
        print("=" * 65)
        print(f"Workflow load (one-time) : {report['store_load_s']:.2f}s")
        if "global_summary" in report:
            g = report["global_summary"]
            print(f"\nAcross all queries (mean / min / max):")
            print(f"  End-to-end   : {g['end_to_end_s']['mean']:.3f}s  [{g['end_to_end_s']['min']:.3f} – {g['end_to_end_s']['max']:.3f}]")
            print(f"  Retrieval    : {g['retrieval_s']['mean']:.3f}s  [{g['retrieval_s']['min']:.3f} – {g['retrieval_s']['max']:.3f}]")
            print(f"  Reranking    : {g['reranking_s']['mean']:.3f}s  [{g['reranking_s']['min']:.3f} – {g['reranking_s']['max']:.3f}]")
            print(f"  Generation   : {g['generation_s']['mean']:.3f}s  [{g['generation_s']['min']:.3f} – {g['generation_s']['max']:.3f}]")
            print(f"\n  Retrieval  % : {g['retrieval_pct_of_e2e']}%")
            print(f"  Reranking  % : {g['reranking_pct_of_e2e']}%")
            print(f"  Generation % : {g['generation_pct_of_e2e']}%")
            print()
            print("  → The biggest number above tells you where to optimise first.")
        print("=" * 65)


def load_config(path: str = "config.json") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config.json not found at {path}")
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Pipeline Latency Benchmark")
    parser.add_argument("--runs",    type=int, default=3,   help="Measured runs per query (default: 3)")
    parser.add_argument("--queries", nargs="+",             help="Custom queries to benchmark")
    parser.add_argument("--config",  default="config.json", help="Path to config.json")
    parser.add_argument("--output",  default=None,          help="Save JSON report to this path")
    parser.add_argument("--crag",    action="store_true",   help="Use CorrectiveRAG workflow instead of AgenticRAG")
    args = parser.parse_args()

    config          = load_config(args.config)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    queries         = args.queries or DEFAULT_QUERIES

    bench  = AgenticBenchmark(config=config, ollama_base_url=ollama_base_url, use_crag=args.crag)
    report = bench.run(queries=queries, runs=args.runs)

    wf_tag      = "crag" if args.crag else "agentic"
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"benchmark_results_{wf_tag}_{ts}.json"
    Path(output_path).write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved → {output_path}")
    print("Run this again after each optimisation to compare before/after.\n")


if __name__ == "__main__":
    main()
