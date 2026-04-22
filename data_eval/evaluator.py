import os
import json
import csv
import sys
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from data_ret.dataloader import VectorStoreLoader, BM25Loader
from data_ret.retriever import SimpleRetriever, HybridRetriever
from data_ret.llm_fast import LLMGenerator         

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.evaluator")

def was_correct_doc_found(result: dict) -> bool:
    return any(result[f"recall@{k}"] == 1 for k in [1, 3, 5, 10])

class RetrieverEvaluator:
    def __init__(self, config: dict, retriever, ollama_base_url: str):
        self.config = config
        self.retriever = retriever
        self.k_values = [1, 3, 5, 10]
        self.metrics_dir = Path(config["evaluation"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.generator = LLMGenerator(config=config, ollama_base_url=ollama_base_url)
        self._last_results: list[dict] = []
        logger.info(f"RetrieverEvaluator ready | llm={config['llm']['model_name']}")

    def evaluate(self, eval_dataset: list[dict]) -> dict:
        top_k = max(self.k_values)
        results = []
        for i, item in enumerate(eval_dataset):
            query_text = item["query"]
            relevant_doc = item["relevant_doc_id"]
            chunks, _ = self.retriever.retrieve(query_text, top_k=top_k)
            retrieved_doc_ids = [c["doc_id"] for c in chunks]
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
                if doc_id == relevant_doc:
                    reciprocal_rank = 1.0 / rank
                    break
            recall_at_k = {
                f"recall@{k}": int(relevant_doc in retrieved_doc_ids[:k])
                for k in self.k_values}
            results.append({
                "query_id":        item["query_id"],
                "query":           query_text,
                "type":            item["type"],
                "source":          item["source"],
                "relevant_doc":    relevant_doc,
                "retrieved_docs":  retrieved_doc_ids,
                "reciprocal_rank": round(reciprocal_rank, 4),
                **recall_at_k})
            if (i + 1) % 50 == 0:
                logger.info(f"  Evaluated {i + 1} / {len(eval_dataset)} queries")
        self._last_results = results
        n = len(results)
        summary = {"total_queries": n,
            "mrr": round(sum(r["reciprocal_rank"] for r in results) / n, 4)}
        for k in self.k_values:
            key = f"recall@{k}"
            summary[key] = round(sum(r[key] for r in results) / n, 4)
        by_type = defaultdict(list)
        for r in results:
            by_type[r["type"]].append(r)
        summary["by_type"] = {}
        for name, group in by_type.items():
            nt = len(group)
            summary["by_type"][name] = {
                "count":     nt,
                "mrr":       round(sum(r["reciprocal_rank"] for r in group) / nt, 4),
                "recall@5":  round(sum(r["recall@5"]  for r in group) / nt, 4),
                "recall@10": round(sum(r["recall@10"] for r in group) / nt, 4)}
        by_source = defaultdict(list)
        for r in results:
            by_source[r["source"]].append(r)
        summary["by_source"] = {}
        for name, group in by_source.items():
            nt = len(group)
            summary["by_source"][name] = {
                "count":     nt,
                "mrr":       round(sum(r["reciprocal_rank"] for r in group) / nt, 4),
                "recall@5":  round(sum(r["recall@5"]  for r in group) / nt, 4),
                "recall@10": round(sum(r["recall@10"] for r in group) / nt, 4)}
        self._save_metrics(summary, results)
        self._print_summary(summary)
        return summary

    def save_detailed_results(self, eval_dataset: list[dict]) -> list[dict]:
        if not self._last_results:
            logger.warning("No cached results — call evaluate() first.")
            return []
        gt_lookup = {item["query_id"]: item for item in eval_dataset}
        detailed = []
        total = len(self._last_results)
        for i, r in enumerate(self._last_results):
            qid = r["query_id"]
            ground_truth = gt_lookup.get(qid, {})
            chunks, _ = self.retriever.retrieve(r["query"])
            retrieved_context = [
                {
                    "rank":         rank,
                    "doc_id":       c["doc_id"],
                    "chunk_index":  c.get("chunk_index", ""),
                    "page_number":  c.get("page_number", ""),
                    "section":      c.get("section_title", ""),
                    "score":        round(float(c["score"]), 4),
                    "text_preview": c["text"][:500],
                }
                for rank, c in enumerate(chunks, start=1)]
            result, _ = self.generator.generate(r["query"], chunks)
            llm_answer = result.answer
            detailed.append({
                "query_id":            qid,
                "query":               r["query"],
                "type":                r["type"],
                "source":              r["source"],
                "relevant_doc_id":     ground_truth.get("relevant_doc_id", ""),
                "section_id":          ground_truth.get("section_id", ""),
                "ground_truth_answer": ground_truth.get("answer", ""),
                "llm_answer":          llm_answer,
                "reciprocal_rank":     r["reciprocal_rank"],
                "recall@1":            r["recall@1"],
                "recall@3":            r["recall@3"],
                "recall@5":            r["recall@5"],
                "recall@10":           r["recall@10"],
                "correct_doc_found":   was_correct_doc_found(r),
                "retrieved_context":   retrieved_context})
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"  Generated {i + 1} / {total}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.metrics_dir / f"detailed_results_{ts}.json"
        json_path.write_text(json.dumps(detailed, indent=2))
        logger.info(f"Detailed results (JSON) → {json_path}")
        csv_path  = self.metrics_dir / f"detailed_results_{ts}.csv"
        fieldnames = [
            "query_id", "query", "type", "source",
            "relevant_doc_id", "section_id",
            "ground_truth_answer", "llm_answer",
            "reciprocal_rank",
            "recall@1", "recall@3", "recall@5", "recall@10",
            "correct_doc_found",
            "top1_doc_id", "top1_score", "top1_text_preview",
            "top2_doc_id", "top2_score", "top2_text_preview",
            "top3_doc_id", "top3_score", "top3_text_preview"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in detailed:
                ctx = d["retrieved_context"]
                writer.writerow({
                    "query_id":            d["query_id"],
                    "query":               d["query"],
                    "type":                d["type"],
                    "source":              d["source"],
                    "relevant_doc_id":     d["relevant_doc_id"],
                    "section_id":          d["section_id"],
                    "ground_truth_answer": d["ground_truth_answer"],
                    "llm_answer":          d["llm_answer"],
                    "reciprocal_rank":     d["reciprocal_rank"],
                    "recall@1":            d["recall@1"],
                    "recall@3":            d["recall@3"],
                    "recall@5":            d["recall@5"],
                    "recall@10":           d["recall@10"],
                    "correct_doc_found":   d["correct_doc_found"],
                    "top1_doc_id":         ctx[0]["doc_id"]       if len(ctx) > 0 else "",
                    "top1_score":          ctx[0]["score"]        if len(ctx) > 0 else "",
                    "top1_text_preview":   ctx[0]["text_preview"] if len(ctx) > 0 else "",
                    "top2_doc_id":         ctx[1]["doc_id"]       if len(ctx) > 1 else "",
                    "top2_score":          ctx[1]["score"]        if len(ctx) > 1 else "",
                    "top2_text_preview":   ctx[1]["text_preview"] if len(ctx) > 1 else "",
                    "top3_doc_id":         ctx[2]["doc_id"]       if len(ctx) > 2 else "",
                    "top3_score":          ctx[2]["score"]        if len(ctx) > 2 else "",
                    "top3_text_preview":   ctx[2]["text_preview"] if len(ctx) > 2 else ""})
        logger.info(f"Detailed results (CSV)  → {csv_path}")
        return detailed

    def _print_summary(self, s: dict):
        print("\n" + "=" * 55)
        print("RETRIEVAL EVALUATION — BASELINE")
        print("=" * 55)
        print(f"  Queries evaluated : {s['total_queries']}")
        print(f"  MRR               : {s['mrr']}")
        for k in self.k_values:
            print(f"  Recall@{k:<3}        : {s[f'recall@{k}']}")
        print("\n  By query type:")
        for name, m in s.get("by_type", {}).items():
            print(f"    {name:<20} n={m['count']:<5} MRR={m['mrr']}  Recall@5={m['recall@5']}")
        print("\n  By source:")
        for name, m in s.get("by_source", {}).items():
            print(f"    {name:<20} n={m['count']:<5} MRR={m['mrr']}  Recall@5={m['recall@5']}")
        print("=" * 55 + "\n")

    def _save_metrics(self, summary: dict, per_query: list):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.metrics_dir / f"retrieval_metrics_{ts}.json"
        json_path.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))
        logger.info(f"Retrieval metrics → {json_path}")

if __name__ == "__main__":
    load_dotenv()
    config_path  = Path(__file__).resolve().parent.parent / "config.json"
    config       = json.load(open(config_path))
    ollama_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    eval_path    = Path(config["evaluation"]["metrics_dir"]) / "eval_dataset.json"
    eval_dataset = json.loads(eval_path.read_text())
    store = VectorStoreLoader(config=config, ollama_base_url=ollama_url, embed_model=config["embeddings"]["model_name"]).load()

    if config["retrieval"]["use_hybrid"]:
        bm25 = BM25Loader(config).load()
        retriever = HybridRetriever(config=config, store=store, bm25_retriever=bm25)
    else:
        retriever = SimpleRetriever(config=config, store=store)

    evaluator = RetrieverEvaluator(config=config, retriever=retriever, ollama_base_url=ollama_url)
    evaluator.evaluate(eval_dataset)
    evaluator.save_detailed_results(eval_dataset)