import json
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

def was_correct_doc_found(result: dict) -> bool:
    return any(result[f"recall@{k}"] == 1 for k in [1, 3, 5, 10])

def build_system_prompt() -> str:
    return """You are a precise research assistant.
Answer the question strictly using the provided context chunks.
If the context does not contain enough information, say: "The context does not contain sufficient information."
Be concise and factual. Return only the answer as plain text — no JSON, no markdown."""

def build_user_prompt(query: str, chunks: list) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = f"[{i}] doc_id: {chunk['doc_id']}\n{chunk['text_preview']}"
        context_blocks.append(block)
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"

class RetrieverEvaluator:
    def __init__(self, config: dict):
        self.config      = config
        self.k_values    = [1, 3, 5, 10]
        self.metrics_dir = Path(config["evaluation"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        ollama_base_url = config["embeddings"]["base_url"]
        embeddings = OllamaEmbeddings(
            model=config["embeddings"]["model_name"],
            base_url=ollama_base_url)
        self.store = Chroma(
            collection_name=config["vector_store"]["collection_name"],
            persist_directory=config["vector_store"]["persist_dir"],
            embedding_function=embeddings)
        self.llm = ChatOllama(
            model=config["llm"]["model_name"],
            base_url=ollama_base_url,
            temperature=config["llm"]["temperature"],
            num_predict=config["llm"]["max_tokens"],)
        self._last_results = []

    def evaluate(self, eval_dataset: list[dict]) -> dict:
        top_k   = max(self.k_values)
        results = []
        for i, item in enumerate(eval_dataset):
            query_id = item["query_id"]
            query_text = item["query"]
            relevant_doc = item["relevant_doc_id"]
            raw_results = self.store.similarity_search_with_score(query_text, k=top_k)
            retrieved_doc_ids = [doc.metadata.get("doc_id", "") for doc, _ in raw_results]
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
                if doc_id == relevant_doc:
                    reciprocal_rank = 1.0 / rank
                    break
            recall_at_k = {
                f"recall@{k}": int(relevant_doc in retrieved_doc_ids[:k])
                for k in self.k_values
            }
            results.append({
                "query_id": query_id,
                "query": query_text,
                "type": item["type"],
                "source": item["source"],
                "relevant_doc": relevant_doc,
                "retrieved_docs": retrieved_doc_ids,
                "reciprocal_rank": round(reciprocal_rank, 4),
                **recall_at_k,
            })
        self._last_results = results
        n = len(results)
        summary = {
            "total_queries": n,
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
                "count": nt,
                "mrr": round(sum(r["reciprocal_rank"] for r in group) / nt, 4),
                "recall@5":  round(sum(r["recall@5"]        for r in group) / nt, 4),
                "recall@10": round(sum(r["recall@10"]       for r in group) / nt, 4),
            }
        by_source = defaultdict(list)
        for r in results:
            by_source[r["source"]].append(r)
        summary["by_source"] = {}
        for name, group in by_source.items():
            nt = len(group)
            summary["by_source"][name] = {
                "count": nt,
                "mrr": round(sum(r["reciprocal_rank"] for r in group) / nt, 4),
                "recall@5": round(sum(r["recall@5"]        for r in group) / nt, 4),
                "recall@10": round(sum(r["recall@10"]       for r in group) / nt, 4),}
        self._save_metrics(summary, results)
        self._print_summary(summary)
        return summary

    def save_detailed_results(self, eval_dataset: list[dict]) -> list[dict]:
        if not self._last_results:
            return []
        gt_lookup = {item["query_id"]: item for item in eval_dataset}
        detailed = []
        total = len(self._last_results)
        for i, r in enumerate(self._last_results):
            qid = r["query_id"]
            ground_truth = gt_lookup.get(qid, {})
            raw_results = self.store.similarity_search_with_score(
                r["query"], k=max(self.k_values))
            retrieved_context = [
                {
                    "rank": rank,
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "chunk_index": doc.metadata.get("chunk_index", ""),
                    "page_number": doc.metadata.get("page_number", ""),
                    "section": doc.metadata.get("section_title", ""),
                    "score": round(float(score), 4),
                    "text_preview": doc.page_content[:500]}
                for rank, (doc, score) in enumerate(raw_results, start=1)
            ]
            llm_answer = self._generate_answer(r["query"], retrieved_context)
            detailed.append({
                "query_id": qid,
                "query": r["query"],
                "type": r["type"],
                "source": r["source"],
                "relevant_doc_id": ground_truth.get("relevant_doc_id", ""),
                "section_id": ground_truth.get("section_id", ""),
                "ground_truth_answer": ground_truth.get("answer", ""),
                "llm_answer": llm_answer,
                "reciprocal_rank": r["reciprocal_rank"],
                "recall@1": r["recall@1"],
                "recall@3": r["recall@3"],
                "recall@5": r["recall@5"],
                "recall@10": r["recall@10"],
                "correct_doc_found": was_correct_doc_found(r),
                "retrieved_context": retrieved_context,
            })
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.metrics_dir / f"detailed_results_{ts}.json"
        json_path.write_text(json.dumps(detailed, indent=2))
        csv_path = self.metrics_dir / f"detailed_results_{ts}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "query_id", "query", "type", "source",
                "relevant_doc_id", "section_id",
                "ground_truth_answer", "llm_answer",
                "reciprocal_rank",
                "recall@1", "recall@3", "recall@5", "recall@10",
                "correct_doc_found",
                "top1_doc_id", "top1_score", "top1_text_preview",
                "top2_doc_id", "top2_score", "top2_text_preview",
                "top3_doc_id", "top3_score", "top3_text_preview",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in detailed:
                ctx = d["retrieved_context"]
                writer.writerow({
                    "query_id": d["query_id"],
                    "query": d["query"],
                    "type": d["type"],
                    "source": d["source"],
                    "relevant_doc_id": d["relevant_doc_id"],
                    "section_id": d["section_id"],
                    "ground_truth_answer": d["ground_truth_answer"],
                    "llm_answer": d["llm_answer"],
                    "reciprocal_rank": d["reciprocal_rank"],
                    "recall@1": d["recall@1"],
                    "recall@3": d["recall@3"],
                    "recall@5": d["recall@5"],
                    "recall@10": d["recall@10"],
                    "correct_doc_found": d["correct_doc_found"],
                    "top1_doc_id": ctx[0]["doc_id"] if len(ctx) > 0 else "",
                    "top1_score":  ctx[0]["score"] if len(ctx) > 0 else "",
                    "top1_text_preview": ctx[0]["text_preview"] if len(ctx) > 0 else "",
                    "top2_doc_id": ctx[1]["doc_id"] if len(ctx) > 1 else "",
                    "top2_score": ctx[1]["score"] if len(ctx) > 1 else "",
                    "top2_text_preview": ctx[1]["text_preview"]   if len(ctx) > 1 else "",
                    "top3_doc_id": ctx[2]["doc_id"] if len(ctx) > 2 else "",
                    "top3_score": ctx[2]["score"] if len(ctx) > 2 else "",
                    "top3_text_preview": ctx[2]["text_preview"] if len(ctx) > 2 else ""})
        return detailed

    def _generate_answer(self, query: str, chunks: list[dict]) -> str:
        try:
            messages = [
                SystemMessage(content=build_system_prompt()),
                HumanMessage(content=build_user_prompt(query, chunks))
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return "LLM generation failed."

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
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.metrics_dir / f"retrieval_metrics_{ts}.json"
        json_path.write_text(
            json.dumps({"summary": summary, "per_query": per_query}, indent=2))

if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    config = json.load(open(config_path))
    eval_path = Path(config["evaluation"]["metrics_dir"]) / "eval_dataset.json"
    eval_dataset = json.loads(eval_path.read_text())
    evaluator = RetrieverEvaluator(config)
    evaluator.evaluate(eval_dataset)
    evaluator.save_detailed_results(eval_dataset)