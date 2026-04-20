"""
    What each metric measures and why:

    ContextPrecision
        Of all retrieved chunks, what fraction were actually useful for
        answering the question? High precision = no noise in retrieval.
        Uses LLM to judge relevance of each chunk.

    ContextRecall
        Did the retrieved chunks contain all the information needed to
        answer the question? Compares retrieved content against the
        ground truth answer. Low recall = missing key chunks.

    Faithfulness
        Is the generated answer grounded in the retrieved context?
        Detects hallucination — cases where the LLM adds facts not in
        the context. Measured by NLI (natural language inference).

    AnswerRelevancy
        Is the generated answer actually addressing the question asked?
        A correct but off-topic answer scores low here.

    AnswerCorrectness
        How factually accurate is the generated answer vs ground truth?
        Combines semantic similarity + factual overlap.
"""

import os
import time
import json
import warnings
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=DeprecationWarning)
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

def load_eval_data(path: Path) -> List[Dict[str, Any]]:
    print(f"Loading eval data from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        return json.load(f)
    
def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)

def build_ragas_llm(config: dict, ollama_base_url: str):
    llm = ChatOllama(
        model=config["evaluation"].get("eval_model", config["llm"]["model_name"]),
        base_url=ollama_base_url,
        temperature=0,
        seed=42,                
        num_predict=1024,       
        num_ctx=config["llm"].get("num_ctx", 4096),
        reasoning=False,       
        keep_alive="60m",      
    )
    embeddings = OllamaEmbeddings(
        model=config["embeddings"]["model_name"],
        base_url=ollama_base_url,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

def build_dataset(data: List[Dict[str, Any]], limit: int = 5) -> Dataset:
    rows = []
    for item in data[:limit]:
        contexts = [doc["text_preview"] for doc in item["retrieved_context"]]
        rows.append({
            "user_input": item["query"],
            "response": item["llm_answer"],
            "retrieved_contexts": contexts,
            "reference": item["ground_truth_answer"]})
    return Dataset.from_pandas(pd.DataFrame(rows))

def build_metrics(ragas_llm, ragas_embeddings) -> list:
    metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy(), AnswerCorrectness()]
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings
    return metrics

def main(filename: str, limit: int = 5):
    eval_data_path = Path(filename)
    config_path    = Path(__file__).resolve().parent.parent / "config.json"
    eval_data = load_eval_data(eval_data_path)
    config    = load_config(config_path)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"Model: {config['evaluation'].get('eval_model', config['llm']['model_name'])}")
    print(f"Embed: {config['embeddings']['model_name']}")
    print(f"Samples: {min(limit, len(eval_data))} / {len(eval_data)}")
    print()
    ragas_llm, ragas_embeddings = build_ragas_llm(config, ollama_base_url)
    metrics = build_metrics(ragas_llm, ragas_embeddings)
    dataset = build_dataset(eval_data, limit=limit)
    print("=" * 65)
    print(f"Evaluating {len(dataset)} samples across {len(metrics)} metrics...")
    print("=" * 65)
    t_eval_start = time.perf_counter()
    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=180,     
            max_retries=2,     
            max_workers=1))
    elapsed_s = round(time.perf_counter() - t_eval_start, 1)
    results_df = result.to_pandas()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = eval_data_path.parent / f"ragas_results_{ts}.csv"
    metric_cols = [c for c in results_df.columns
                   if c in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]]
    summary_row = {col: "MEAN" if col not in metric_cols else results_df[col].mean() for col in results_df.columns}
    summary_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
    summary_df.to_csv(output_path, index=False)
    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    scores = {}
    for name in metric_names:
        try:
            raw = result[name]                    
            valid = [v for v in raw if v == v]       
            scores[name] = sum(valid) / len(valid) if valid else float("nan")
        except (KeyError, TypeError):
            scores[name] = float("nan")
    print("\n" + "=" * 65)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Model: {config['evaluation'].get('eval_model', config['llm']['model_name'])}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Total time: {elapsed_s:.1f}s  ({elapsed_s / len(dataset):.1f}s per sample)")
    print()
    for metric_name, score in scores.items():
        is_nan = score != score
        if is_nan:
            bar    = "░" * 20
            status = "nan  ← LLM parse failed"
        else:
            filled = round(score * 20)
            bar    = "█" * filled + " " * (20 - filled)
            if score >= 0.7:
                label = "✓ good"
            elif score >= 0.4:
                label = "~ moderate"
            else:
                label = "✗ needs work"
            status = f"{score:.3f}  {label}"
        print(f"  {metric_name:<22} [{bar}]  {status}")
    print()
    print(f"  Results saved: {output_path}")
    print()

if __name__ == "__main__":
    filename = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/logs/eval/detailed_results_20260420_014246.json"
    main(filename, limit=3)