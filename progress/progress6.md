# Day 5 — RAGAS Evaluation

**Date:** April 19, 2026

---

## What I was trying to do

Build an end-to-end evaluation script using RAGAS that takes the detailed results JSON produced by `evaluator.py` and scores the RAG pipeline across multiple dimensions — not just whether the right document was retrieved, but whether the answer was faithful, relevant, and correct.

---

## What I built

`data_eval/eval_ragas.py` — a script that loads the detailed results file, converts it into a RAGAS-compatible dataset, runs 5 metrics using a local Ollama LLM as the judge, and saves per-sample scores to a CSV.

---

## What I learnt

### About RAGAS metrics

RAGAS measures different failure modes in a RAG pipeline. Each metric targets a specific question:

| Metric | Question it answers |
|---|---|
| `ContextPrecision` | Are the retrieved chunks actually relevant to the question? |
| `ContextRecall` | Did retrieval miss anything the ground truth needs? |
| `Faithfulness` | Is the answer grounded in the context, or is the LLM hallucinating? |
| `AnswerRelevancy` | Does the answer actually address the question asked? |
| `AnswerCorrectness` | Is the answer factually correct compared to the ground truth? |

The key insight is that these metrics target different parts of the pipeline. A low `ContextRecall` means the retriever is the problem. A low `Faithfulness` means the generator is hallucinating even when the context is good. You need all of them together to know where to focus.

### About how RAGAS works internally

Most RAGAS metrics are **LLM-as-judge** — they don't compute a number mathematically. Instead, they prompt an LLM with a scoring question like *"is this answer supported by the context?"* and use the response as the score. This is why RAGAS needs an LLM configured and why each evaluation is slow — every sample-metric combination is a separate LLM call.

The total number of LLM calls is simply: `samples × metrics`. With 5 samples and 5 metrics, that is 25 LLM calls. This is visible in the progress bar as `Evaluating: 0/25`.

### About RAGAS dataset columns

RAGAS v0.2 renamed all dataset columns from the old API. If you use the old names the evaluation silently fails with a column validation error:

| Old (v0.1) | New (v0.2) |
|---|---|
| `question` | `user_input` |
| `answer` | `response` |
| `contexts` | `retrieved_contexts` |
| `ground_truths` (list) | `reference` (single string) |

### About context formatting

Each retrieved chunk must be a **separate string** in the `retrieved_contexts` list — not joined into one blob. RAGAS evaluates per-chunk precision, so joining destroys that granularity. Wrong vs right:

```python
# Wrong — loses per-chunk information
"retrieved_contexts": ["\n\n".join(doc["text_preview"] for doc in item["retrieved_context"])]

# Right — each chunk is its own string
"retrieved_contexts": [doc["text_preview"] for doc in item["retrieved_context"]]
```

### About RAGAS and local LLMs

RAGAS's new native integrations only support OpenAI, Google, and HuggingFace. For Ollama you have to use `LangchainLLMWrapper` and `LangchainEmbeddingsWrapper`. These are deprecated in v0.2 but still work — there is no clean alternative for local models yet.

---

## Questions I had and what I found

**Q: Why is the progress bar showing 25 even though I only have 5 samples?**  
Because RAGAS runs each metric independently per sample. 5 samples × 5 metrics = 25 separate LLM calls. To reduce this during development, use fewer metrics or fewer samples.

**Q: Why are these deprecation warnings appearing?**  
Two reasons. First, the import path for metric classes moved from `ragas.metrics` to `ragas.metrics.collections` in v0.2. Second, `LangchainLLMWrapper` is being phased out in favour of native provider integrations. The first one can be fixed immediately by updating the import path. The second one cannot be fixed cleanly without switching away from Ollama.

---

## Challenges and how I resolved them

### File named `ragas.py` shadowing the installed package
**Problem:** `ModuleNotFoundError: No module named 'ragas.metrics'; 'ragas' is not a package`  
**Cause:** Python searches the current directory before installed packages. A file named `ragas.py` in the same directory shadows the real `ragas` package entirely.  
**Fix:** Renamed the file to `eval_ragas.py`. Rule: never name your file the same as a package you import from.

### `ragas` package not installed
**Problem:** `ModuleNotFoundError: No module named 'ragas'`  
**Fix:** `uv add ragas`

### `context_relevancy` does not exist in v0.2
**Problem:** `ImportError: cannot import name 'context_relevancy'`  
**Cause:** This metric was removed in the newer RAGAS version.  
**Fix:** Removed it from imports and metrics list.

### Metrics must be instantiated as objects
**Problem:** `TypeError: All metrics must be initialised metric objects`  
**Cause:** Old RAGAS used module-level singletons (`context_precision`). New RAGAS requires class instances (`ContextPrecision()`).  
**Fix:** Updated all metrics to use PascalCase class names with parentheses.

### TimeoutErrors causing `context_precision: nan`
**Problem:** RAGAS runs metric jobs concurrently by default. Local Ollama cannot respond fast enough when hit with parallel requests, so most jobs timed out and the metric returned `nan`.  
**Fix:** Added `RunConfig(timeout=120, max_retries=3, max_workers=1)` to force sequential evaluation.

### Column validation error
**Problem:** `ValueError: metric [context_precision] requires column 'reference'`  
**Cause:** Using old v0.1 column names (`ground_truths`, `question`, etc.) with a v0.2 RAGAS install.  
**Fix:** Updated `build_dataset()` to use new column names.

---

## First results

```
context_precision  : nan   ← timed out, not a real score
context_recall     : 0.60
faithfulness       : 0.50
answer_relevancy   : 0.59
answer_correctness : 0.11
```

`answer_correctness` of 0.11 is low but expected. Looking at the sample data, the correct document was being retrieved at rank 5, not rank 1 — the LLM was generating answers from suboptimal context. The retriever is the bottleneck, not the generator.

---

## Pipeline status

```
Ingestion → Chunking → Embedding → Vector Store → Retrieval → Generation → Evaluation (RAGAS) ✓
```
