# Day 7 — Retrieval & Generation Pipeline

**Date:** April 17, 2026

---
## What I was trying to do

Focus on the RAG evaluation methods and preparation of the dataset.

---

## What I learnt

### About the evaluation dataset structure
The three metadata files share a single UUID key per query. Once I understood that, joining them was straightforward:

- `queries.json` → the question text, type, source
- `qrels.json` → which document answers it, which section
- `answers.json` → the ground truth answer

The UUID is the join key across all three. `qrels.json` has exactly one relevant `doc_id` per query — simpler than the standard TREC format which can have multiple relevant documents.

### About the doc_id matching problem
Before writing a single line of evaluation code, I checked whether the doc_ids in `qrels.json` matched what was stored in ChromaDB. They did both use the full arxiv versioned filename (e.g. `2401.01872v2`). This is the #1 silent failure mode in evaluation — if the IDs don't match you get Recall@k = 0 for every query and have no idea why.

### About Recall@k vs MRR
These measure different things and you need both:
- **Recall@k** answers: "did the correct document appear *anywhere* in my top-k results?"
- **MRR** answers: "how *high* did the correct document rank?"

You could have Recall@5 = 0.99 but MRR = 0.4, which would mean the system almost always finds the right document but usually buries it at rank 4 or 5. That's a reranking problem, not a retrieval problem. The distinction matters when deciding what to fix next.

### About corpus size and evaluation integrity
My first evaluation run used only 100 PDFs and returned MRR = 0.94, Recall@5 = 0.99. These numbers look impressive but they're misleading with only 100 documents in the store, the retriever is picking from a tiny pool. The real test is at 1,000 PDFs where the retriever has to distinguish the correct paper from 999 others, many of which are thematically similar arXiv papers. Baseline numbers only mean something when the corpus is at full scale.

### About the source breakdown
Even with the inflated small-corpus numbers, one split in the results is genuinely informative:

| Source type | MRR |
|---|---|
| text-table | 1.0 |
| text | 0.95 |
| text-image | 0.92 |

Text-image queries score lowest because those questions are designed to be answered from a figure or diagram in the paper — and my system only indexes text. The retriever still finds the right document (the surrounding text is relevant enough) but ranks it lower. This is a real signal that image embeddings would help for that subset.

Text-table queries score perfect — pymupdf4llm converts tables to markdown, so the structured data is well-represented as plain text and retrieves cleanly.

### About separating data preparation from evaluation
The decision to split `eval_prep.py` (prepare once, save to disk) and `evaluator.py` (load and run) turned out to be important. When I scale to 1,000 PDFs, I just re-run `eval_prep.py` to get a larger filtered dataset, then re-run `evaluator.py`. The evaluator never cares how many queries there are — it just loops whatever it receives.

---

## Questions I had and what I found

**Q: Should I hardcode the number of evaluable queries anywhere?**
No. The evaluator loads whatever `eval_dataset.json` contains. As more PDFs get indexed, re-running `eval_prep.py` produces a larger dataset automatically. Zero code changes needed.

**Q: Why evaluate at k=1, 3, 5, and 10 instead of just one value?**
Each k tells a different story. Recall@1 tells you how often the system gets it right on the first result — a high bar. Recall@10 tells you the ceiling — how often it finds the answer at all given enough results. The gap between Recall@1 and Recall@10 shows you how much room reranking has to improve things.

**Q: Why save both JSON and CSV?**
JSON is for the next stage — RAGAS evaluation reads the full chunk text from the JSON. CSV is for human inspection — open it in a spreadsheet and you can immediately see which queries failed, what was retrieved instead, and whether there's a pattern.

---

## Challenges and how I resolved them

### `FileNotFoundError: config.json`
**Problem:** Running `python data_eval/eval_prep.py` from inside the `data_eval/` folder couldn't find `config.json`.  
**Fix:** Used `Path(__file__).resolve().parent.parent / "config.json"` in the `__main__` block. Now the script resolves paths relative to its own location, not wherever you run it from. Works from any directory.

### `KeyError: 'meta_dir'`
**Problem:** The `config.json` evaluation block was missing the `meta_dir` key.  
**Fix:** Added the full evaluation config block with `meta_dir`, `metrics_dir`, `qrels_path`, and `answers_path`.

### `NameError: relevant_doc_in_results is not defined`
**Problem:** Defined the helper function inside the class instead of outside it. Python couldn't find it when called from a method.  
**Fix:** Rewrote the entire `evaluator.py` cleanly from scratch with the helper function defined at module level, above the class.

---

## Current output files

After running both scripts, `logs/eval/` contains:

| File | What it is |
|---|---|
| `eval_dataset.csv` | Filtered eval set — queries, ground truth, doc mapping |
| `eval_dataset.json` | Same, used by evaluator |
| `retrieval_metrics_*.json` | Aggregate MRR + Recall@k + breakdown by type and source |
| `detailed_results_*.json` | Per-query: query + ground truth + LLM answer + top-10 chunks |
| `detailed_results_*.csv` | Same flattened — open in spreadsheet |

---
