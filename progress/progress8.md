# Day 5 — Hybrid Retrieval (Dense + BM25)

**Date:** April 20, 2026

---

## What I was trying to do

The baseline retrieval pipeline was working — dense vector search using ChromaDB and nomic-embed-text. But pure dense retrieval has a blind spot: exact keyword matches. A query like `"RMSE"` might miss chunks that contain the exact string "RMSE" if those chunks aren't semantically close to the query embedding. I wanted to fix that by adding a sparse retrieval layer and combining both with a principled merging strategy.

---

## What I built

Added a hybrid retrieval mode controlled by a single flag in `config.json`. When `use_hybrid: false` the pipeline behaves exactly as before. When `use_hybrid: true` both dense and sparse retrieval run and their results are merged using RRF.

New additions:

- `SparseVectorStore` class in `data_injest/vector_store.py` — builds a BM25 index from all embedded chunks and saves it as a `.pkl` file
- `BM25Loader` class in `data_ret/dataloader.py` — loads the `.pkl` at runtime in under a second
- `HybridRetriever` class in `data_ret/retriever.py` — runs both retrievers and merges results with manual RRF
- Stage 5 in `main_data.py` — builds the BM25 index during ingestion, flag-gated
- `inspect_bm25.py` and `test_bm25_store.py` — tooling to inspect and test the BM25 index
- `evaluator.py` refactored — now accepts a retriever as a constructor argument instead of building its own store internally

---

## What I learnt

### About BM25
BM25 is a keyword-based ranking algorithm. It scores chunks by counting exact word occurrences — no semantic understanding at all. If the query says `"car"` and the document says `"automobile"`, BM25 scores zero. But if the query says `"RMSE"`, BM25 will find every chunk containing exactly that string, which dense retrieval might miss if the surrounding context isn't semantically similar.

This makes BM25 and dense retrieval genuinely complementary — they fail in opposite situations.

### About Reciprocal Rank Fusion (RRF)
The naive way to combine two ranked lists is to average their scores. That doesn't work because the scores are on completely different scales — cosine similarity is 0 to 1, BM25 TF-IDF scores can be in the hundreds. You can't average them meaningfully.

RRF solves this by throwing away raw scores entirely and only using rank position:

```
score = 1 / (60 + rank)
```

The 60 is a smoothing constant that prevents the top rank from completely dominating. You compute this score for each chunk from each retriever, sum them up, and sort. A chunk that ranks highly in both lists wins — which is exactly what you want.

### About BM25 persistence
BM25 is purely in-memory — there's no built-in way to save it like ChromaDB saves to disk. The standard approach is `pickle`. Build the index once during ingestion, dump it to a `.pkl` file, and load it in under a second at runtime. Rebuilding from 182,692 chunks every startup would take 10–30 seconds.

### About candidate_k
Each sub-retriever fetches `max(top_k * 2, 20)` results before fusion, not just `top_k`. The reason: a chunk ranked #8 in dense search but #1 in BM25 should surface in the final results — but if dense only returns `top_k=5` chunks, that chunk is invisible. Fetching a wider candidate pool before merging and then trimming to `top_k` after RRF gives the fusion step something to work with.

---

## Questions I had and what I found

**Q: Why not use LangChain's EnsembleRetriever?**
Tried to. It was removed in LangChain 1.x. Attempted to downgrade to `langchain==0.3.0` but that broke `langchain_ollama` (incompatible `langchain_core` version). Reverted and implemented RRF manually instead — it's actually more transparent this way.

**Q: Why is score set to 0.0 in HybridRetriever results?**
Because RRF discards all raw scores — only rank position is used. There's no meaningful score to report at the end. The LLM only reads the text anyway, so this doesn't affect generation.

**Q: Why does the BM25 test flag page_number as missing for page 0?**
Python treats `0` as falsy, so `if not doc.metadata.get("page_number")` incorrectly flags first-page chunks as missing. The fix is `if doc.metadata.get("page_number") is None` which correctly distinguishes between "not present" and "is zero".

---

## Challenges and how I resolved them

### EnsembleRetriever doesn't exist in LangChain 1.x
Spent time chasing the right import path. `langchain.retrievers`, `langchain.retrievers.ensemble`, `langchain_community.retrievers.ensemble` — all failed. Downgrading LangChain broke the Ollama integration. Final fix: wrote RRF manually, which took about 20 lines and is now fully transparent with no hidden dependency. Also found `langchain_classic` as an alternative workaround.

### ChromaDB crashing with "too many SQL variables"
With 182,692 chunks in the store, any `collection.get(limit=N)` call where N is the full count hits SQLite's hard variable limit. This crashed `test_vector_store.py`, `test_bm25_store.py`, and `eval_prep.py`. Fixed everywhere by replacing single large `.get()` calls with a batched loop using `limit=5000` and `offset` pagination.

### evaluator.py ignoring the injected retriever
The `__init__` method was accepting a `retriever` argument but then immediately overwriting it by rebuilding a new `SimpleRetriever` internally. The `__main__` block was also calling `RetrieverEvaluator` without defining `retriever` first. Both fixed — evaluator now purely delegates to whatever retriever is passed in, and `__main__` builds the correct retriever based on the flag before passing it.

---

## Current pipeline state

```
Ingestion: PDFs → Chunks → Embeddings → ChromaDB → BM25 index (if use_hybrid: true)

Retrieval:
  use_hybrid: false → SimpleRetriever (dense only)
  use_hybrid: true  → HybridRetriever (dense + BM25 + RRF)

Evaluation: eval_prep → evaluator (MRR, Recall@k) → llm_judge / eval_ragas
```

Full corpus: 182,692 chunks from 1,000 PDFs in ChromaDB. BM25 index tested on 10-PDF subset, needs to be rebuilt for full corpus.

---

## What's next

- Rebuild BM25 index for the full 1,000-PDF corpus
- Run full evaluation comparing dense vs hybrid (MRR and Recall@k numbers)
- Flask API layer
- Vite React frontend
- README with benchmark findings
