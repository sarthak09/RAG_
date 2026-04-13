# Day 3 — Retrieval & Generation Pipeline

**Date:** April 13, 2026

---
## What I was trying to do

Complete the retrieval and generation stages of the RAG pipeline so that a user query flows all the way through to a grounded answer with citations.

---

## What I built

### Project structure
Introduced a modular `data_ret/` folder where each concern lives in its own file. Everything is wired together via `main_run.py` which loads config and env, initialises each component, and runs the pipeline. I have further chnaged the sstructure of the project so that I can keep data injection part seprate.

```
backend/
├── main_run.py
├── config.json
├── .env
└── data_ret/
    ├── dataloader.py   ← loads ChromaDB vector store
    ├── retriever.py    ← embeds query, searches store
    ├── prompts.py      ← system + user prompt builders
    ├── models.py       ← Pydantic output shapes
    └── llm.py          ← calls Gemma 4, parses response
```

### Config vs environment split
Decided to keep only `OLLAMA_BASE_URL` in `.env` and read everything else — model names, chunk size, top_k — from `config.json`. This makes the codebase cleaner and easier to swap models without touching code.

---

## What I learnt

### About retrieval
- `QdrantClient` is just a Python driver for the Qdrant vector database server — same idea as `psycopg2` for Postgres. It sends an HTTP request to the running Qdrant container and returns the nearest matching vectors.
- However, since my embeddings are stored in ChromaDB (not Qdrant), I used `langchain-chroma` to load the persisted store and run `similarity_search_with_score` against it.
- Retrieval is just one function: embed the query → find the closest stored vectors → return the matching chunks with scores.

### About the vector store
- ChromaDB stores vectors with a payload. The fields available depend entirely on what you stored during ingestion.
- My ingestion stored `doc_id` but not `chunk_id` or `source_file` as separate payload fields, so those came back as `None` at retrieval time.
- Fixed this by generating a positional `chunk_id` at retrieval time: `{doc_id}_chunk_{i}`.

### About the LLM call
- The prompt has two parts — a **system prompt** telling the LLM to answer only from context and return strict JSON, and a **user prompt** injecting the retrieved chunks and the question.
- The `used_chunk_ids` field in the LLM response is how citations are traced back to source documents.
- Separating prompts into their own file (`prompts.py`) keeps `llm.py` clean and makes it easy to experiment with different prompt strategies later.

### About Pydantic output models
- Defined `RAGResponse` and `FailedRAGResponse` as separate Pydantic models. This means every code path returns a predictable shape — callers never have to check if a field exists.
- `FailedRAGResponse` has `success=False` by default and carries an `error` field, so failures are handled gracefully instead of crashing.

---

## Questions I had and what I found

**Q: Why use `QdrantClient` if I already have ChromaDB?**  
The earlier architecture planned for Qdrant. Once I confirmed ChromaDB was already populated from ingestion, I switched to `langchain-chroma` which loads the persisted SQLite store directly.

**Q: Why is `chunk_id` None in retrieved results?**  
Because it was never stored as a metadata field during ingestion. The metadata only had `doc_id`. Fixed by generating it positionally at retrieval time.

---

## Challenges and how I resolved them

### Dimension mismatch — 384 vs 768
**Problem:** `Collection expecting embedding with dimension of 384, got 768`  
**Cause:** The vector store had been built with a different embedding model than what was in `config.json`. The stored vectors were 384-dim but `nomic-embed-text` produces 768-dim.  
**Fix:** Deleted the vector store and re-ran ingestion with the correct model so both sides matched.

### LLM returning invalid JSON
**Problem:** Gemma 4 sometimes wraps its output in markdown code fences (` ```json `), and chunk text containing newlines or quotes was breaking JSON parsing.  
**Fix:** Added `_extract_json()` in `llm.py` that strips fences first, then uses `re.search(r'\{.*\}', raw, re.DOTALL)` to extract the JSON object regardless of surrounding text. `re.DOTALL` was the key — it makes `.` match newlines so multiline JSON is captured.

### Citations showing count 0
**Problem:** Citations were empty even though the LLM was answering correctly.  
**Cause:** The chunk IDs the LLM was referencing were `None` since the field wasn't in the metadata.  
**Fix:** Generating positional chunk IDs in the retriever fixed this. Citations now populate correctly and trace back to the right `doc_id`.

---

## Pipeline status

```
 Ingestion → Chunking → Embedding → Vector Store → Retrieval → Generation with citations
```

---
