# Day 3 — Embedding, Vector Store, and Getting the Pipeline to Actually Work

**Date:** April 12, 2026

---

## What I Was Trying to Do Today

Get the full ingestion pipeline running end to end — from raw PDFs all the way to vectors stored in ChromaDB. On paper this sounds like just wiring things together. In practice it involved a lot of debugging, a few fundamental lessons, and one architecture mistake that I had to undo — and one library choice that backfired completely.

---

## What I Built

Five new components today:

- **`models.py`** — a shared file that defines `Chunk` and `EmbeddedChunk` as the single source of truth for data structures across the whole pipeline
- **`embedder.py`** — converts chunks into 768-dimensional vectors using `nomic-embed-text` via Ollama
- **`vector_store.py`** — stores those vectors (plus all metadata) in ChromaDB and persists them to disk
- **`test_embeddings.py`** — 6 tests to verify the embedding model is working correctly
- **`test_vector_store.py`** — 10 tests to verify everything stored in ChromaDB is correct and queryable

---

## What I Learned About Embeddings

An embedding model takes text and compresses it into a fixed list of numbers — 768 of them for `nomic-embed-text`. These numbers encode meaning. Similar sentences land close together in that 768-dimensional space. That closeness is exactly what makes similarity search possible later.

**Dimensions don't change with chunk size.** A 100-character chunk and a 2000-character chunk both produce a 768-number vector. What changes is the *quality* of those numbers. A focused chunk about one idea produces a sharp, specific vector. A massive chunk covering five different topics produces a blurry vector that doesn't match any query well.

**You can't change dimensions without changing models.** 768 is baked into `nomic-embed-text`'s weights. To get 1024 dimensions you'd need `mxbai-embed-large`. And if you ever swap models, you have to re-embed the entire corpus because the two models use completely different coordinate systems — mixing them is like comparing GPS coordinates with compass bearings.

**Auto-detecting dimensions.** Instead of hardcoding `768` in `config.json`, I made the embedder embed a single dummy word at startup and measure the output length. Now if I switch models, the dimension updates automatically. One less thing to forget to change.

**Batch vs one-by-one.** I implemented both. Batch sends 32 chunks in one API call instead of 32 separate calls. The actual computation time is similar but the overhead per call adds up — batch is roughly 10-20x faster for the full dataset. One-by-one is still useful for debugging individual failures.

---

## What I Learned About ChromaDB

ChromaDB is a local vector database that persists to disk. You point it at a folder and it handles everything — storing vectors, storing metadata alongside them, and doing approximate nearest-neighbour search when you query.

The key design decision was **what metadata to store with each vector**. I ended up storing:

- Citation info: `source_file`, `arxiv_id`, `section_title`, `page_number`
- Position info: `chunk_index`, `is_first_chunk`, `is_last_chunk`, `total_chunks_in_doc`
- Operational info: `embedding_model`, `chunk_size_config`, `overlap_config`, `embedded_at`

The citation fields matter because once the system retrieves a chunk and generates an answer, I want to be able to say *"this answer came from arxiv:2301.04567, section 3"* rather than just returning text with no source. The operational fields matter because if retrieval quality is poor months from now, I need to know exactly what configuration produced that index.

---

## The LangChain Chroma Trap

I initially switched `vector_store.py` to use `langchain-chroma` to keep the stack consistent. The code looked clean. It ran without errors. But the test suite showed vectors stored at **384 dimensions** instead of 768 — even though logs confirmed `nomic-embed-text` was producing 768-dim vectors.

What was happening: `langchain_chroma.Chroma.add_texts()` silently **ignored my pre-computed vectors** and re-embedded the text itself using its own default model (`all-MiniLM-L6-v2`, 384-dim). This is a known design issue — LangChain Chroma is built for workflows where LangChain controls the embedder end to end. When you manage embeddings yourself (compute them with Ollama, pass them in), LangChain overwrites your work without warning.

The fix was to revert to raw `chromadb` directly. `collection.add(embeddings=[...])` does exactly what you tell it — your 768-dim vectors go in untouched. The lesson: **LangChain abstractions are helpful when you want LangChain to manage the whole flow. When you own a step yourself, use the underlying library directly.**

This is also why the `loader.py` switch to `langchain_community.PyMuPDFLoader` is fine — the loader just returns text, LangChain isn't making any decisions about the output. But for vector storage where the embedding values matter precisely, the abstraction was dangerous.

---

## The Architecture Mistake

I defined `EmbeddedChunk` as a dataclass in both `embedder.py` and `vector_store.py`. They had the same name but were completely separate Python classes. The embedder created objects using its own definition (5 fields). The vector store expected objects from its definition (17 fields). Python didn't complain — it just failed silently at runtime when the vector store tried to access `source_file` and got `AttributeError`.

The fix was `models.py` — one file, one definition, everything imports from there. The rule going forward: if more than one file needs the same data structure, it lives in `models.py`.

The same problem existed in `splitter.py` — it had its own `Chunk` dataclass with only 5 fields. Fixed by importing `Chunk` from `models.py` and updating `_split_document()` to populate `source_file`, `arxiv_id`, `section_title`, `is_first_chunk`, `is_last_chunk` from the `ParsedDocument` the loader provides.

---

## Loader Switches

Switched both loaders to use LangChain document loaders:

- `loader.py`: `pymupdf4llm.to_markdown()` → `langchain_community.PyMuPDFLoader` — same underlying engine, now returns one `Document` per page which gives real `page_number` values for citations
- `loader2.py`: `DoclingLoader` from `langchain_community` was attempted but the class isn't exposed in the installed version. Reverted to direct `docling.DocumentConverter` since it's just a thin wrapper anyway — no functional difference

---

## Questions I Had and What I Found

**Does chunk size affect embedding dimensions?**
No. Dimensions are fixed by the model. What chunk size affects is the *quality* of those dimensions — too small and there's not enough context, too large and the vector becomes a blur of unrelated ideas.

**What's the right batch size for Ollama?**
Experimentally, 32 is safe for `nomic-embed-text` locally. Too large and you risk OOM on the GPU. The batch size is now in `config.json` so it's easy to tune.

**Should persist_dir be in `.env` or `config.json`?**
`config.json`. `.env` is for secrets and machine-specific credentials — API keys, passwords, ports. A directory path is application config, not a secret.

**What is PageIndex?**
Came across it today — a vectorless RAG approach that builds a hierarchical table-of-contents from documents and uses LLM reasoning to navigate it instead of vector similarity. The argument is that similarity ≠ relevance, and relevance requires reasoning. Interesting alternative paradigm worth mentioning in the README, but requires OpenAI API and doesn't fit the local-only constraint.

**When should you use LangChain wrappers vs underlying libraries directly?**
Use LangChain wrappers for loaders, splitters, and LLM calls — they add convenience without taking ownership of your data. Avoid them for vector storage when you're managing embeddings yourself — LangChain assumes it owns the embedding step and will silently override your vectors.

---

## Challenges and How I Resolved Them

**`nomic-embed-text` not found (404)**
Ollama was running but the model wasn't pulled. Fixed with `ollama pull nomic-embed-text`. Lesson: Ollama running ≠ model available. Always verify with `ollama list` before running the pipeline.

**`'Chunk' object has no attribute 'source_file'` — twice**
First occurrence: `vector_store.py` had its own `EmbeddedChunk` definition. Fixed by importing from `models.py`.
Second occurrence: `splitter.py` still had its own `Chunk` definition. Fixed same way. The error was the same both times — a locally defined dataclass missing new fields. The root cause was always duplicate class definitions.

**Embedder was successfully embedding but vector store was receiving 0 chunks**
The embedding was working (Ollama was responding with 200 OK) but `_to_embedded()` crashed on every chunk because of the missing field. This meant `embedded_chunks` was an empty list by the time it reached the vector store — which logged "No chunks to add" without explaining why. Added a dimension log line to surface this earlier.

**LangChain Chroma silently storing 384-dim vectors instead of 768**
`langchain_chroma.Chroma.add_texts()` ignored pre-computed embeddings and re-embedded text with its own default model. Discovered only in the test suite — the pipeline ran without errors and reported the correct insert count. Fixed by reverting to raw `chromadb.PersistentClient` with `collection.add(embeddings=[...])` which stores exactly what you give it.

**`DoclingLoader` import error**
`from langchain_community.document_loaders import DoclingLoader` failed — the class isn't exposed in the installed version of `langchain-community`. Reverted `loader2.py` to the direct `docling.DocumentConverter` import. No functional change.

---

## Current State of the Pipeline

```
PDFs → loader.py → splitter.py → embedder.py → vector_store.py
     
```

50 documents run as a test. Full 1,000-PDF run in progress.

- **50 documents parsed**
- **~10,000 chunks created**
- **~10,000 vectors stored in ChromaDB at 768 dimensions**
- **Persist dir**: `data/vector_store/` — survives process restarts

---

## What's Next

Retrieval pipeline — given a user query, embed it using the same `nomic-embed-text` model, search ChromaDB for top-k similar chunks, and return results with full citation metadata. This is where `qrels.json` becomes useful — I can finally measure whether the right chunks are actually being retrieved.