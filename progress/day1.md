# Progress Update

**Date:** April 10, 2026  

---

## What I'm Building & Why

I'm building a **production-grade RAG (Retrieval-Augmented Generation) system** from scratch. The goal is simple — I want to deeply understand how RAG works in the real world, not just follow a tutorial. By the end of this project I want to be able to explain every design decision, every tradeoff, and every failure I hit along the way.


---

## Dataset: Vectara Open RAG Benchmark

The dataset I chose is the [Vectara Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench). Here's what it contains:

- **1,000 arXiv research papers** (PDFs) across all scientific domains
- **3,045 QA pairs** — real questions with ground truth answers
- **400 positive documents** (each one answers some queries)
- **600 hard negative documents** — papers that look relevant but aren't. These are designed to fool retrieval systems.

What makes this dataset special for learning RAG is that it comes with **ground truth** — I know exactly which document and which section should answer each question. Most real-world RAG projects don't have this. I can actually measure whether my system is working rather than just guessing.

The metadata breaks down into 4 key files:

| File | What it is | How I'll use it |
|------|-----------|----------------|
| `queries.json` | 3,045 questions | Test retrieval |
| `qrels.json` | Query → doc → section mapping | Measure retrieval accuracy |
| `answers.json` | Ground truth answers | Measure generation quality |
| `pdf_urls.json` | arXiv download links | Done its job |

---

## What I Learned Till now

### About the RAG Pipeline
RAG has 4 main stages I'll be building one by one:
1. **Ingestion** — Load PDFs, parse them, chunk them, embed them, store in vector DB
2. **Retrieval** — Take a query, embed it, find similar chunks
3. **Generation** — Feed retrieved chunks + query to an LLM, get an answer
4. **Evaluation** — Measure how good the retrieval and generation actually are

### About the Dataset
- The corpus has multimodal content — text, tables, and images all mixed inside PDFs. This is realistic and hard.
- Queries come in two flavors: **extractive** (answer is literally in the text) and **abstractive** (requires synthesis). The latter is harder for RAG.
- Hard negatives are a real challenge — papers from similar domains that contain similar vocabulary but don't actually answer the query.

### About the Libraries
- **`uv`** — Much faster than pip/conda for package management. Uses `pyproject.toml` for clean dependency tracking.
- **`huggingface_hub`** — Better than `datasets` library for downloading raw files when the schema is inconsistent.
- **`langchain-ollama`** — LangChain's native integration for running local models through Ollama. Clean API.
- **`nomic-embed-text`** — Local embedding model via Ollama. 768-dimensional vectors, fast.

### About Security
Running everything locally via **Ollama** means no data leaves my machine. No API keys for the LLM, no cloud costs, no data privacy concerns. For a project dealing with research papers this is a clean approach.

---

## Challenges & How I Resolved Them

### 1. `load_dataset` crashed with schema mismatch
**Problem:** HuggingFace's `datasets` library tried to parse the corpus JSONs and hit a `pyarrow.lib.ArrowInvalid` error — columns were switching between object and string types across documents.

**Why it happened:** The corpus files have inconsistent schemas because different papers have different structures (some have tables, some don't, etc.).

**Fix:** Switched from `load_dataset()` to `hf_hub_download()` which pulls raw files without trying to parse or validate schema. Lesson: when data is messy, bypass the abstraction and go closer to the metal.

### 2. Duplicate method overriding the fix
**Problem:** Had two `load_pdf_urls()` methods in the same class. Python silently used the last one, which was the old broken version.

**Fix:** Deleted the duplicate. Lesson: always search for duplicate method names when a fix isn't working.

### 3. Conda's base environment overriding uv venv
**Problem:** After activating the uv virtualenv, `which python` still pointed to Anaconda's Python.

**Fix:** Ran `source .venv/bin/activate` explicitly. The `(backend)` prefix in the terminal prompt is not enough to confirm which Python is active — always verify with `which python`.

---

## Current Project Structure

```
backend/
├── data/
│   └── raw/
│       ├── metadata/pdf/arxiv/   ← queries, qrels, answers, pdf_urls
│       └── pdfs/                 ← 1000 arXiv PDFs (downloading)
├── app/
│   └── core/ingestion/
│       └── download.py           ← PDF downloader with resume support
├── .env                          ← local secrets (gitignored)
├── pyproject.toml
├── requirements.txt
└── test_llm.py

```

---

## What's Running Right Now

The full 1,000 PDF download is running in the background. Resume logic is built in — if it crashes, re-running skips already-downloaded files.

---

## Tomorrow's Plan

- Set up PDF parsing (extract text, tables, images from raw PDFs)
- Decide on chunking strategy
- Start building the vector store pipeline