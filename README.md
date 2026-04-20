# Building a Production-Grade RAG System from Scratch

This is not a tutorial follow-along. I built this to actually understand how Retrieval-Augmented Generation works in the real world — every design decision, every tradeoff, every failure included.

---

## What is RAG and Why Does it Matter?

Large language models are powerful but they have a fundamental problem — they only know what they were trained on. Ask them about a paper published last month, or a document inside your company, and they'll either make something up or tell you they don't know.

RAG fixes this. Instead of relying purely on the model's memory, you:
1. Store your documents in a searchable database
2. When a question comes in, find the most relevant chunks
3. Feed those chunks to the LLM as context
4. Let the model answer using real, retrieved information

The result is a system that can answer questions grounded in your actual data, not hallucinated from training weights.

---

## The Dataset

I used the [Vectara Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) — a dataset purpose-built for evaluating RAG systems.

**What's in it:**
- 1,000 arXiv research papers (PDFs) across all scientific domains
- 3,045 question-answer pairs with ground truth
- 400 positive documents (each answers some queries)
- 600 hard negative documents — papers designed to fool retrieval systems

**Why this dataset?**

Most RAG projects are built and evaluated by vibes. You ask a question, the answer looks reasonable, you move on. This dataset gives you actual ground truth — I know exactly which document and which section should answer each question. That means I can measure retrieval accuracy and generation quality with real numbers.

The multimodal content (text, tables, images all mixed in PDFs) also mirrors what production RAG has to deal with. It's not clean.

---

## Pipeline Overview

```
PDFs → Parse → Chunk → Embed → Vector Store
                                     ↓
Query → Embed → Retrieve → Rerank → Generate → Answer
                                                  ↓
                                             Evaluate
```

| Stage | What happens |
|-------|-------------|
| **Ingestion** | Load PDFs, extract text/tables/images, split into chunks, generate embeddings, store in vector DB |
| **Retrieval** | Embed the query, find semantically similar chunks via vector search |
| **Generation** | Feed retrieved chunks + query to an LLM, generate a grounded answer |
| **Evaluation** | Measure retrieval accuracy (did we find the right chunk?) and generation quality (is the answer faithful?) |

---

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | Flask | Lightweight, explicit, production-deployable |
| LLM | Gemma 4 via Ollama | Fully local — no data leaves the machine |
| Embeddings | nomic-embed-text via Ollama | Local, 768-dim, fast |
| Orchestration | LangChain | Standard RAG tooling |
| Package manager | uv | 10-100x faster than pip, deterministic lockfile |

**On the local-only decision:** Everything runs through Ollama. No API keys, no cloud costs, no data privacy concerns. For a system ingesting research papers this is the right call.

---

## Project Structure

```
project1/
├── backend/
    ├── app/
    │   ├── api/           # Flask endpoints (ingest, query, evaluate)
    │   ├── core/
    │   │   ├── ingestion/ # PDF loading, chunking, embedding
    │   │   ├── retrieval/ # Vector store, retriever
    │   │   ├── generation/# LLM generation
    │   │   └── evaluation/# RAG metrics
    │   └── config.py
    ├── data/
    │   └── raw/
    │       ├── pdfs/      # 1,000 arXiv PDFs
    │       └── metadata/  # queries, qrels, answers
    ├── progress/          # Daily learning logs
    ├── test_llm.py
    └── README.md

```

---

## Progress Logs

I'm keeping daily logs of what I built, what I learned, and what broke. These are honest — failures and fixes included.

- [Day 01 — Project Setup & Dataset Download](./progress/day_01.md)

---

## Key Findings

> This section grows as the project progresses.

- HuggingFace's `datasets` library fails on inconsistent JSON schemas. `hf_hub_download` is more reliable for raw file access.
- arXiv rate limits are real. A 1.5s delay between requests is the minimum to avoid getting blocked.
- Resume logic in download scripts is non-negotiable for large datasets — builds should be restartable.

---

## Running Locally

```bash
# Clone
git clone <repo-url>
cd project1/backend

# Setup environment
uv venv
source .venv/bin/activate
uv sync

# Pull models (requires Ollama installed)
ollama pull gemma4
ollama pull nomic-embed-text

# Configure
cp .env.example .env  # add your settings

# Download dataset
python app/core/ingestion/download.py
```

---

## What's Next

- PDF parsing — extract clean text, tables, images
- Chunking strategy — fixed size vs semantic vs section-based
- Vector store setup
- Retrieval pipeline with evaluation against qrels











Benchmark:
run
"""
benchmark.py — RAG Pipeline Latency Profiler
=============================================
Measures each stage of the pipeline independently so you know
exactly where time is being spent BEFORE making any optimisation.

Stages measured:
  1. Store load          — loading ChromaDB + embedding model into memory
  2. Retrieval           — vector search time
  3. Prompt build        — assembling system + user prompt
  4. LLM TTFT            — time from request start → first token received
  5. LLM total           — full generation time (all tokens)
  6. JSON parse          — extracting structured output from raw LLM response
  7. End-to-end          — wall clock from query in → final result out

Usage:
    python benchmark.py
    python benchmark.py --runs 5 --queries "What is RAG?" "How does attention work?"
"""