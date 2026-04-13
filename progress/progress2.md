# Day 2 — Parsing and Splitting the PDFs

**Date:** April 11, 2026

---

## What I Did Today

Data was already downloaded. Today was about figuring out how to extract clean, usable text from 1,000 arXiv PDFs and split them into chunks that can be embedded and stored.

---

## Questions I Had Going In

- How do I extract text from PDFs that have columns, tables, and images?
- Should I treat every PDF the same way or handle different content types differently?
- How do I split a 50,000 character document into meaningful pieces without losing context?
- Does the splitting strategy actually matter or is it just a detail?

---

## What I Learned About Parsing

Started with `pymupdf4llm` — converts PDFs to markdown. The key thing it does that simpler parsers don't is handle two-column layouts correctly. arXiv papers use two columns and a naive parser reads across both columns producing garbage text. `pymupdf4llm` detects columns and reads them independently.

When I inspected the output I noticed something important — section headings like `**1. Introduction.**` were not converted to `##` markdown headings. They stayed as bold text. This is not a bug in the parser. The PDF itself doesn't encode those as heading elements, so there's nothing for the parser to detect.

I also explored `Docling` (IBM) as an alternative. It runs a layout detection model per page and does a better job of identifying what's a heading vs bold text. The tradeoff is speed — it's significantly slower than `pymupdf4llm` for 1,000 PDFs. I tested both and decided to stick with `pymupdf4llm` for now and account for the heading inconsistency in the splitter instead.

**Key lesson**: parser output quality directly shapes every downstream decision. You can't design a chunking strategy without looking at the actual parsed text first.

### What About Images and Tables?

arXiv papers have three content types — text, tables, and images. `pymupdf4llm` handles text and simple tables well (outputs them as markdown tables). Images are skipped for now.

The plan for images is to add them as Experiment B later — extract images using `fitz`, embed them with wither CLIP or SigLIP (a CLIP alternative from Google that's better on scientific figures), store in a separate ChromaDB collection, and measure whether retrieval scores improve. The `qrels.json` ground truth makes this comparison rigorous.

---

## What I Learned About Chunking

The core problem: embedding models have a context window limit. You can't embed a 50,000 character paper as one vector — it won't fit and even if it did, the vector would be too generic to retrieve specific answers.

The strategy I used is **RecursiveCharacterTextSplitter** with custom separators ordered from most structural to least:

```
\n## → \n### → \n** → \n\n → \n → space
```

The "recursive" part means it tries each separator in order. It always prefers to cut at a heading boundary over a paragraph boundary, a paragraph over a sentence, a sentence over a word. This way chunks respect the document's natural structure as much as possible.

**Why multiple separators matter**: a single separator isn't enough. Split only on `##` and long sections overflow the chunk size. Split only on paragraphs and you ignore section boundaries entirely. The ordered list gives you graceful degradation — it always finds the most meaningful cut available.

**Why overlap matters**: with `chunk_overlap=100`, adjacent chunks share 100 characters. If an answer happens to sit at the boundary between two chunks, it's fully captured in at least one of them. Without overlap you'd slice answers in half. I have kept 20% overlap as a starters.

**Config values**: `chunk_size=512`, `chunk_overlap=100`. These aren't magic numbers — they're a starting point to evaluate against. The `qrels.json` ground truth lets me measure whether different chunk sizes actually improve retrieval.

---

## Challenges and How I Resolved Them

**ModuleNotFoundError for langchain.text_splitter**
The import path changed in newer LangChain versions. `from langchain.text_splitter import ...` no longer works. Fixed by switching to `from langchain_text_splitters import RecursiveCharacterTextSplitter`.

**Section headings not detected as markdown**
Expected `## Introduction` but got `**1. Introduction.**`. Root cause: arXiv authors don't use styled headings in their PDFs — they use bold numbered text. The parser can only extract what's structurally there. Fixed in the splitter by adding `\n**` as a separator so bold sections are still used as split boundaries.

**conda overriding uv virtual environment**
Terminal showed the `(backend)` prefix but `python` still pointed to Anaconda's Python. Fixed by explicitly running `source .venv/bin/activate` and verifying with `which python`.

---

## Architecture Decisions Made Today

- **Config-driven design**: all settings (chunk size, model names, paths, limits) live in `config.json`. No hardcoded values in code. Switching between loaders or changing chunk size is a one-line JSON change.
- **`max_pdfs` limit in config**: lets me test on 3-5 PDFs before committing to 1,000. Same pattern used in the downloader from Day 1.
- **Observability from day one**: every stage saves a JSON stats file to `logs/`. After running the splitter I can see total chunks, avg chunk size, min/max — without reading log files line by line.
- **Preserving both loaders**: `loader.py` (pymupdf4llm) and `loader2.py` (Docling) both exist. The active one is controlled by `config.json`. This lets me compare outputs and document findings.

---

