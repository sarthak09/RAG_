# Day 5 — Latency Profiling and the Thinking Mode Discovery

**Date:** April 19, 2026

---

## What I Was Trying to Do

The full pipeline was working — retrieval, generation, citations. But generation was taking 10-13 seconds per query on an RTX 5080 with 16GB GDDR7. That's too slow for any real application. The goal today was to understand exactly where the time was going and fix it.

Before touching anything, I wanted actual numbers. Not guesses — measurements. So the first thing I built was a benchmark.

---

## What I Built

**`benchmark.py`** — a self-contained latency profiler that runs the full pipeline and measures each stage independently:

- Store load time (one-time cold start cost)
- Retrieval time (vector search in ChromaDB)
- Prompt build time (assembling system + user prompt)
- LLM TTFT — time to first token
- LLM total generation time
- JSON parse time
- End-to-end wall clock

The key design decision: use streaming (`stream=True`) instead of blocking calls to measure TTFT. With `.invoke()` you only see total time. With streaming you record the exact moment the first token arrives — which is what the user actually experiences.

**`llm_fast.py`** — a drop-in replacement for `llm.py` with an identical interface (`LLMGenerator`, same `__init__`, same `generate()` signature) but using `ollama.Client` directly instead of `ChatOllama`. One import line change in `main_run.py` to swap them.

---

## Baseline Numbers

Running the benchmark before any changes:

```
TTFT mean    : 10.9s
E2E mean     : 13.0s
Retrieval    : 0.06s   ← already perfect
LLM % of E2E : 99.6%
```

The retrieval was fine. Everything was the LLM.

---

## What I Learned

### Local LLM vs Cloud tradeoffs

For RAG specifically the key tradeoffs are:

- **Privacy** — local models keep documents on-machine, which matters for research papers and sensitive content
- **Cost** — no per-token charges, which adds up fast with large context windows and batch evaluation
- **Capability** — cloud models (GPT-4o, Claude) are significantly better at following structured output formats like JSON, which matters for citation extraction
- **Context window** — cloud models handle 128K tokens vs a local 8B model's 2-8K, which limits how many chunks you can pass

For this project, local-only is the right call. The dataset is large and I want to run evaluation over thousands of queries without API costs.

### TTFT vs Total generation time

These are two separate numbers that mean different things:

- **TTFT** is how long the model takes to produce its first token — this is the "prefill" phase where the model reads your entire prompt
- **Total time** is TTFT + however long it takes to stream all the output tokens

A long TTFT means the prompt is too long or the model is doing hidden work before answering. A long total-minus-TTFT means the answer itself is long (more tokens to generate).

### Gemma4's thinking mode

Gemma4 has a built-in "thinking" capability. When it detects a complex system prompt — like one asking for structured JSON output — it activates silent reasoning. It spends time generating hidden tokens that work through the problem before producing any visible output. You never see these tokens. They just add latency.

This is useful for math, logic, and multi-step reasoning. It's wasteful for RAG because the answer is already sitting in the context chunks — the model just needs to read and extract it, not reason through uncertainty.

The problem: LangChain's `ChatOllama` has no way to pass `think=False` to the Ollama API. It silently ignores unknown parameters. So every call through LangChain was triggering thinking mode without me knowing.

### LangChain wrappers lag behind Ollama

When Ollama adds a new parameter to their API, LangChain's wrapper takes weeks or months to expose it. `think=False` is a recent addition for Gemma4 and `ChatOllama` doesn't support it yet. The `ollama` Python library (the official Ollama client) exposes it immediately because it's a direct API call.

This is the same pattern as the vector store issue from Day 3 — LangChain abstractions are helpful when you want LangChain to manage the entire flow. When you need precise control over a specific step, use the underlying library directly.

---

## Questions I Had and What I Found

**Q: Can I fix the thinking mode issue inside LangChain without switching libraries?**

Technically yes — `model_kwargs={"think": False}` might work in some `langchain_ollama` versions. But it's unreliable and version-dependent. The `ollama` library is the official client and always reflects the current API. For any parameter that matters precisely (like this one), use the direct library.

**Q: Will disabling thinking mode hurt answer quality?**

No, for RAG. Thinking mode improves accuracy on tasks where the model needs to reason through uncertainty — math problems, logic puzzles, multi-step code. RAG is a reading comprehension task: the answer is in the context, the model just needs to find it. Thinking adds no value and costs 10-20 seconds.

The one exception is multi-hop questions where the answer requires combining information across multiple chunks. For those, thinking might help. But for the benchmark queries, it's pure overhead.

**Q: Why did setting `num_ctx 4096` in the Modelfile make things worse?**

The default `num_ctx` for `gemma4:latest` in Ollama is 2048. Setting it to 4096 doubled the KV cache size — more memory to allocate, more computation per token. Flash Attention helped some, but not enough to overcome a 2x larger context window. Lesson: always check the default before changing a parameter.

---

## Debugging Journey

This session had a lot of wrong turns. Recording them because they're as instructive as the fixes.

### Wrong turn 1 — Assuming the manual `ollama serve` had GPU access

Stopped the systemd Ollama service and ran `ollama serve` manually to test env var changes. The terminal showed `(base)` conda environment and Ollama ran without error, but latency tripled. Reason: manually launched processes don't inherit the CUDA library paths that systemd sets up. The model fell back to CPU silently.

Fix: always use `sudo systemctl edit ollama` to set env vars and restart with systemd.

### Wrong turn 2 — Setting num_ctx 4096 instead of 2048

Created `gemma4-rag` Modelfile with `num_ctx 4096` thinking this would improve generation. It made E2E jump from 13s to 27s. Cause: doubled the KV cache, which outweighed Flash Attention gains.

Fix: `ollama show gemma4:latest` first — always check the existing defaults before overriding.

### Wrong turn 3 — `think false` is not a valid Modelfile PARAMETER

Tried to add `PARAMETER think false` to the Modelfile. Ollama threw `Error: unknown parameter 'think'`. Thinking mode is an API-level parameter, not a model configuration parameter — it belongs in the API call, not the Modelfile.

### The real fix

Replaced `ChatOllama` with `ollama.Client` in both the benchmark and `llm_fast.py`. The `ollama` library sends `think=False` directly in the API payload. Latency dropped immediately.

---

## Final Numbers

```
                  BEFORE          AFTER          IMPROVEMENT
TTFT              10.9s  →        0.21s          52x faster
E2E mean          13.0s  →        3.2s           4x faster
E2E best case     8.2s   →        1.4s           6x faster
Retrieval         0.06s  →        0.05s          unchanged
```

The model is capable of 120 tokens/second on the RTX 5080. With thinking mode disabled and a direct API call, it actually uses that speed.

---

## What's Next

- Apply the `llm_fast.py` fix to the full evaluation pipeline (`evaluator.py`, `llm_judge.py`)
- Implement streaming responses in the Flask API so the frontend shows output as it arrives rather than waiting for the full response
- Semantic caching — cache answers to previously seen queries so repeat questions skip the LLM entirely
- Context compression — reduce the number of tokens sent to the LLM per query to bring E2E closer to 1s

---
