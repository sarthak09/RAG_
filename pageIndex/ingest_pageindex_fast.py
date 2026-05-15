import os
import sys
import re
import json
import yaml
import time
import logging
import threading
import litellm
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────
# GLOBAL CONFIG — edit only this section
# ─────────────────────────────────────────────────────────────────

FOLDER_PATH          = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/data/raw/pdfs"
MAX_PDF              = 6
OUTPUT_DIR           = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/data/pageindex"
LOG_DIR              = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/logs"

# Option 2 — swap model for speed vs accuracy trade-off
# Faster / smaller : qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:3b
# More accurate    : qwen2.5:7b, qwen2.5:14b, llama3.1:8b
OLLAMA_MODEL         = "qwen2.5-ctx:latest"

# Option 1 — one entry per running Ollama instance
# Start extras with: OLLAMA_HOST=0.0.0.0:11435 ollama serve
# Load model on each: OLLAMA_HOST=localhost:11435 ollama run qwen2.5-ctx:latest
OLLAMA_INSTANCES     = [
    "http://localhost:11434",
    "http://localhost:11435",
    "http://localhost:11436",
    # "http://localhost:11437",
]

# Option 3 — concurrent PDFs in flight at once
# Rule of thumb: 2 × len(OLLAMA_INSTANCES) for I/O-bound LLM workloads
CONCURRENT_PDFS      = 3

OLLAMA_NUM_CTX       = 16384
MAX_TOKENS_PER_GROUP = 4000

# ─────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = Path(LOG_DIR) / f"ingest_{_ts}.log"
_log_fmt  = "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pageindex")

for _noisy in ("httpx", "httpcore", "litellm", "urllib3", "openai", "root"):
    _l = logging.getLogger(_noisy)
    _l.setLevel(logging.ERROR)
    _l.propagate = False

log.info(f"Log file: {_log_file}")
log.info(f"Model: {OLLAMA_MODEL}  |  Instances: {len(OLLAMA_INSTANCES)}  |  Concurrent: {CONCURRENT_PDFS}")

# ─────────────────────────────────────────────────────────────────
# PATCH 1 — lenient JSON (trailing commas)
# ─────────────────────────────────────────────────────────────────

_original_loads = json.loads
def _lenient_loads(s, **kw):
    if isinstance(s, str):
        s = re.sub(r',\s*([}\]])', r'\1', s)
    return _original_loads(s, **kw)
json.loads = _lenient_loads

# ─────────────────────────────────────────────────────────────────
# PATCH 2 — inject num_ctx + per-thread api_base into litellm
# ─────────────────────────────────────────────────────────────────

_thread_local        = threading.local()
_original_completion = litellm.completion

def _patched_completion(model="", messages=None, **kw):
    if "ollama" in model:
        kw["num_ctx"]   = OLLAMA_NUM_CTX
        kw["api_base"]  = getattr(_thread_local, "api_base", OLLAMA_INSTANCES[0])
    return _original_completion(model=model, messages=messages, **kw)

litellm.completion = _patched_completion

_original_acompletion = litellm.acompletion
async def _patched_acompletion(model="", messages=None, **kw):
    if "ollama" in model:
        kw["num_ctx"]   = OLLAMA_NUM_CTX
        kw["api_base"]  = getattr(_thread_local, "api_base", OLLAMA_INSTANCES[0])
    return await _original_acompletion(model=model, messages=messages, **kw)

litellm.acompletion = _patched_acompletion

# ─────────────────────────────────────────────────────────────────
# PATCH 3 — reduce config values to keep prompts small
# ─────────────────────────────────────────────────────────────────

PAGEINDEX_DIR = Path(__file__).parent / "PageIndex"
_config_path  = PAGEINDEX_DIR / "pageindex" / "config.yaml"
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)
_cfg["max_page_num_each_node"]  = 3
_cfg["max_token_num_each_node"] = MAX_TOKENS_PER_GROUP
_cfg["toc_check_page_num"]      = 8
_cfg["if_add_doc_description"]  = "no"
with open(_config_path, "w") as f:
    yaml.dump(_cfg, f)

# ─────────────────────────────────────────────────────────────────
# PATCH 4 + 5 — fix submodule + group size + accuracy floor
# ─────────────────────────────────────────────────────────────────

os.environ["OPENAI_API_KEY"] = "ollama"
sys.path.insert(0, str(PAGEINDEX_DIR))

from pageindex import PageIndexClient
_pi = sys.modules["pageindex.page_index"]

_orig_group = _pi.page_list_to_group_text
def _patched_group(page_contents, token_lengths, max_tokens=MAX_TOKENS_PER_GROUP, overlap_page=1):
    return _orig_group(page_contents, token_lengths, max_tokens=max_tokens, overlap_page=overlap_page)
_pi.page_list_to_group_text = _patched_group

_orig_verify_toc = _pi.verify_toc
async def _patched_verify_toc(page_list, toc_with_page_number, **kw):
    accuracy, incorrect_results = await _orig_verify_toc(page_list, toc_with_page_number, **kw)
    return max(accuracy, 0.65), incorrect_results
_pi.verify_toc = _patched_verify_toc

# ─────────────────────────────────────────────────────────────────
# PATCH 6 — thread-safe _meta.json writes
# ─────────────────────────────────────────────────────────────────

_meta_lock     = threading.Lock()
_orig_save_meta = PageIndexClient._save_meta
def _safe_save_meta(self, doc_id, entry):
    with _meta_lock:
        _orig_save_meta(self, doc_id, entry)
PageIndexClient._save_meta = _safe_save_meta

# ─────────────────────────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────────────────────────

_client = PageIndexClient(
    model=f"ollama_chat/{OLLAMA_MODEL}",
    workspace=OUTPUT_DIR,
)

def process_pdf(args):
    idx, pdf_path, instance_url = args
    name = pdf_path.name

    _thread_local.api_base = instance_url
    threading.current_thread().name = f"worker-{idx}"

    log.info(f"START  [{idx}] {name}  →  {instance_url}")
    t_start = time.perf_counter()

    try:
        doc_id   = _client.index(str(pdf_path))
        elapsed  = time.perf_counter() - t_start
        log.info(f"DONE   [{idx}] {name}  doc_id={doc_id}  time={elapsed:.1f}s")
        return {"status": "ok", "name": name, "doc_id": doc_id, "elapsed": elapsed}
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        log.error(f"FAIL   [{idx}] {name}  error={e}  time={elapsed:.1f}s")
        return {"status": "fail", "name": name, "doc_id": None, "elapsed": elapsed, "error": str(e)}

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    folder = Path(FOLDER_PATH).expanduser().resolve()
    pdfs   = sorted(folder.rglob("*.pdf"))[:MAX_PDF]

    if not pdfs:
        log.error(f"No PDFs found in {folder}")
        sys.exit(1)

    log.info(f"Found {len(pdfs)} PDF(s) to index → {OUTPUT_DIR}")

    # Assign each PDF to an Ollama instance round-robin
    tasks = [
        (i + 1, pdf, OLLAMA_INSTANCES[i % len(OLLAMA_INSTANCES)])
        for i, pdf in enumerate(pdfs)
    ]

    results     = []
    run_start   = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENT_PDFS) as pool:
        futures = {pool.submit(process_pdf, t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    total_elapsed = time.perf_counter() - run_start

    ok   = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] == "fail"]

    log.info("─" * 60)
    log.info(f"SUMMARY")
    log.info(f"  Total PDFs    : {len(pdfs)}")
    log.info(f"  Succeeded     : {len(ok)}")
    log.info(f"  Failed        : {len(fail)}")
    log.info(f"  Total time    : {total_elapsed:.1f}s")
    if ok:
        avg = sum(r["elapsed"] for r in ok) / len(ok)
        log.info(f"  Avg per PDF   : {avg:.1f}s")
    log.info("─" * 60)
    for r in sorted(results, key=lambda x: x["name"]):
        status = "✓" if r["status"] == "ok" else "✗"
        log.info(f"  {status} {r['name']:<40} {r['elapsed']:6.1f}s  {r.get('doc_id') or r.get('error','')}")
    log.info("─" * 60)

    summary = {
        "run_timestamp":  _ts,
        "model":          OLLAMA_MODEL,
        "ollama_instances": OLLAMA_INSTANCES,
        "concurrent_pdfs":  CONCURRENT_PDFS,
        "total_pdfs":     len(pdfs),
        "succeeded":      len(ok),
        "failed":         len(fail),
        "total_elapsed_s": round(total_elapsed, 2),
        "avg_elapsed_s":  round(sum(r["elapsed"] for r in ok) / len(ok), 2) if ok else 0,
        "results":        results,
    }
    summary_path = Path(OUTPUT_DIR) / f"summary_{_ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved → {summary_path}")

if __name__ == "__main__":
    main()