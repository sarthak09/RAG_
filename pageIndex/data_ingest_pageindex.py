import os
import sys
import re
import json
import yaml
import time
import logging
import litellm
from pathlib import Path
from datetime import datetime

class PageIndexIngester:
    def __init__(self, config: dict, base_dir: Path):
        self._cfg = config["pageindex"]
        self._base = base_dir
        self._pdf_dir    = self._resolve(self._cfg["pdf_dir"])
        self._output_dir = self._resolve(self._cfg["output_dir"])
        self._log_dir    = self._resolve(self._cfg["log_dir"])
        self._repo_dir   = self._resolve(self._cfg["pageindex_repo"])
        self._model      = self._cfg["model"]
        self._num_ctx    = self._cfg["num_ctx"]
        self._max_tokens = self._cfg["max_tokens_per_group"]
        self._max_pdfs   = self._cfg["max_pdfs"]
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = self._setup_logger()
        self._apply_patches()
        self._client = self._create_client()

    def _resolve(self, relative: str) -> Path:
        return (self._base / relative).resolve()

    def _setup_logger(self) -> logging.Logger:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = self._log_dir / f"page_index_data_ingest_{timestamp}.log"
        logger    = logging.getLogger(f"pageindex_ingest_{timestamp}")
        logger.setLevel(logging.INFO)
        handler   = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        print(f"Log → {log_file}")
        return logger

    def _apply_patches(self):
        os.environ["OPENAI_API_KEY"] = "ollama"
        _orig_loads = json.loads
        def _lenient_loads(s, **kw):
            if isinstance(s, str):
                s = re.sub(r',\s*([}\]])', r'\1', s)
            return _orig_loads(s, **kw)
        json.loads = _lenient_loads
        num_ctx = self._num_ctx
        _orig_completion = litellm.completion
        def _patched_completion(model="", messages=None, **kw):
            if "ollama" in model:
                kw["num_ctx"] = num_ctx
            return _orig_completion(model=model, messages=messages, **kw)
        litellm.completion = _patched_completion
        _orig_acompletion = litellm.acompletion
        async def _patched_acompletion(model="", messages=None, **kw):
            if "ollama" in model:
                kw["num_ctx"] = num_ctx
            return await _orig_acompletion(model=model, messages=messages, **kw)
        litellm.acompletion = _patched_acompletion
        config_path = self._repo_dir / "pageindex" / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["max_page_num_each_node"]  = 3
        cfg["max_token_num_each_node"] = self._max_tokens
        cfg["toc_check_page_num"]      = 8
        cfg["if_add_doc_description"]  = "no"
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
        sys.path.insert(0, str(self._repo_dir))
        from pageindex import PageIndexClient
        self._PageIndexClient = PageIndexClient
        _pi = sys.modules["pageindex.page_index"]
        max_tokens  = self._max_tokens
        _orig_group = _pi.page_list_to_group_text

        def _patched_group(page_contents, token_lengths, max_tokens=max_tokens, overlap_page=1):
            return _orig_group(page_contents, token_lengths, max_tokens=max_tokens, overlap_page=overlap_page)

        _pi.page_list_to_group_text = _patched_group

        async def _skip_verify(page_list, list_result, start_index=1, N=None, model=None):
            return 1.0, []
        _pi.verify_toc = _skip_verify

    def _create_client(self):
        return self._PageIndexClient(
            model=f"ollama_chat/{self._model}",
            workspace=str(self._output_dir))

    def _summary_path(self) -> Path:
        return self._output_dir / "ingestion_summary.json"

    def _get_processed_paths(self) -> set:
        path = self._summary_path()
        if not path.exists():
            return set()
        with open(path) as f:
            data = json.load(f)
        return set(data.get("docs", {}).values())

    def _update_summary(self, doc_id: str, pdf_path: str):
        path = self._summary_path()
        data = {"model": self._model, "total": 0, "docs": {}}
        if path.exists():
            with open(path) as f:
                data = json.load(f)
        data["docs"][doc_id] = pdf_path
        data["total"] = len(data["docs"])
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _index_pdf(self, pdf_path: Path) -> tuple:
        start  = time.time()
        doc_id = self._client.index(str(pdf_path))
        elapsed = round(time.time() - start, 2)
        return doc_id, elapsed

    def _get_pdfs(self, resume: bool, single_file: str) -> list:
        if single_file:
            target = Path(single_file).expanduser().resolve()
            if not target.exists():
                raise FileNotFoundError(f"File not found: {target}")
            return [target]
        all_pdfs = sorted(self._pdf_dir.rglob("*.pdf"))[:self._max_pdfs]
        if resume:
            processed = self._get_processed_paths()
            remaining = [p for p in all_pdfs if str(p) not in processed]
            print(f"Resume: {len(all_pdfs) - len(remaining)} done, {len(remaining)} remaining")
            return remaining
        return all_pdfs

    def run(self, resume: bool = False, single_file: str = None):
        pdfs = self._get_pdfs(resume, single_file)
        total = len(pdfs)
        if total == 0:
            print("Nothing to process.")
            return
        print(f"Processing {total} PDF(s) → {self._output_dir}")
        for i, pdf in enumerate(pdfs, 1):
            print(f"[{i}/{total}] {pdf.name} ...", end=" ", flush=True)
            try:
                doc_id, elapsed = self._index_pdf(pdf)
                self._update_summary(doc_id, str(pdf))
                self._logger.info(f"SUCCESS | {pdf.name} | {elapsed}s")
                print(f"done ({elapsed}s)")
            except Exception as e:
                self._logger.info(f"FAILURE | {pdf.name} | {e}")
                print(f"FAILED")
        print("Ingestion complete.")
