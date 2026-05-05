import os
import sys
import re
import json
from pathlib import Path
import yaml

FOLDER_PATH = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/data/raw/pdfs"
MAX_PDF      = 2
OLLAMA_MODEL = "qwen2.5:7b"
OUTPUT_DIR   = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/data/pageindex"
 

os.environ["OPENAI_API_KEY"] = "ollama"
 
_original_loads = json.loads
def _lenient_loads(s, **kw):
    if isinstance(s, str):
        s = re.sub(r',\s*([}\]])', r'\1', s)
    return _original_loads(s, **kw)
json.loads = _lenient_loads
 
PAGEINDEX_DIR = Path(__file__).parent / "PageIndex"
config_path   = PAGEINDEX_DIR / "pageindex" / "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)
cfg["max_page_num_each_node"]   = 4
cfg["max_token_num_each_node"]  = 4000
cfg["toc_check_page_num"]       = 10
cfg["if_add_doc_description"]   = "no"
with open(config_path, "w") as f:
    yaml.dump(cfg, f)
 
sys.path.insert(0, str(PAGEINDEX_DIR))
from pageindex import PageIndexClient
 
def main():
    folder = Path(FOLDER_PATH).expanduser().resolve()
    pdfs   = sorted(folder.rglob("*.pdf"))[:MAX_PDF]
 
    if not pdfs:
        print(f"No PDFs found in {folder}")
        sys.exit(1)
 
    print(f"Found {len(pdfs)} PDF(s) to index -> {OUTPUT_DIR}")
 
    client = PageIndexClient(
        model=f"ollama_chat/{OLLAMA_MODEL}",
        workspace=OUTPUT_DIR,
    )
 
    results = {}
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        try:
            doc_id = client.index(str(pdf))
            results[doc_id] = str(pdf)
        except Exception as e:
            print(f"  SKIP: {e}")
 
    summary_path = Path(OUTPUT_DIR) / "ingestion_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": OLLAMA_MODEL, "total": len(results), "docs": results}, f, indent=2)
 
    print(f"\nDone. {len(results)} PDF(s) indexed.")
    print(f"Output: {OUTPUT_DIR}")
 
if __name__ == "__main__":
    main()
 
