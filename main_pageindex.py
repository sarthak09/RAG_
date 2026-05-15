import argparse
import json
from pathlib import Path
from pageIndex.data_ingest_pageindex import PageIndexIngester

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="PageIndex Data Ingestion Pipeline")
    parser.add_argument("--config", default="./config.json", help="Path to config.json")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from last successful PDF")
    parser.add_argument("--single", default=None, help="Path to a single PDF to process")
    return parser.parse_args()

def main():
    args       = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    base_dir    = config_path.parent
    config = load_config(config_path)
    ingester = PageIndexIngester(config, base_dir)
    ingester.run(resume=args.resume, single_file=args.single)

if __name__ == "__main__":
    main()
