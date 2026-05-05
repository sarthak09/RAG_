import json
import re
from pathlib import Path

import pymupdf


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pdf_files(pdf_dir: str, max_pdfs: int):
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    return pdf_files[:max_pdfs]


def extract_pages(pdf_file: Path):
    doc = pymupdf.open(str(pdf_file))

    pages = []

    for index, page in enumerate(doc, start=1):
        text = page.get_text() or ""
        pages.append(
            {
                "page_number": index,
                "text": text,
            }
        )

    return pages


def is_heading(line: str):
    line = line.strip()

    if len(line) < 3 or len(line) > 120:
        return False

    patterns = [
        r"^\d+\.\s+[A-Z][A-Za-z0-9 ,:\-\(\)]+",
        r"^\d+\.\d+\.\s+[A-Z][A-Za-z0-9 ,:\-\(\)]+",
        r"^[IVX]+\.\s+[A-Z][A-Za-z0-9 ,:\-\(\)]+",
        r"^(Abstract|Introduction|Background|Methods|Methodology|Experiments|Results|Discussion|Conclusion|References|Appendix)$",
        r"^[A-Z][A-Z\s\-]{4,}$",
    ]

    return any(re.match(pattern, line) for pattern in patterns)


def heading_level(title: str):
    title = title.strip()

    if re.match(r"^\d+\.\d+\.", title):
        return 2

    if re.match(r"^\d+\.", title):
        return 1

    if title.lower() in ["abstract", "introduction", "methods", "methodology", "results", "discussion", "conclusion", "references"]:
        return 1

    return 1


def extract_headings(pages):
    headings = []

    for page in pages:
        page_number = page["page_number"]
        lines = page["text"].splitlines()

        for line in lines:
            clean_line = " ".join(line.strip().split())

            if is_heading(clean_line):
                headings.append(
                    {
                        "title": clean_line,
                        "level": heading_level(clean_line),
                        "start_index": page_number,
                    }
                )

    return headings


def add_end_pages(headings, total_pages: int):
    if not headings:
        return []

    for i, heading in enumerate(headings):
        if i < len(headings) - 1:
            heading["end_index"] = max(
                heading["start_index"],
                headings[i + 1]["start_index"] - 1,
            )
        else:
            heading["end_index"] = total_pages

    return headings


def build_tree(headings):
    root = []

    current_level_1 = None

    for index, heading in enumerate(headings):
        node = {
            "node_id": str(index).zfill(4),
            "title": heading["title"],
            "start_index": heading["start_index"],
            "end_index": heading["end_index"],
            "nodes": [],
        }

        if heading["level"] == 1:
            root.append(node)
            current_level_1 = node

        elif heading["level"] == 2 and current_level_1:
            current_level_1["nodes"].append(node)

        else:
            root.append(node)

    clean_empty_nodes(root)

    return root


def clean_empty_nodes(nodes):
    for node in nodes:
        if node.get("nodes"):
            clean_empty_nodes(node["nodes"])
        else:
            node.pop("nodes", None)


def build_pageindex(pdf_file: Path):
    pages = extract_pages(pdf_file)
    headings = extract_headings(pages)

    if not headings:
        headings = [
            {
                "title": "Document",
                "level": 1,
                "start_index": 1,
                "end_index": len(pages),
            }
        ]
    else:
        headings = add_end_pages(headings, len(pages))

    tree = build_tree(headings)

    return {
        "document_name": pdf_file.name,
        "document_path": str(pdf_file),
        "total_pages": len(pages),
        "index_type": "simple_pageindex",
        "nodes": tree,
    }


def main():
    config = load_config("config.json")

    pdf_dir = config["ingestion"]["pdf_dir"]
    max_pdfs = min(config["ingestion"]["max_pdfs"], 10)

    output_dir = Path("data/pageindex/indexes")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = get_pdf_files(pdf_dir, max_pdfs)

    saved_files = []

    print(f"Found {len(pdf_files)} PDFs")
    print(f"Saving PageIndex files to: {output_dir}")

    for pdf_file in pdf_files:
        try:
            print(f"\nBuilding simple PageIndex for: {pdf_file.name}")

            pageindex = build_pageindex(pdf_file)

            output_file = output_dir / f"{pdf_file.stem}_pageindex.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(pageindex, f, indent=2, ensure_ascii=False)

            saved_files.append(str(output_file))

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Failed: {pdf_file.name}")
            print(f"Error: {e}")

    manifest_path = output_dir / "pageindex_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_files": len(saved_files),
                "files": saved_files,
            },
            f,
            indent=2,
        )

    print("\nDone")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
