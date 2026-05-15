import sys
import statistics
from pathlib import Path
from collections import defaultdict
import chromadb
from chromadb.config import Settings

RECURSIVE_DIR    = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/data/vector_store/first"
SEMANTIC_DIR     = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/data/vector_store/semantic"
COLLECTION_NAME  = "rag_documents"
FETCH_BATCH_SIZE = 1000


def load_all_metadata(persist_dir: str, collection_name: str) -> list[dict]:
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(name=collection_name)
    total = collection.count()
    print(f"  Connected to: {persist_dir}")
    print(f"  Collection  : {collection_name}")
    print(f"  Total chunks: {total}")

    all_metadata = []
    offset = 0
    while offset < total:
        batch = collection.get(
            limit=FETCH_BATCH_SIZE,
            offset=offset,
            include=["metadatas", "documents"]
        )
        for meta, doc in zip(batch["metadatas"], batch["documents"]):
            meta["text"] = doc
            meta["char_count"] = len(doc)
            all_metadata.append(meta)
        offset += FETCH_BATCH_SIZE
        print(f"  Fetched {min(offset, total)}/{total} chunks...")

    return all_metadata


def compute_stats(metadata: list[dict], label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  COMPUTING STATS: {label}")
    print(f"{'='*60}")

    char_counts = [m["char_count"] for m in metadata]
    doc_ids = set(m["doc_id"] for m in metadata)

    chunks_per_doc = defaultdict(int)
    chars_per_doc  = defaultdict(list)
    chunks_per_page = defaultdict(int)
    section_titles = []

    for m in metadata:
        doc_id = m["doc_id"]
        chunks_per_doc[doc_id] += 1
        chars_per_doc[doc_id].append(m["char_count"])
        chunks_per_page[m.get("page_number", 0)] += 1
        title = m.get("section_title", "")
        if title:
            section_titles.append(title)

    chunks_per_doc_counts = list(chunks_per_doc.values())
    avg_char_per_doc = [statistics.mean(v) for v in chars_per_doc.values()]

    size_buckets = {"<100": 0, "100-300": 0, "300-600": 0, "600-1000": 0, "1000+": 0}
    for c in char_counts:
        if c < 100:
            size_buckets["<100"] += 1
        elif c < 300:
            size_buckets["100-300"] += 1
        elif c < 600:
            size_buckets["300-600"] += 1
        elif c < 1000:
            size_buckets["600-1000"] += 1
        else:
            size_buckets["1000+"] += 1

    return {
        "label": label,
        "total_chunks": len(metadata),
        "total_documents": len(doc_ids),
        "char_counts": char_counts,
        "chunks_per_doc": chunks_per_doc_counts,
        "avg_char_per_doc": avg_char_per_doc,
        "size_buckets": size_buckets,
        "section_titles_found": len(section_titles),
        "chunks_per_page": chunks_per_page,
        "embedding_model": metadata[0].get("embedding_model", "unknown") if metadata else "unknown",
    }


def print_stats(s: dict):
    cc = s["char_counts"]
    cpd = s["chunks_per_doc"]

    print(f"\n{'='*60}")
    print(f"  STRATEGY: {s['label']}")
    print(f"{'='*60}")

    print(f"\n  --- General ---")
    print(f"  Total chunks          : {s['total_chunks']:,}")
    print(f"  Total documents       : {s['total_documents']:,}")
    print(f"  Embedding model       : {s['embedding_model']}")

    print(f"\n  --- Chunk Size (characters) ---")
    print(f"  Min                   : {min(cc):,}")
    print(f"  Max                   : {max(cc):,}")
    print(f"  Mean                  : {statistics.mean(cc):,.1f}")
    print(f"  Median                : {statistics.median(cc):,.1f}")
    print(f"  Std deviation         : {statistics.stdev(cc):,.1f}")
    print(f"  25th percentile       : {sorted(cc)[len(cc)//4]:,}")
    print(f"  75th percentile       : {sorted(cc)[3*len(cc)//4]:,}")

    print(f"\n  --- Chunk Size Distribution ---")
    total = s["total_chunks"]
    for bucket, count in s["size_buckets"].items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<12} : {count:>6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  --- Chunks per Document ---")
    print(f"  Min chunks/doc        : {min(cpd):,}")
    print(f"  Max chunks/doc        : {max(cpd):,}")
    print(f"  Mean chunks/doc       : {statistics.mean(cpd):,.1f}")
    print(f"  Median chunks/doc     : {statistics.median(cpd):,.1f}")
    print(f"  Std deviation         : {statistics.stdev(cpd):,.1f}")

    print(f"\n  --- Section Titles ---")
    print(f"  Chunks with section title : {s['section_titles_found']:,}")
    pct = s["section_titles_found"] / s["total_chunks"] * 100
    print(f"  Coverage                  : {pct:.1f}%")

    sorted_pages = sorted(s["chunks_per_page"].items())
    if sorted_pages:
        busiest_page, busiest_count = max(sorted_pages, key=lambda x: x[1])
        print(f"\n  --- Page Distribution ---")
        print(f"  Unique pages with chunks  : {len(sorted_pages):,}")
        print(f"  Busiest page              : page {busiest_page} ({busiest_count} chunks)")


def print_comparison(r: dict, s: dict):
    print(f"\n{'='*60}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")

    rcc = r["char_counts"]
    scc = s["char_counts"]
    rcpd = r["chunks_per_doc"]
    scpd = s["chunks_per_doc"]

    print(f"\n  {'Metric':<35} {'Recursive':>15} {'Semantic':>15} {'Diff':>12}")
    print(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*12}")

    def row(label, rv, sv, fmt=".1f"):
        diff = sv - rv
        sign = "+" if diff > 0 else ""
        print(f"  {label:<35} {rv:>15{fmt}} {sv:>15{fmt}} {sign}{diff:>{12}{fmt}}")

    def row_int(label, rv, sv):
        diff = sv - rv
        sign = "+" if diff > 0 else ""
        print(f"  {label:<35} {rv:>15,} {sv:>15,} {sign}{diff:>12,}")

    row_int("Total chunks",           r["total_chunks"],                    s["total_chunks"])
    row_int("Total documents",         r["total_documents"],                 s["total_documents"])
    row("Mean chunk size (chars)",  statistics.mean(rcc),                 statistics.mean(scc))
    row("Median chunk size (chars)", statistics.median(rcc),              statistics.median(scc))
    row("Min chunk size (chars)",   min(rcc),                             min(scc))
    row("Max chunk size (chars)",   max(rcc),                             max(scc))
    row("Std dev chunk size",       statistics.stdev(rcc),                statistics.stdev(scc))
    row("Mean chunks/doc",          statistics.mean(rcpd),                statistics.mean(scpd))
    row("Median chunks/doc",        statistics.median(rcpd),              statistics.median(scpd))
    row_int("Section titles found",   r["section_titles_found"],            s["section_titles_found"])

    print(f"\n  --- Chunk Size Distribution Comparison ---")
    print(f"  {'Bucket':<12}  {'Recursive':>12}  {'Semantic':>12}  {'Diff':>10}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    for bucket in r["size_buckets"]:
        rv = r["size_buckets"][bucket]
        sv = s["size_buckets"][bucket]
        diff = sv - rv
        sign = "+" if diff > 0 else ""
        r_pct = rv / r["total_chunks"] * 100
        s_pct = sv / s["total_chunks"] * 100
        print(f"  {bucket:<12}  {rv:>6,} ({r_pct:4.1f}%)  {sv:>6,} ({s_pct:4.1f}%)  {sign}{diff:>10,}")

    print(f"\n  --- Key Observations ---")

    chunk_diff_pct = (s["total_chunks"] - r["total_chunks"]) / r["total_chunks"] * 100
    sign = "more" if chunk_diff_pct > 0 else "fewer"
    print(f"  Semantic produced {abs(chunk_diff_pct):.1f}% {sign} chunks than recursive")

    mean_diff_pct = (statistics.mean(scc) - statistics.mean(rcc)) / statistics.mean(rcc) * 100
    sign = "larger" if mean_diff_pct > 0 else "smaller"
    print(f"  Semantic chunks are {abs(mean_diff_pct):.1f}% {sign} on average")

    r_small = r["size_buckets"]["<100"] / r["total_chunks"] * 100
    s_small = s["size_buckets"]["<100"] / s["total_chunks"] * 100
    print(f"  Tiny chunks (<100 chars): recursive={r_small:.1f}%  semantic={s_small:.1f}%")

    r_large = r["size_buckets"]["1000+"] / r["total_chunks"] * 100
    s_large = s["size_buckets"]["1000+"] / s["total_chunks"] * 100
    print(f"  Large chunks (1000+ chars): recursive={r_large:.1f}%  semantic={s_large:.1f}%")


def main():
    print("\n" + "="*60)
    print("  CHUNKING STRATEGY COMPARISON")
    print("  Recursive  vs  Semantic")
    print("="*60)

    print(f"\n  Paths:")
    print(f"  Recursive : {RECURSIVE_DIR}")
    print(f"  Semantic  : {SEMANTIC_DIR}")

    for path in [RECURSIVE_DIR, SEMANTIC_DIR]:
        if not Path(path).exists():
            print(f"\n  ERROR: path not found: {path}")
            print(f"  Update RECURSIVE_DIR / SEMANTIC_DIR at the top of this file.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  LOADING RECURSIVE VECTOR STORE")
    print(f"{'='*60}")
    recursive_meta = load_all_metadata(RECURSIVE_DIR, COLLECTION_NAME)

    print(f"\n{'='*60}")
    print(f"  LOADING SEMANTIC VECTOR STORE")
    print(f"{'='*60}")
    semantic_meta = load_all_metadata(SEMANTIC_DIR, COLLECTION_NAME)

    recursive_stats = compute_stats(recursive_meta, "RECURSIVE")
    semantic_stats  = compute_stats(semantic_meta,  "SEMANTIC")

    print_stats(recursive_stats)
    print_stats(semantic_stats)
    print_comparison(recursive_stats, semantic_stats)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()