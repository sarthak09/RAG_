from langchain_chroma import Chroma
from data_ret.tracer import PipelineTimer
from langchain_community.retrievers import BM25Retriever

class SimpleRetriever:
    def __init__(self, config: dict, store: Chroma):
        self.top_k = config["retrieval"]["top_k"]
        self.store = store

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        k = top_k or self.top_k
        if not query or not query.strip():
            return [], 0.0
        timer = PipelineTimer()
        try:
            with timer.measure("retrieval"):
                results = self.store.similarity_search_with_score(query, k=k)
            chunks = [
                {
                    "text": doc.page_content,
                    "score": round(float(score), 4),
                    "doc_id": doc.metadata.get("doc_id"),
                    "chunk_id": doc.metadata.get("chunk_id") or f"{doc.metadata.get('doc_id')}_chunk_{i}",
                    "arxiv_id": doc.metadata.get("arxiv_id"),
                    "source_file": doc.metadata.get("source_file"),
                    "page_number": doc.metadata.get("page_number"),
                    "section_title": doc.metadata.get("section_title"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                }
                for i, (doc, score) in enumerate(results)
            ]
            return chunks, timer.get("retrieval")

        except Exception as e:
            raise