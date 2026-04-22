import logging
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from data_ret.tracer import PipelineTimer

logger = logging.getLogger("rag.retriever")

def _build_chunks(docs_with_scores: list, scored: bool = True) -> list[dict]:
    chunks = []
    for i, item in enumerate(docs_with_scores):
        if scored:
            doc, score = item
        else:
            doc = item
            score = 0.0
        chunks.append({
            "text":          doc.page_content,
            "score":         round(float(score), 4),
            "doc_id":        doc.metadata.get("doc_id"),
            "chunk_id":      doc.metadata.get("chunk_id") or f"{doc.metadata.get('doc_id')}_chunk_{i}",
            "arxiv_id":      doc.metadata.get("arxiv_id"),
            "source_file":   doc.metadata.get("source_file"),
            "page_number":   doc.metadata.get("page_number"),
            "section_title": doc.metadata.get("section_title"),
            "chunk_index":   doc.metadata.get("chunk_index"),})
    return chunks

class Reranker:
    def __init__(self, model_name: str, top_n: int):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        logger.info(f"Reranker ready | model={model_name} | top_n={top_n}")

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        pairs = [[query, c["text"]] for c in chunks]
        scores = self.model.predict(pairs)
        scored_chunks = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True)
        result = []
        for score, chunk in scored_chunks[:self.top_n]:
            chunk["score"] = round(float(score), 4)
            result.append(chunk)
        return result

class SimpleRetriever:
    def __init__(self, config: dict, store: Chroma):
        cfg = config["retrieval"]
        self.top_k = cfg["top_k"]
        self.use_rerank = cfg.get("use_rerank", False)
        self.store = store
        print("SimpleRetriever initialized")
        if self.use_rerank:
            print("Reranking enabled for SimpleRetriever")
            self.candidate_k = cfg.get("candidate_k", 20)
            self.reranker = Reranker(cfg["rerank_model"], cfg["rerank_top_k"])
            logger.info(
                f"SimpleRetriever | mode=dense+rerank "
                f"| candidate_k={self.candidate_k} "
                f"| rerank_top_k={cfg['rerank_top_k']}")
        else:
            print("Reranking disabled for SimpleRetriever")
            self.candidate_k = self.top_k
            self.reranker    = None
            logger.info(f"SimpleRetriever | mode=dense | top_k={self.top_k}")

    def retrieve(self, query: str, top_k: int = None) -> tuple[list[dict], float]:
        override_k = top_k is not None
        k = top_k if override_k else self.candidate_k
        if not query or not query.strip():
            return [], 0.0
        timer = PipelineTimer()
        try:
            with timer.measure("retrieval"):
                results = self.store.similarity_search_with_score(query, k=k)
            chunks = _build_chunks(results, scored=True)
            if self.use_rerank and not override_k:
                with timer.measure("reranking"):
                    chunks = self.reranker.rerank(query, chunks)
                logger.debug(f"Reranked | before={len(results)} | after={len(chunks)}")
            return chunks, timer.get("retrieval")
        except Exception as e:
            logger.error(f"SimpleRetriever.retrieve failed: {e}")
            raise

class HybridRetriever:
    def __init__(self, config: dict, store: Chroma, bm25_retriever: BM25Retriever):
        cfg = config["retrieval"]
        self.top_k = cfg["top_k"]
        self.use_rerank = cfg.get("use_rerank", False)
        print("HybridRetriever initialized")
        if self.use_rerank:
            print("Reranking enabled for HybridRetriever")
            candidate_k = cfg.get("candidate_k", 20)
            self.reranker = Reranker(cfg["rerank_model"], cfg["rerank_top_k"])
            logger.info(
                f"HybridRetriever | mode=hybrid+rerank "
                f"| candidate_k={candidate_k} "
                f"| rerank_top_k={cfg['rerank_top_k']}")
        else:
            print("Reranking disabled for HybridRetriever")
            candidate_k   = max(self.top_k * 2, 20)
            self.reranker = None
            logger.info(
                f"HybridRetriever | mode=hybrid "
                f"| candidate_k={candidate_k} | top_k={self.top_k}")
        self._candidate_k = candidate_k
        dense_retriever = store.as_retriever(search_kwargs={"k": candidate_k})
        bm25_retriever.k = candidate_k
        self._ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.6, 0.4])

    def retrieve(self, query: str, top_k: int = None) -> tuple[list[dict], float]:
        override_k = top_k is not None
        k = top_k if override_k else self.top_k
        if not query or not query.strip():
            return [], 0.0
        timer = PipelineTimer()
        try:
            with timer.measure("retrieval"):
                results = self._ensemble.invoke(query)
            chunks = _build_chunks(results, scored=False)
            if self.use_rerank and not override_k:
                with timer.measure("reranking"):
                    chunks = self.reranker.rerank(query, chunks)
                logger.debug(f"Reranked | before={len(results)} | after={len(chunks)}")
            else:
                chunks = chunks[:k]
            return chunks, timer.get("retrieval")
        except Exception as e:
            logger.error(f"HybridRetriever.retrieve failed: {e}")
            raise