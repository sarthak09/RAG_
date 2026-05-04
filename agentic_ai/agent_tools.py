import logging
from typing import Any, Type
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import BaseTool
from sentence_transformers import CrossEncoder
from data_ret.tracer import PipelineTimer

logger = logging.getLogger("rag.agent.tools")

def _doc_to_chunk(doc, score: float = 0.0, index: int = 0) -> dict:
    return {
        "text":          doc.page_content,
        "score":         round(float(score), 4),
        "doc_id":        doc.metadata.get("doc_id"),
        "chunk_id":      doc.metadata.get("chunk_id") or f"{doc.metadata.get('doc_id')}_chunk_{index}",
        "arxiv_id":      doc.metadata.get("arxiv_id"),
        "source_file":   doc.metadata.get("source_file"),
        "page_number":   doc.metadata.get("page_number"),
        "section_title": doc.metadata.get("section_title"),
        "chunk_index":   doc.metadata.get("chunk_index"),
    }

class VectorRetrieveInput(BaseModel):
    query: str = Field(description="Query string to search the vector store")
    top_k: int = Field(default=20, description="Number of chunks to retrieve")

class VectorRetrieverTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "vector_retriever"
    description: str = "Retrieve semantically similar chunks from the vector store for a given query"
    args_schema: Type[BaseModel] = VectorRetrieveInput
    store: Any
    candidate_k: int = 20

    @classmethod
    def from_config(cls, config: dict, store) -> "VectorRetrieverTool":
        instance = cls(store=store, candidate_k=config["agentic"].get("candidate_k", 20))
        logger.info(f"VectorRetrieverTool ready | candidate_k={instance.candidate_k}")
        return instance

    def _run(self, query: str, top_k: int = None) -> tuple[list[dict], float]:
        if not query or not query.strip():
            return [], 0.0
        k = top_k or self.candidate_k
        timer = PipelineTimer()
        try:
            with timer.measure("vector_retrieve"):
                results = self.store.similarity_search_with_score(query, k=k)
            chunks = [_doc_to_chunk(doc, score, i) for i, (doc, score) in enumerate(results)]
            return chunks, timer.get("vector_retrieve")
        except Exception as e:
            logger.error(f"VectorRetrieverTool._run failed: {e}")
            return [], 0.0

class BM25RetrieveInput(BaseModel):
    query: str = Field(description="Query string for BM25 lexical search")
    top_k: int = Field(default=20, description="Number of chunks to retrieve")

class BM25Tool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "bm25_retriever"
    description: str = "Retrieve chunks using BM25 lexical keyword matching"
    args_schema: Type[BaseModel] = BM25RetrieveInput
    bm25: Any
    candidate_k: int = 20

    @classmethod
    def from_config(cls, config: dict, bm25_retriever) -> "BM25Tool":
        instance = cls(bm25=bm25_retriever, candidate_k=config["agentic"].get("candidate_k", 20))
        instance.bm25.k = instance.candidate_k
        logger.info(f"BM25Tool ready | candidate_k={instance.candidate_k}")
        return instance

    def _run(self, query: str, top_k: int = None) -> tuple[list[dict], float]:
        if not query or not query.strip():
            return [], 0.0
        if top_k:
            self.bm25.k = top_k
        timer = PipelineTimer()
        try:
            with timer.measure("bm25_retrieve"):
                docs = self.bm25.invoke(query)
            chunks = [_doc_to_chunk(doc, 0.0, i) for i, doc in enumerate(docs)]
            return chunks, timer.get("bm25_retrieve")
        except Exception as e:
            logger.error(f"BM25Tool._run failed: {e}")
            return [], 0.0

class RerankInput(BaseModel):
    query: str = Field(description="Original user query for relevance scoring")
    chunks: list[dict] = Field(description="List of retrieved chunks to rerank")

class RerankerTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "reranker"
    description: str = "Rerank retrieved chunks by relevance to the query using a cross-encoder model"
    args_schema: Type[BaseModel] = RerankInput
    model: Any
    top_n: int = 5

    @classmethod
    def from_config(cls, config: dict) -> "RerankerTool":
        model_name = config["retrieval"]["rerank_model"]
        top_n = config["agentic"].get("final_top_k", 5)
        logger.info(f"Loading RerankerTool | model={model_name}")
        instance = cls(model=CrossEncoder(model_name), top_n=top_n)
        logger.info(f"RerankerTool ready | top_n={top_n}")
        return instance

    def _run(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        pairs = [[query, c["text"]] for c in chunks]
        scores = self.model.predict(pairs)
        scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in scored[: self.top_n]:
            chunk = dict(chunk)
            chunk["score"] = round(float(score), 4)
            result.append(chunk)
        return result