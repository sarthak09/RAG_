import logging
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class SimpleRetriever:
    def __init__(self, config: dict, store: Chroma):
        self.top_k = config["retrieval"]["top_k"]
        self.store = store
        logger.info("Retriever initialised — top_k: %d", self.top_k)

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        k = top_k or self.top_k

        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        try:
            logger.info("Retrieving top-%d chunks for query: '%s'", k, query[:80])
            results = self.store.similarity_search_with_score(query, k=k)

            chunks = [
                {
                    "text":     doc.page_content,
                    "score":    round(float(score), 4),
                    "doc_id":   doc.metadata.get("doc_id"),
                    "chunk_id": doc.metadata.get("chunk_id") or f"{doc.metadata.get('doc_id')}_chunk_{i}",
                    "source":   doc.metadata.get("source_file") or doc.metadata.get("doc_id")
                }
                for i, (doc, score) in enumerate(results)
            ]

            logger.info("Retrieved %d chunks", len(chunks))
            return chunks

        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            raise