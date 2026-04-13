from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source: str | None = None
    snippet: str


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    model: str
    total_chunks_used: int
    success: bool = True


class FailedRAGResponse(BaseModel):
    query: str
    answer: str = "Unable to generate a response due to an error."
    citations: list[Citation] = []
    model: str
    error: str
    success: bool = False