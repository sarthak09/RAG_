from pydantic import BaseModel

class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    arxiv_id: str | None = None
    source_file: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    chunk_index: int | None = None
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

class LatencyTrace(BaseModel):
    query: str
    strategy: str = "dense"         
    timestamp: str = ""
    query_encoding_ms: float = 0.0    
    retrieval_ms: float = 0.0        
    reranking_ms: float = 0.0       
    generation_ms: float = 0.0       
    total_ms: float = 0.0            
    top_k_retrieved: int = 0          
    chunks_sent_to_llm: int = 0       
    prompt_char_count: int = 0        
    success: bool = True
    error: str = ""