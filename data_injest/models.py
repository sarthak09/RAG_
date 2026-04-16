from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    char_count: int
    source_file: str = ""
    arxiv_id: str = ""
    page_number: int = 0
    section_title: str = ""
    total_chunks_in_doc: int = 0
    is_first_chunk: bool = False
    is_last_chunk: bool = False

@dataclass
class EmbeddedChunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    char_count: int
    vector: list[float]
    source_file: str = ""
    arxiv_id: str = ""
    page_number: int = 0
    section_title: str = ""
    total_chunks_in_doc: int = 0
    is_first_chunk: bool = False
    is_last_chunk: bool = False
    embedding_model: str = ""
    chunk_size_config: int = 0
    overlap_config: int = 0
    embedded_at: str = field(default_factory=lambda: datetime.now().isoformat())
