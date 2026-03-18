"""
src/models.py
"""
from enum import Enum
from pydantic import BaseModel


class FinancialCompany(str, Enum):
    """Supported fintech companies."""
    NEQUI = "nequi"
    BOLD = "bold"
    ADDI = "addi"


class DocumentChunk(BaseModel):
    """Represents a single chunk of a processed document."""
    chunk_id: str
    company: FinancialCompany
    source_file: str
    document_type: str
    content: str
    chunk_index: int


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with its relevance score."""
    chunk: DocumentChunk
    score: float


class RAGResponse(BaseModel):
    """Final response from the RAG pipeline."""
    question: str
    answer: str
    sources: list[RetrievedChunk]
    companies_referenced: list[FinancialCompany]