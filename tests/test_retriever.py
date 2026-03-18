"""
src/test_retriever.py
"""
from src.retriever import Retriever
from src.models import FinancialCompany


def test_retriever_returns_results():
    """Retriever must return results for a valid query."""
    retriever = Retriever(top_k=3)
    results = retriever.retrieve("¿Cuáles son los costos de la cuenta?")
    assert len(results) > 0
    assert len(results) <= 3


def test_retriever_scores_between_zero_and_one():
    """All relevance scores must be between 0 and 1."""
    retriever = Retriever(top_k=5)
    results = retriever.retrieve("transferencias y límites")
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_retriever_returns_document_chunks():
    """Each result must contain a valid DocumentChunk."""
    retriever = Retriever(top_k=3)
    results = retriever.retrieve("comisiones y tarifas")
    for r in results:
        assert r.chunk.content != ""
        assert r.chunk.company in FinancialCompany


def test_retrieve_by_company_filters_correctly():
    """retrieve_by_company must return only chunks from the requested company."""
    retriever = Retriever(top_k=5)
    results = retriever.retrieve_by_company("cuenta de ahorros", "nequi")
    for r in results:
        assert r.chunk.company.value == "nequi"


def test_results_ordered_by_score():
    """Results must be ordered from highest to lowest score."""
    retriever = Retriever(top_k=5)
    results = retriever.retrieve("requisitos para abrir cuenta")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)