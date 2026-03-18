import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from src.models import RAGResponse, RetrievedChunk, FinancialCompany
from src.retriever import Retriever

load_dotenv()

SYSTEM_PROMPT = """You are a financial assistant specialized in Colombian fintech products.
You answer questions about the terms, conditions, and regulations of Nequi, Bold, and Addi.

Rules:
- Answer ONLY based on the provided context. Never invent information.
- If the answer is not in the context, say exactly: "I could not find that information in the available documents."
- Always cite which company and document type your answer comes from.
- Be concise and clear. Users are real people trying to understand financial products.
- Answer in the same language the user asked in (Spanish or English).
- Never give legal or financial advice — only explain what the documents say."""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a readable context block."""
    lines = []
    for i, r in enumerate(chunks, 1):
        lines.append(
            f"[Source {i} — {r.chunk.company.value.upper()} / "
            f"{r.chunk.document_type} / score: {r.score:.2f}]\n"
            f"{r.chunk.content}"
        )
    return "\n\n---\n\n".join(lines)


def _extract_companies(chunks: list[RetrievedChunk]) -> list[FinancialCompany]:
    """Extract unique companies from retrieved chunks."""
    seen = set()
    companies = []
    for r in chunks:
        if r.chunk.company not in seen:
            seen.add(r.chunk.company)
            companies.append(r.chunk.company)
    return companies


def _call_groq(prompt: str) -> str:
    """Call Groq LLM and return the response text."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def _call_openrouter(prompt: str) -> str:
    """Call OpenRouter as fallback and return the response text."""
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model="nvidia/nemotron-super-49b-v1:free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def _generate_answer(question: str, context: str) -> str:
    """Generate answer using provider fallback chain."""
    prompt = f"""Answer the following question using ONLY the context provided below.

Question: {question}

Context:
{context}

Answer:"""

    providers = [
        ("groq", _call_groq),
        ("openrouter", _call_openrouter),
    ]

    errors = []
    for name, fn in providers:
        try:
            return fn(prompt)
        except Exception as e:
            errors.append(f"{name}: {e}")

    raise RuntimeError("All providers failed:\n" + "\n".join(errors))


def answer(
    question: str,
    company_filter: str | None = None,
    top_k: int = 5,
    retriever: Retriever | None = None,
) -> RAGResponse:
    """
    Answer a question about fintech TyC documents using RAG.

    Args:
        question: User question in Spanish or English.
        company_filter: Optional — restrict to 'nequi', 'bold', or 'addi'.
        top_k: Number of chunks to retrieve.
        retriever: Optional pre-loaded retriever — avoids reloading the model.

    Returns:
        RAGResponse with answer, sources and companies referenced.
    """
    if retriever is None:
        retriever = Retriever(top_k=top_k)

    if company_filter:
        chunks = retriever.retrieve_by_company(question, company_filter)
    else:
        chunks = retriever.retrieve(question)

    if not chunks:
        return RAGResponse(
            question=question,
            answer="I could not find relevant information in the available documents.",
            sources=[],
            companies_referenced=[],
        )

    context = _build_context(chunks)
    answer_text = _generate_answer(question, context)

    return RAGResponse(
        question=question,
        answer=answer_text,
        sources=chunks,
        companies_referenced=_extract_companies(chunks),
    )