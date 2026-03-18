from src.rag import answer
from src.retriever import Retriever

TEST_QUESTIONS = [
    "¿Cuánto cobra Nequi por transferencias?",
    "¿Qué pasa si no pago mi crédito con Addi?",
    "¿Cuáles son los límites de la cuenta Bold?",
    "What are the requirements to open a Nequi account?",
    "¿Puedo usar Nequi en el exterior?",
]


def print_response(response):
    print(f"\n{'='*60}")
    print(f"Q: {response.question}")
    print(f"\nA: {response.answer}")
    print(f"\nSources ({len(response.sources)}):")
    for r in response.sources:
        print(
            f"  [{r.score:.2f}] {r.chunk.company.value.upper()} / "
            f"{r.chunk.document_type}"
        )
    companies = [c.value for c in response.companies_referenced]
    print(f"\nCompanies: {', '.join(companies)}")


def main():
    print("Fintech RAG Assistant — Testing\n")
    print("Loading retriever...")

    # Load once — reuse for all questions
    retriever = Retriever(top_k=5)
    
    for question in TEST_QUESTIONS:
        response = answer(question, retriever=retriever)
        print_response(response)


if __name__ == "__main__":
    main()