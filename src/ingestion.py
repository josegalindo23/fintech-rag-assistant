import pickle
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.models import DocumentChunk, FinancialCompany

load_dotenv()

# Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Chunking config
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

# Embedding model — free, runs locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# File to company mapping
FILE_MAPPING = {
    "nequi_app_reglamento.txt": (FinancialCompany.NEQUI, "app_reglamento"),
    "nequi_tarjeta_tyc.txt": (FinancialCompany.NEQUI, "tarjeta_tyc"),
    "bold_cuenta_reglamento.txt": (FinancialCompany.BOLD, "cuenta_reglamento"),
    "bold_bolsillos_tyc.txt": (FinancialCompany.BOLD, "bolsillos_tyc"),
    "bold_tarjeta_reglamento.txt": (FinancialCompany.BOLD, "tarjeta_reglamento"),
    "addi_tyc_generales.txt": (FinancialCompany.ADDI, "tyc_generales"),
    "addi_condiciones_credito.txt": (FinancialCompany.ADDI, "condiciones_credito"),
}


def load_raw_document(filepath: Path) -> str:
    """Load and clean raw text from a file."""
    text = filepath.read_text(encoding="utf-8")
    # Remove excessive whitespace and blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text: Full document text.
        chunk_size: Target words per chunk.
        overlap: Words to repeat between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def process_documents() -> list[DocumentChunk]:
    """
    Load all raw documents, split into chunks and return
    a list of DocumentChunk objects with metadata.
    """
    all_chunks: list[DocumentChunk] = []

    for filename, (company, doc_type) in FILE_MAPPING.items():
        filepath = RAW_DATA_DIR / filename

        if not filepath.exists():
            print(f"  Warning: {filename} not found — skipping")
            continue

        print(f"  Processing {filename}...")
        text = load_raw_document(filepath)
        chunks = split_into_chunks(text)

        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=str(uuid4()),
                company=company,
                source_file=filename,
                document_type=doc_type,
                content=chunk_text,
                chunk_index=i,
            )
            all_chunks.append(chunk)

        print(f"    → {len(chunks)} chunks generated")

    return all_chunks


def build_vector_index(chunks: list[DocumentChunk]) -> tuple[faiss.Index, list[DocumentChunk]]:
    """
    Generate embeddings for all chunks and build a FAISS index.

    Args:
        chunks: List of DocumentChunk objects.

    Returns:
        Tuple of (FAISS index, ordered list of chunks).
    """
    print("\n  Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk.content for chunk in chunks]
    print(f"  Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"  Index built — {index.ntotal} vectors, dimension {dimension}")
    return index, chunks


def save_index(index: faiss.Index, chunks: list[DocumentChunk]) -> None:
    """Save FAISS index and chunk metadata to disk."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(PROCESSED_DATA_DIR / "index.faiss"))

    with open(PROCESSED_DATA_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n  Saved index and {len(chunks)} chunks to {PROCESSED_DATA_DIR}")


def load_index() -> tuple[faiss.Index, list[DocumentChunk]]:
    """Load FAISS index and chunk metadata from disk."""
    index = faiss.read_index(str(PROCESSED_DATA_DIR / "index.faiss"))

    with open(PROCESSED_DATA_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def run_ingestion() -> None:
    """Full ingestion pipeline — process, embed and index all documents."""
    print("Starting document ingestion...\n")

    chunks = process_documents()

    if not chunks:
        raise RuntimeError("No documents found in data/raw/ — check file names")

    print(f"\n  Total chunks: {len(chunks)}")

    index, chunks = build_vector_index(chunks)
    save_index(index, chunks)

    print("\nIngestion complete.")


if __name__ == "__main__":
    run_ingestion()