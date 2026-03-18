"""
src/retriever.py

"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.models import RetrievedChunk
from src.ingestion import EMBEDDING_MODEL, load_index

load_dotenv()


class Retriever:
    """
    Semantic retriever over the fintech document index.
    Converts a query to a vector and finds the most similar chunks.
    """

    def __init__(self, top_k: int = 5):
        """
        Args:
            top_k: Number of chunks to retrieve per query.
        """
        self.top_k = top_k
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index, self.chunks = load_index()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User question in any language.

        Returns:
            List of RetrievedChunk ordered by relevance score.
        """
        # Embed the query with the same model used for documents
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)

        # Search the index
        scores, indices = self.index.search(query_vector, self.top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(chunk=chunk, score=float(score))
            )

        return results

    def retrieve_by_company(
        self,
        query: str,
        company: str,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks filtered to a specific company.

        Args:
            query: User question.
            company: Company name — 'nequi', 'bold', or 'addi'.

        Returns:
            Filtered list of RetrievedChunk.
        """
        all_results = self.retrieve(query)
        return [r for r in all_results if r.chunk.company.value == company]