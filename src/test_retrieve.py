"""
test_retrieve.py
Week 1: Quick sanity check - query -> top-5 relevant chunks via dense search.

Run: python src/test_retrieve.py "your question here"
"""

import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "studymate_chunks"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def search(query: str, top_k: int = 5):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path="./qdrant_data")

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    print(f"\nQuery: {query}\n")
    for i, hit in enumerate(results, 1):
        print(f"--- Result {i} (score: {hit.score:.3f}) | source: {hit.payload['source']} ---")
        print(hit.payload["text"][:300], "...\n")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")
    search(query)
