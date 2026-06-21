"""
retrieve.py
Week 2: Hybrid retrieval = dense search (Qdrant) + BM25 keyword search,
combined via Reciprocal Rank Fusion (RRF), then re-ranked with a cross-encoder.
"""

import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

COLLECTION_NAME = "studymate_chunks"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_INDEX_PATH = Path(__file__).parent.parent / "bm25_index.pkl"

DENSE_TOP_K = 20      # how many candidates dense search returns before fusion
BM25_TOP_K = 20        # how many candidates BM25 returns before fusion
RERANK_TOP_K = 5       # final number of chunks sent to the LLM after re-ranking


class HybridRetriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        print("Loading re-ranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        print("Connecting to Qdrant...")
        self.client = QdrantClient(path="./qdrant_data")

        print("Loading BM25 index...")
        self.bm25, self.bm25_chunks = self._load_bm25_index()

    def _load_bm25_index(self):
        """
        Build (or load cached) BM25 index from all chunks currently stored in Qdrant.
        We pull every chunk's text+payload once and index it for keyword search.
        """
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                return pickle.load(f)

        # Pull all points from Qdrant to build the BM25 index
        all_chunks = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            all_chunks.extend(points)
            if offset is None:
                break

        tokenized = [p.payload["text"].lower().split() for p in all_chunks]
        bm25 = BM25Okapi(tokenized)

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump((bm25, all_chunks), f)

        return bm25, all_chunks

    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K):
        query_vector = self.embed_model.encode(query, normalize_embeddings=True).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points
        # Return list of (chunk_text, source, score, unique_key)
        return [
            (r.payload["text"], r.payload["source"], r.score, r.payload["text"][:50])
            for r in results
        ]

    def _bm25_search(self, query: str, top_k: int = BM25_TOP_K):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            (
                self.bm25_chunks[i].payload["text"],
                self.bm25_chunks[i].payload["source"],
                scores[i],
                self.bm25_chunks[i].payload["text"][:50],
            )
            for i in ranked_indices
        ]

    def _fuse_rrf(self, dense_results, bm25_results, k=60):
        """
        Reciprocal Rank Fusion: combine two ranked lists into one,
        using each item's RANK (not raw score) so dense and BM25 scores
        (which are on different scales) can be fairly combined.
        """
        fused_scores = {}
        chunk_lookup = {}

        for rank, (text, source, score, key) in enumerate(dense_results):
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (k + rank + 1)
            chunk_lookup[key] = (text, source)

        for rank, (text, source, score, key) in enumerate(bm25_results):
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (k + rank + 1)
            chunk_lookup[key] = (text, source)

        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        return [(chunk_lookup[k][0], chunk_lookup[k][1]) for k in sorted_keys]

    def _rerank(self, query: str, candidates, top_k=RERANK_TOP_K):
        """Re-rank fused candidates using a cross-encoder for higher precision."""
        pairs = [(query, text) for text, source in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(text, source, float(score)) for (text, source), score in ranked[:top_k]]

    def retrieve(self, query: str, top_k=RERANK_TOP_K):
        dense_results = self._dense_search(query)
        bm25_results = self._bm25_search(query)
        fused = self._fuse_rrf(dense_results, bm25_results)
        reranked = self._rerank(query, fused, top_k=top_k)
        return reranked


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")

    retriever = HybridRetriever()
    results = retriever.retrieve(query)

    print(f"\nQuery: {query}\n")
    for i, (text, source, score) in enumerate(results, 1):
        print(f"--- Result {i} (rerank score: {score:.3f}) | source: {source} ---")
        print(text[:300], "...\n")
