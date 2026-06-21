"""
run_eval.py
Week 3: Compare baseline (dense-only) retrieval vs. hybrid (dense+BM25+rerank) retrieval.

Metric used: Hit Rate @ K
For each question, check whether the expected_source file appears anywhere
in the top-K retrieved chunks' sources. This is a standard, simple retrieval
quality metric - did we surface the right document at all.

Run: python eval/run_eval.py [path_to_eval_set.json]
If no path given, defaults to eval/qa_set.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from retrieve import HybridRetriever

COLLECTION_NAME = "studymate_chunks"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5

DEFAULT_EVAL_SET_PATH = Path(__file__).parent / "qa_set.json"


def dense_only_search(model, client, query: str, top_k=TOP_K):
    """Baseline: plain dense search, no BM25, no re-ranking."""
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return [r.payload["source"] for r in results]


def hit_rate(retrieved_sources_per_query, acceptable_sources_per_query):
    """Fraction of queries where ANY acceptable source appears in retrieved results."""
    hits = 0
    for retrieved, acceptable in zip(retrieved_sources_per_query, acceptable_sources_per_query):
        if any(src in retrieved for src in acceptable):
            hits += 1
    return hits / len(acceptable_sources_per_query) if acceptable_sources_per_query else 0.0


def mean_reciprocal_rank(retrieved_sources_per_query, acceptable_sources_per_query):
    """
    MRR: for each query, find the rank (1-indexed position) of the FIRST
    acceptable source in the retrieved list. Score = 1/rank for that query
    (1.0 if correct source is #1, 0.5 if it's #2, 0.33 if #3, etc.)
    If no acceptable source appears at all, score = 0 for that query.
    Final MRR = average of these per-query scores.

    Unlike hit_rate (binary hit/miss), MRR rewards ranking the correct
    source HIGHER, so it can still show differences between methods even
    when hit_rate is saturated at 100%.
    """
    reciprocal_ranks = []
    for retrieved, acceptable in zip(retrieved_sources_per_query, acceptable_sources_per_query):
        rank_score = 0.0
        for position, source in enumerate(retrieved, start=1):
            if source in acceptable:
                rank_score = 1.0 / position
                break  # only the FIRST matching position counts
        reciprocal_ranks.append(rank_score)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def main():
    eval_set_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVAL_SET_PATH
    print(f"Using eval set: {eval_set_path}\n")

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    questions = [item["question"] for item in eval_set]
    acceptable_sources_list = [item["acceptable_sources"] for item in eval_set]

    print(f"Loaded {len(questions)} eval questions.\n")

    print("Loading models for baseline (dense-only)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path="./qdrant_data")

    print("Running baseline (dense-only) retrieval...")
    baseline_results = []
    for q in questions:
        sources = dense_only_search(embed_model, client, q, top_k=TOP_K)
        baseline_results.append(sources)

    # IMPORTANT: close this connection before opening another one to the same
    # on-disk folder - Qdrant's local mode only allows one active connection at a time.
    client.close()

    print("Loading hybrid retriever (dense + BM25 + rerank)...")
    hybrid_retriever = HybridRetriever()

    print("Running hybrid retrieval...")
    hybrid_results = []
    for q in questions:
        results = hybrid_retriever.retrieve(q, top_k=TOP_K)
        sources = [source for text, source, score in results]
        hybrid_results.append(sources)

    baseline_hit_rate = hit_rate(baseline_results, acceptable_sources_list)
    hybrid_hit_rate = hit_rate(hybrid_results, acceptable_sources_list)

    baseline_mrr = mean_reciprocal_rank(baseline_results, acceptable_sources_list)
    hybrid_mrr = mean_reciprocal_rank(hybrid_results, acceptable_sources_list)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Baseline (dense-only)  Hit Rate @ {TOP_K}: {baseline_hit_rate:.2%}")
    print(f"Hybrid (dense+BM25+rerank) Hit Rate @ {TOP_K}: {hybrid_hit_rate:.2%}")
    print(f"Hit Rate improvement: {(hybrid_hit_rate - baseline_hit_rate):+.2%}")
    print()
    print(f"Baseline (dense-only)  MRR: {baseline_mrr:.3f}")
    print(f"Hybrid (dense+BM25+rerank) MRR: {hybrid_mrr:.3f}")
    print(f"MRR improvement: {(hybrid_mrr - baseline_mrr):+.3f}")

    print("\n--- Per-question breakdown ---")
    for i, q in enumerate(questions):
        acceptable = acceptable_sources_list[i]
        baseline_hit = any(src in baseline_results[i] for src in acceptable)
        hybrid_hit = any(src in hybrid_results[i] for src in acceptable)

        def first_rank(retrieved):
            for pos, src in enumerate(retrieved, start=1):
                if src in acceptable:
                    return pos
            return None

        b_rank = first_rank(baseline_results[i])
        h_rank = first_rank(hybrid_results[i])
        flag = ""
        if (b_rank is None or (h_rank and h_rank < b_rank)):
            flag = "  <- hybrid ranked it higher" if h_rank else ""
        elif (h_rank is None or (b_rank and b_rank < h_rank)):
            flag = "  <- hybrid ranked it lower" if b_rank else ""
        print(f"[{i+1}] {q[:55]:55s} | baseline rank: {str(b_rank):4s} | hybrid rank: {str(h_rank):4s}{flag}")

    # Save results to a file for your resume/report
    results_summary = {
        "eval_set_used": str(eval_set_path.name),
        "baseline_hit_rate": baseline_hit_rate,
        "hybrid_hit_rate": hybrid_hit_rate,
        "hit_rate_improvement": hybrid_hit_rate - baseline_hit_rate,
        "baseline_mrr": baseline_mrr,
        "hybrid_mrr": hybrid_mrr,
        "mrr_improvement": hybrid_mrr - baseline_mrr,
        "num_questions": len(questions),
        "top_k": TOP_K,
    }
    output_filename = f"eval_results_{eval_set_path.stem}.json"
    with open(Path(__file__).parent / output_filename, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to eval/{output_filename}")


if __name__ == "__main__":
    main()
