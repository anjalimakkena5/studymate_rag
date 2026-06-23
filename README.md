---
title: StudyMate RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 📚 StudyMate — Hybrid RAG Study Assistant

**Live Demo:** [anjali018-studymate-rag.hf.space](https://anjali018-studymate-rag.hf.space)

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions over subject lecture notes (Computer Networks, Generative AI) using hybrid retrieval, cross-encoder re-ranking, and grounded LLM generation with citation transparency and hallucination safeguards.

Built end-to-end: ingestion pipeline -> hybrid retrieval -> evaluation harness -> API -> frontend -> containerized deployment.

---

## Why this isn't just another RAG demo

Most RAG tutorials stop at "embed chunks, retrieve top-k, ask an LLM." This project goes further:

- **Hybrid retrieval** (dense embeddings + BM25 keyword search + cross-encoder re-ranking) instead of dense-only search
- **Measured, not assumed, improvement** - a real evaluation harness comparing baseline vs. hybrid retrieval using Hit Rate and Mean Reciprocal Rank (MRR)
- **A faithfulness safeguard** that catches and prevents the LLM from silently answering off its own pretrained knowledge when retrieval confidence is low - verified working both locally and in production (see Findings below)
- **Actually deployed** - live on Hugging Face Spaces via Docker, not just running on localhost

---

## Architecture

```
                    Streamlit Frontend
              (chat UI, citations, scores)
                          |
                        HTTP
                          v
                   FastAPI Backend
        (relevance-threshold safeguard,
         request/response validation)
                          |
        ------------------------------------
        |                 |                |
   Dense Search      BM25 Keyword     Cross-Encoder
   (Qdrant +          Search           Re-ranker
    bge-base)        (rank-bm25)
        |                 |                |
        ------ Reciprocal Rank Fusion ------
                          |
                          v
              Top-K relevant chunks
                          |
                          v
        Groq LLM (Llama 3.1) -> grounded answer
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) |
| Vector store | Qdrant (local/on-disk) |
| Keyword search | BM25 (`rank-bm25`) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Llama 3.1 via Groq API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker, Hugging Face Spaces |

---

## Evaluation Results

Retrieval quality was measured using **Hit Rate@5** and **Mean Reciprocal Rank (MRR)**, comparing a dense-only baseline against the full hybrid pipeline (dense + BM25 + re-ranking).

| Eval Set | Baseline Hit Rate | Hybrid Hit Rate | Baseline MRR | Hybrid MRR |
|---|---|---|---|---|
| Diverse (balanced, multi-subject) | 100% | 100% | 1.00 | 1.00 |
| Terminology-dense (Generative AI) | 86.4% | **100%** | 0.742 | **0.889** |
| Closely-related topics (TCP variants) | 100% | 96.4%* | 0.954 | 0.964 |

\* See Findings below - this single regression was diagnosed, not just reported.

**Headline result:** on terminology-dense technical content, hybrid retrieval improved Hit Rate by **+13.6 percentage points** and MRR by **+0.145** over dense-only search - concrete evidence that BM25's keyword matching meaningfully complements dense embeddings on jargon-heavy text.

---

## Findings worth knowing

Building this surfaced a few non-obvious lessons, documented rather than hidden:

1. **Ground-truth ambiguity in eval design.** An early eval set assumed one "correct" source document per question. In practice, multiple documents covered the same topic (e.g., a course textbook *and* a dedicated handout both explain the same algorithm), which made retrieval look artificially worse than it was. Fixed by allowing multiple acceptable sources per question.

2. **Hit Rate saturates; MRR doesn't.** Once retrieval is good enough that the correct source almost always appears in the top-5, Hit Rate stops being a useful comparison metric. MRR (which rewards *rank*, not just presence) revealed real differences hybrid search alone couldn't show.

3. **Hybrid search isn't a strict upgrade.** On closely related, vocabulary-overlapping topics (e.g., "TCP Reno" vs. "TCP NewReno"), BM25's exact-keyword bias occasionally pulled rank away from the correct source. Diagnosed via direct inspection of retrieved chunks rather than just accepting the aggregate metric.

4. **LLMs can silently bypass weak retrieval.** A casually-phrased query ("what is gen ai") produced a *correct-sounding* answer even though retrieval confidence was low - because the LLM answered from its own pretrained knowledge instead of the provided context. Fixed with a relevance-threshold check that forces an honest "I don't have enough information" response when retrieval quality is too low. Verified working correctly in the live production deployment.

---

## Project Structure

```
studymate_rag/
├── src/
│   ├── ingest.py        # PDF/PPTX -> chunks -> embeddings -> Qdrant
│   ├── retrieve.py      # Hybrid retrieval: dense + BM25 + RRF fusion + re-ranking
│   ├── generate.py      # Grounded answer generation via Groq
│   └── api.py           # FastAPI backend with relevance safeguards
├── eval/
│   ├── run_eval.py      # Hit Rate + MRR evaluation harness
│   └── qa_set_*.json    # Labeled eval sets per subject
├── app.py               # Streamlit frontend
├── Dockerfile           # CPU-optimized, non-root container
├── entrypoint.sh        # Ingestion-on-startup + dual-process launch
└── requirements.txt
```

---

## Running Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your subject PDFs/PPTs to data/
python3 src/ingest.py

# Terminal 1
uvicorn src.api:app --port 8000

# Terminal 2
streamlit run app.py
```

Requires a free Groq API key (console.groq.com) in a `.env` file:
```
GROQ_API_KEY=your_key_here
```

---

## Author

Built by Anjali Makkena, M.Tech, NITK Surathkal.