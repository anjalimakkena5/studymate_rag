"""
api.py
Week 4: FastAPI backend wrapping the RAG pipeline.

Run: uvicorn src.api:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive Swagger UI
"""

import os
import sys
import time
import logging
from pathlib import Path

# Ensure this file's own directory (src/) is on the import path.
# Needed because uvicorn imports this as "src.api", which does NOT
# automatically add src/ itself to sys.path the way running a script directly does.
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from retrieve import HybridRetriever
from generate import generate_answer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("studymate-api")

app = FastAPI(title="StudyMate RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local/dev use; restrict this before any public deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the retriever ONCE at startup, not per-request - loading models is slow.
retriever = None


@app.on_event("startup")
def load_models():
    global retriever
    logger.info("Loading hybrid retriever (this happens once, at startup)...")
    retriever = HybridRetriever()
    logger.info("Models loaded. API ready.")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    source: str
    text_preview: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


@app.get("/")
def root():
    return {"status": "ok", "message": "StudyMate RAG API is running. See /docs for usage."}


@app.get("/health")
def health_check():
    return {"status": "healthy", "retriever_loaded": retriever is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start_time = time.time()

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not yet loaded. Try again shortly.")

    try:
        chunks = retriever.retrieve(request.question, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed internally.")

    # Relevance threshold: filter out INDIVIDUAL chunks with low re-rank scores,
    # not just check the top one. Without this, a strong #1 result can let
    # irrelevant, noisy chunks (negative scores) ride along into the LLM's
    # context and the displayed sources - polluting both the answer and the UI.
    MIN_RELEVANCE_SCORE = 0.0
    chunks = [c for c in chunks if c[2] >= MIN_RELEVANCE_SCORE]

    if not chunks:
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Query: '{request.question[:50]}' | No chunks above relevance threshold")
        return QueryResponse(
            question=request.question,
            answer="I don't have enough information in the provided notes to answer this confidently.",
            sources=[],
            latency_ms=round(latency_ms, 2),
        )

    try:
        answer = generate_answer(request.question, chunks)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Answer generation failed internally.")

    latency_ms = (time.time() - start_time) * 1000

    logger.info(f"Query: '{request.question[:50]}' | Latency: {latency_ms:.0f}ms")

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[
            SourceChunk(source=source, text_preview=text[:150], score=round(score, 3))
            for text, source, score in chunks
        ],
        latency_ms=round(latency_ms, 2),
    )