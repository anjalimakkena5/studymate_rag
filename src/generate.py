"""
generate.py
Week 2: Take retrieved chunks + user query -> call Groq LLM -> grounded answer with citations.

Run: python src/generate.py "your question here"
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

from retrieve import HybridRetriever

load_dotenv()  # reads .env file and loads GROQ_API_KEY into environment

GROQ_MODEL = "llama-3.1-8b-instant"  # fast, free-tier friendly Groq model

SYSTEM_PROMPT = """You are a study assistant that answers questions ONLY using the provided context chunks from the user's subject notes.

Rules:
- If the answer is fully supported by the context, answer clearly and concisely.
- If the context does NOT contain enough information to answer, say so explicitly: "I don't have enough information in the provided notes to answer this confidently." Do NOT make up an answer.
- Keep answers concise and exam-relevant.
- Do NOT include a "Sources:" line in your answer - the system already tracks and displays sources separately and accurately. Just focus on answering the question well.
"""


def build_context_block(chunks):
    """Format retrieved chunks into a numbered context block the LLM can cite from."""
    context = ""
    for i, (text, source, score) in enumerate(chunks, 1):
        context += f"[Chunk {i} | Source: {source}]\n{text}\n\n"
    return context


def generate_answer(query: str, chunks):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    context_block = build_context_block(chunks)

    user_prompt = f"""Context chunks:
{context_block}

Question: {query}

Answer the question using ONLY the context above."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def ask(query: str):
    print("Retrieving relevant chunks...")
    retriever = HybridRetriever()
    chunks = retriever.retrieve(query)

    if not chunks:
        print("No relevant chunks found in your notes for this question.")
        return

    print("Generating answer...\n")
    answer = generate_answer(query, chunks)

    print(f"Question: {query}\n")
    print(f"Answer:\n{answer}\n")

    print("--- Retrieved chunks used (for transparency) ---")
    for i, (text, source, score) in enumerate(chunks, 1):
        print(f"[{i}] {source} (rerank score: {score:.3f})")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")
    ask(query)
