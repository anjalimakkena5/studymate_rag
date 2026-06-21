# StudyMate RAG — Week 1 Setup

## What this does (Week 1 scope)
- Loads your subject PDFs from `data/`
- Splits them into overlapping text chunks
- Generates embeddings (free, local model: BAAI/bge-base-en-v1.5)
- Stores everything in a local Qdrant vector database (no server/account needed — it's on-disk)
- Lets you test retrieval: ask a question, get the top-5 most relevant chunks

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Add your data
Drop your subject PDFs into the `data/` folder. Example:
```
data/
├── distributed_systems_unit1.pdf
├── distributed_systems_unit2.pdf
└── ml_lecture_notes.pdf
```

## Run ingestion (Week 1, Step 1)
```bash
python src/ingest.py
```
This will:
- Print how many chunks were created per file
- Generate embeddings (first run downloads the model, ~400MB, one-time)
- Store everything in `qdrant_data/` (created automatically)

## Test retrieval (Week 1, Step 2)
```bash
python src/test_retrieve.py "What is the CAP theorem?"
```
Or run without an argument and it'll prompt you for a question.

You should see the top-5 most relevant chunks from your notes, with similarity scores and which source file each came from.

## Notes
- First run of `ingest.py` will download the embedding model — needs internet, only happens once.
- Each time you run `ingest.py`, it wipes and rebuilds the collection from scratch (fine for now — we'll make this incremental later if needed).
- If retrieval looks bad (irrelevant chunks), don't worry yet — Week 2 adds hybrid search + re-ranking which fixes most of this. Week 1 goal is just "does the pipeline run end to end."

## Next (Week 2 preview)
- Add BM25 keyword search alongside this dense search
- Combine both with score fusion
- Add re-ranking
- Wire up Groq API for actual answer generation with citations
