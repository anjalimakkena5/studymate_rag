#!/bin/bash
# entrypoint.sh
# Runs ingestion on first startup (if vector data doesn't exist yet),
# then starts BOTH the API (internal, port 8000) and the Streamlit
# frontend (port 7860 - the port Hugging Face Spaces expects to see).

set -e

if [ ! -d "qdrant_data" ] || [ -z "$(ls -A qdrant_data 2>/dev/null)" ]; then
    echo "No existing vector data found. Running ingestion on demo dataset..."
    DATA_DIR=/app/data_demo python src/ingest.py
else
    echo "Existing vector data found, skipping ingestion."
fi

echo "Starting API server in background (internal, port 8000)..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Give the API a moment to start loading models before Streamlit tries to use it
sleep 5

echo "Starting Streamlit frontend (port 7860 - visible to Hugging Face Spaces)..."
exec streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true