#!/bin/bash
# entrypoint.sh
# Runs ingestion on first startup (if vector data doesn't exist yet), then starts the API.

set -e

if [ ! -d "qdrant_data" ] || [ -z "$(ls -A qdrant_data 2>/dev/null)" ]; then
    echo "No existing vector data found. Running ingestion on demo dataset..."
    DATA_DIR=/app/data_demo python src/ingest.py
else
    echo "Existing vector data found, skipping ingestion."
fi

echo "Starting API server..."
exec uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"
