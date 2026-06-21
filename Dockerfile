FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Demo dataset only - own course lecture material, NOT the copyrighted
# textbooks (Cormen, Head First series) which stay local-only, never deployed.
COPY data_demo/ ./data_demo/

EXPOSE 8000

CMD ["./entrypoint.sh"]

