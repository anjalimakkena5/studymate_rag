FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

# Install CPU-only PyTorch FIRST. Without this, pip pulls the full GPU/CUDA
# build of torch (multiple GB of nvidia-cublas/cudnn/etc.) which is completely
# wasted on Render's free tier (no GPU available) and makes builds extremely slow.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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
