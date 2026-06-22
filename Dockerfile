FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user src/ ./src/
COPY --chown=user app.py .
COPY --chown=user entrypoint.sh .
RUN chmod +x entrypoint.sh
COPY --chown=user data_demo/ ./data_demo/
RUN chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"
EXPOSE 7860
CMD ["./entrypoint.sh"]
