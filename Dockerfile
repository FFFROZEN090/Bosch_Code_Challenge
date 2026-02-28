FROM python:3.11-slim

WORKDIR /app

# System deps for building native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache-friendly)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "." 2>/dev/null || \
    pip install --no-cache-dir .

# Copy source code and data
COPY src/ src/
COPY data/docs/ data/docs/
COPY data/test-questions.csv data/
COPY scripts/ scripts/

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Build FAISS index at build time (no LLM needed for this)
RUN python scripts/ingest.py

# Install package in editable mode with full source
RUN pip install --no-cache-dir -e .

EXPOSE 5001

ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
