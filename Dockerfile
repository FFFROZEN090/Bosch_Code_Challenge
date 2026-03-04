FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy source first (pyproject.toml needs src/ for setuptools package discovery)
COPY pyproject.toml ./
COPY src/ src/
COPY data/docs/ data/docs/
COPY data/test-questions.csv data/
COPY scripts/ scripts/
COPY ui/ ui/

# Install Python dependencies + package
RUN pip install --no-cache-dir -e ".[ui]"

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Build FAISS index at build time (no LLM needed for this)
RUN python scripts/ingest.py

# SERVICE_MODE: "api" (MLflow serving) or "ui" (Streamlit)
ENV SERVICE_MODE=api

EXPOSE 5001 8501

ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
