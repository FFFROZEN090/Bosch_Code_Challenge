#!/usr/bin/env bash
set -e

echo "=== ME Engineering Assistant - Starting ==="

# 1. Wait for Ollama to be reachable
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "Waiting for Ollama at ${OLLAMA_URL} ..."
for i in $(seq 1 30); do
    if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Ollama not reachable after 30s. Exiting."
        exit 1
    fi
    sleep 1
done

# 2. Pull model if not already present
MODEL="${OLLAMA_MODEL:-mistral:7b}"
echo "Ensuring model '${MODEL}' is available..."
curl -sf "${OLLAMA_URL}/api/pull" -d "{\"name\": \"${MODEL}\"}" > /dev/null 2>&1 || true

# 3. Build FAISS index (skip if already exists)
if [ ! -f data/faiss_index/index.faiss ]; then
    echo "Building FAISS index..."
    python scripts/ingest.py
else
    echo "FAISS index already exists, skipping ingest."
fi

# 4. Log model to MLflow and capture run_id
echo "Logging model to MLflow..."
RUN_ID=$(python -m me_assistant.model.log 2>&1 | grep "Model logged to MLflow run:" | awk '{print $NF}')
echo "MLflow run ID: ${RUN_ID}"

# 5. Start MLflow model serving
echo "Starting MLflow model server on port 5001..."
exec mlflow models serve \
    -m "runs:/${RUN_ID}/model" \
    -p 5001 \
    --host 0.0.0.0 \
    --no-conda
