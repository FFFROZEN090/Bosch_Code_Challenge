#!/usr/bin/env bash
set -e

# ── Banner ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       ME Engineering Assistant — Starting           ║"
echo "║       Mode: ${SERVICE_MODE:-api}                                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── [1/4] Wait for Ollama ───────────────────────────────────
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "[1/4] Waiting for Ollama at ${OLLAMA_URL} ..."

for i in $(seq 1 60); do
    if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "      Ollama is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Ollama not reachable after 60s. Exiting."
        exit 1
    fi
    sleep 1
done

# ── [2/4] Pull model ───────────────────────────────────────
MODEL="${OLLAMA_MODEL:-mistral:7b}"
echo "[2/4] Ensuring model '${MODEL}' is available..."
curl -sf "${OLLAMA_URL}/api/pull" -d "{\"name\": \"${MODEL}\"}" > /dev/null 2>&1 || true
echo "      Model ready."

# ── [3/4] Check FAISS index ────────────────────────────────
echo "[3/4] Checking FAISS index..."
if [ ! -f data/faiss_index/index.faiss ]; then
    echo "      Building FAISS index..."
    python scripts/ingest.py
else
    echo "      FAISS index exists, skipping ingest."
fi

# ── [4/4] Start service ────────────────────────────────────
echo "[4/4] Starting service (mode=${SERVICE_MODE:-api})..."

case "${SERVICE_MODE}" in
    ui)
        echo "      Launching Streamlit UI on port 8501..."
        exec streamlit run ui/app.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --server.headless true \
            --server.enableCORS false \
            --server.enableXsrfProtection false \
            --browser.gatherUsageStats false
        ;;
    api|*)
        echo "      Logging model to MLflow..."
        RUN_ID=$(python -m me_assistant.model.log 2>&1 | grep "Model logged to MLflow run:" | awk '{print $NF}')
        echo "      MLflow run ID: ${RUN_ID}"
        echo "      Launching MLflow model server on port 5001..."
        exec mlflow models serve \
            -m "runs:/${RUN_ID}/model" \
            -p 5001 \
            --host 0.0.0.0 \
            --no-conda
        ;;
esac
