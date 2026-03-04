.PHONY: install ingest log-model serve eval eval-mlflow lint test clean ui \
       docker-up docker-down docker-build docker-logs docker-clean

# ── Local Development ─────────────────────────────────────
install:
	pip install -e ".[dev,ui]"

ingest:
	python scripts/ingest.py

log-model:
	python -m me_assistant.model.log

serve:
	mlflow models serve -m runs:/latest/model -p 5001 --no-conda

eval:
	python scripts/evaluate.py

eval-mlflow:
	python scripts/evaluate.py --mlflow

lint:
	pylint src/me_assistant

test:
	pytest tests/ -v

ui:
	streamlit run ui/app.py --server.port 8501

clean:
	rm -rf data/faiss_index/*
	rm -rf mlruns/

# ── Docker ────────────────────────────────────────────────
docker-up:
	docker compose up --build -d
	@echo ""
	@echo "  ME Engineering Assistant is starting..."
	@echo "  UI:  http://localhost"
	@echo "  API: http://localhost/api/invocations"
	@echo ""
	@echo "  View logs: make docker-logs"

docker-down:
	docker compose down

docker-build:
	docker compose build

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down -v --rmi local
