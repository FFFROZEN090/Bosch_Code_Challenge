.PHONY: install ingest log-model serve eval eval-mlflow lint test clean

install:
	pip install -e ".[dev]"

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

clean:
	rm -rf data/faiss_index/*
	rm -rf mlruns/
