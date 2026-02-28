"""Log the ME Assistant model to MLflow tracking."""

import logging

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from me_assistant.config import (
    FAISS_INDEX_DIR,
    DOCS_DIR,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
)
from me_assistant.model.pyfunc import MEAssistantModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "me-engineering-assistant"


def log_model() -> str:
    """Log the MEAssistantModel to MLflow with artifacts and signature.

    Returns:
        The MLflow run ID.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "question")]),
        outputs=Schema([
            ColSpec("string", "answer"),
            ColSpec("string", "route"),
            ColSpec("string", "sources"),
            ColSpec("double", "confidence"),
            ColSpec("double", "latency_ms"),
        ]),
    )

    artifacts = {
        "faiss_index": str(FAISS_INDEX_DIR),
        "docs": str(DOCS_DIR),
    }

    with mlflow.start_run() as run:
        mlflow.log_params({
            "llm_model": OLLAMA_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_strategy": "hybrid_faiss_fulldoc",
            "router_type": "rule_based",
        })

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MEAssistantModel(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=[
                "langchain>=0.3.0,<0.4",
                "langchain-community>=0.3.0,<0.4",
                "langchain-huggingface>=0.1.0,<0.2",
                "langgraph>=0.2.0,<0.4",
                "faiss-cpu>=1.9.0,<2",
                "sentence-transformers>=3.0.0,<4",
                "pandas>=2.0.0,<3",
                "mlflow>=2.17.0,<3",
            ],
        )

        run_id = run.info.run_id
        logger.info("Model logged to MLflow run: %s", run_id)
        logger.info("Load with: mlflow.pyfunc.load_model('runs:/%s/model')", run_id)

    return run_id


if __name__ == "__main__":
    log_model()
