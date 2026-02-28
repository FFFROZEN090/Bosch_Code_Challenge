"""MLflow PythonModel wrapper for the ME Engineering Assistant."""

import json
import logging
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from me_assistant.config import EMBEDDING_MODEL
from me_assistant.ingest.loader import load_document
from me_assistant.ingest.splitter import split_document, create_full_doc_chunk
from me_assistant.agent.graph import build_graph

logger = logging.getLogger(__name__)


class MEAssistantModel(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper around the LangGraph agent pipeline.

    Artifacts expected:
        faiss_index: Directory containing index.faiss and index.pkl.
        docs:        Directory containing the ECU markdown files.

    The model loads the FAISS index and raw docs at startup, builds
    the LangGraph, and runs it for each incoming question.
    """

    def load_context(self, context):
        """Load FAISS index and documents from MLflow artifacts, build graph."""
        # Resolve artifact paths
        faiss_path = context.artifacts["faiss_index"]
        docs_path = context.artifacts["docs"]

        logger.info("Loading FAISS index from %s", faiss_path)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.index = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        logger.info("Loading documents from %s", docs_path)
        docs_dir = Path(docs_path)
        full_doc_chunks = []
        for md_file in sorted(docs_dir.glob("*.md")):
            loaded = load_document(md_file)
            full_doc_chunks.append(create_full_doc_chunk(loaded))

        self.full_doc_chunks = full_doc_chunks

        logger.info("Building LangGraph agent pipeline...")
        self.graph = build_graph(self.index, self.full_doc_chunks)
        logger.info("MEAssistantModel loaded successfully.")

    def predict(self, context, model_input, params=None):
        """Run the agent pipeline for each question.

        Args:
            context: MLflow context (unused during predict).
            model_input: DataFrame with a 'question' column.
            params: Optional parameters (unused).

        Returns:
            DataFrame with columns: answer, route, sources, confidence, latency_ms.
        """
        questions = model_input["question"].tolist()
        results = []

        for question in questions:
            state = self.graph.invoke({"question": question})
            results.append({
                "answer": state.get("answer", ""),
                "route": state.get("route", ""),
                "sources": json.dumps(state.get("sources", [])),
                "confidence": state.get("confidence", 0.0),
                "latency_ms": state.get("latency_ms", 0.0),
            })

        return pd.DataFrame(results)
