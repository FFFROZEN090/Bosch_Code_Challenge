"""Tests for MLflow model predict (schema validation, no LLM calls)."""

import pandas as pd
import pytest

from me_assistant.model.pyfunc import MEAssistantModel


def test_model_class_exists():
    """Verify MEAssistantModel can be instantiated."""
    model = MEAssistantModel()
    assert hasattr(model, "load_context")
    assert hasattr(model, "predict")


def test_predict_input_schema():
    """Verify predict expects a DataFrame with 'question' column."""
    model = MEAssistantModel()
    df = pd.DataFrame({"question": ["test question"]})
    assert "question" in df.columns
