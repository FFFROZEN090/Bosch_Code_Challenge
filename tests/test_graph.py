"""Tests for LangGraph agent pipeline (no LLM calls).

Uses regex routing strategy to avoid requiring a running Ollama instance.
"""

from unittest.mock import patch

import pytest
from me_assistant.ingest.loader import load_all_documents
from me_assistant.ingest.splitter import split_all_documents
from me_assistant.ingest.indexer import build_faiss_index
from me_assistant.agent.nodes import (
    classify_node,
    make_retrieve_single_node,
    make_retrieve_compare_node,
)


@pytest.fixture(scope="module")
def pipeline_data():
    """Build index and chunks for testing nodes."""
    loaded_docs = load_all_documents()
    section_chunks, full_doc_chunks = split_all_documents(loaded_docs)
    index = build_faiss_index(section_chunks)
    return index, full_doc_chunks


@patch("me_assistant.agent.nodes.ROUTING_STRATEGY", "regex")
def test_classify_node_single():
    state = {"question": "What is the RAM of ECU-850?"}
    result = classify_node(state)
    assert result["route"] == "ECU_800"
    assert "ECU-850" in result["matched_models"]
    assert result["route_reason"]


@patch("me_assistant.agent.nodes.ROUTING_STRATEGY", "regex")
def test_classify_node_compare():
    state = {"question": "Compare ECU-750 and ECU-850"}
    result = classify_node(state)
    assert result["route"] == "COMPARE"


def test_retrieve_single_node(pipeline_data):
    index, _ = pipeline_data
    node = make_retrieve_single_node(index)
    state = {
        "question": "temperature range",
        "route": "ECU_700",
        "matched_models": ["ECU-750"],
    }
    result = node(state)
    assert "context" in result
    assert len(result["context"]) > 0
    assert "sources" in result
    assert len(result["sources"]) > 0


def test_retrieve_compare_node(pipeline_data):
    _, full_doc_chunks = pipeline_data
    node = make_retrieve_compare_node(full_doc_chunks)
    state = {"question": "compare all models", "route": "COMPARE"}
    result = node(state)
    assert "context" in result
    assert "ECU-700" in result["context"] or "ECU-750" in result["context"]
    assert len(result["sources"]) == 3
