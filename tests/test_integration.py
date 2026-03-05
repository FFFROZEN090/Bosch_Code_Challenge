"""Full pipeline integration tests with mocked LLM.

These tests run the complete LangGraph (classify → retrieve → check_evidence →
validate_confidence → synthesize) with a mocked Ollama call, verifying that
all state fields are populated correctly through the entire flow.

Uses regex routing strategy to avoid requiring a running Ollama instance
for the classify node.
"""

from unittest.mock import patch

import pytest
from me_assistant.ingest.loader import load_all_documents
from me_assistant.ingest.splitter import split_all_documents
from me_assistant.ingest.indexer import build_faiss_index
from me_assistant.agent.graph import build_graph


MOCK_LLM_ANSWER = "The ECU-750 operates from -40°C to +85°C."


@pytest.fixture(scope="module")
def graph():
    """Build the full agent graph with real FAISS index."""
    loaded_docs = load_all_documents()
    section_chunks, full_doc_chunks = split_all_documents(loaded_docs)
    index = build_faiss_index(section_chunks)
    return build_graph(index, full_doc_chunks)


@patch("me_assistant.agent.nodes.ROUTING_STRATEGY", "regex")
@patch("me_assistant.agent.nodes._call_ollama", return_value=MOCK_LLM_ANSWER)
def test_full_pipeline_single_source(mock_ollama, graph):
    """Test full pipeline for a single-source ECU-700 query."""
    state = graph.invoke({"question": "What is the temperature range of ECU-750?"})

    assert state["route"] == "ECU_700"
    assert "ECU-750" in state["matched_models"]
    assert len(state["context"]) > 0
    assert state["answer"] == MOCK_LLM_ANSWER
    assert state["confidence"] > 0.0
    assert state["latency_ms"] >= 0
    assert state["evidence_sufficient"] is True
    assert state["needs_human_review"] is False
    assert len(state["sources"]) > 0

    mock_ollama.assert_called_once()


@patch("me_assistant.agent.nodes.ROUTING_STRATEGY", "regex")
@patch("me_assistant.agent.nodes._call_ollama", return_value="Comparison results here.")
def test_full_pipeline_compare(mock_ollama, graph):
    """Test full pipeline for a comparison query."""
    state = graph.invoke({"question": "Compare ECU-750 and ECU-850 CAN bus speed"})

    assert state["route"] == "COMPARE"
    assert state["answer"] == "Comparison results here."
    assert state["confidence"] >= 0.9  # COMPARE route has high confidence
    assert len(state["sources"]) == 3  # All 3 docs injected
    assert state["evidence_sufficient"] is True

    mock_ollama.assert_called_once()


@patch("me_assistant.agent.nodes.ROUTING_STRATEGY", "regex")
@patch("me_assistant.agent.nodes._call_ollama", return_value="ECU-850 has 2 GB LPDDR4.")
def test_full_pipeline_model_specific(mock_ollama, graph):
    """Test that model-specific queries retrieve from the correct source."""
    state = graph.invoke({"question": "What is the RAM of ECU-850?"})

    assert state["route"] == "ECU_800"
    assert "ECU-850" in state["matched_models"]
    assert state["answer"] == "ECU-850 has 2 GB LPDDR4."
    assert state["confidence"] > 0.0

    # Verify sources reference the correct document
    source_files = [s["source_file"] for s in state["sources"]]
    assert any("ECU-800" in f for f in source_files)

    mock_ollama.assert_called_once()
