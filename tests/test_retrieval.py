"""Tests for retrieval functions."""

import pytest
from me_assistant.ingest.loader import load_all_documents
from me_assistant.ingest.splitter import split_all_documents
from me_assistant.ingest.indexer import build_faiss_index
from me_assistant.retrieval.retriever import (
    retrieve_by_series,
    retrieve_by_model,
    retrieve_all_docs,
)


@pytest.fixture(scope="module")
def test_data():
    """Load docs, build index, and return test fixtures."""
    loaded_docs = load_all_documents()
    section_chunks, full_doc_chunks = split_all_documents(loaded_docs)
    index = build_faiss_index(section_chunks)
    return index, section_chunks, full_doc_chunks


def test_retrieve_by_series_700(test_data):
    index, _, _ = test_data
    docs = retrieve_by_series(index, "operating temperature", "700")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["series"] == "700"


def test_retrieve_by_series_800(test_data):
    index, _, _ = test_data
    docs = retrieve_by_series(index, "RAM memory", "800")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["series"] == "800"


def test_retrieve_by_model_ecu750(test_data):
    index, _, _ = test_data
    docs = retrieve_by_model(index, "temperature range", "ECU-750")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["model"] == "ECU-750"


def test_retrieve_by_model_ecu850b(test_data):
    index, _, _ = test_data
    docs = retrieve_by_model(index, "NPU AI capabilities", "ECU-850b")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["model"] == "ECU-850b"


def test_retrieve_all_docs(test_data):
    _, _, full_doc_chunks = test_data
    text = retrieve_all_docs(full_doc_chunks)
    assert "ECU-700" in text
    assert "ECU-800" in text
    assert "ECU-850b" in text
    assert len(text) > 100


def test_retrieve_no_cross_contamination(test_data):
    """700 series retrieval should never return 800 series chunks."""
    index, _, _ = test_data
    docs = retrieve_by_series(index, "CAN bus interface", "700")
    for doc in docs:
        assert doc.metadata["series"] == "700", (
            f"Cross-contamination: got series={doc.metadata['series']} "
            f"from 700-series query"
        )
