"""Tests for the rule-based query router."""

import pytest
from me_assistant.agent.router import route_query


# All 10 official test questions with expected routes
TEST_CASES = [
    (1, "What is the maximum operating temperature for the ECU-750?", "ECU_700"),
    (2, "How much RAM does the ECU-850 have?", "ECU_800"),
    (3, "What are the AI capabilities of the ECU-850b?", "ECU_800"),
    (4, "What are the differences between ECU-850 and ECU-850b?", "COMPARE"),
    (5, "Compare the CAN bus capabilities of ECU-750 and ECU-850.", "COMPARE"),
    (6, "What is the power consumption of the ECU-850b under load?", "ECU_800"),
    (7, "Which ECU models support Over-the-Air (OTA) updates?", "COMPARE"),
    (8, "How does the storage capacity compare across all ECU models?", "COMPARE"),
    (9, "Which ECU can operate in the harshest temperature conditions?", "COMPARE"),
    (10, "How do you enable the NPU on the ECU-850b?", "ECU_800"),
]


@pytest.mark.parametrize("qid,question,expected_route", TEST_CASES)
def test_route_all_questions(qid, question, expected_route):
    result = route_query(question)
    assert result.route == expected_route, (
        f"Q{qid}: expected {expected_route}, got {result.route} "
        f"(reason: {result.reason})"
    )


def test_route_unknown_fallback():
    result = route_query("Tell me about automotive systems")
    assert result.route == "UNKNOWN"


def test_route_case_insensitive():
    result = route_query("What is the RAM of ecu-850?")
    assert result.route == "ECU_800"


def test_route_850b_not_matched_as_850():
    result = route_query("Tell me about ECU-850b specs")
    assert result.route == "ECU_800"
    assert "ECU-850b" in result.matched_models


def test_route_explicit_compare_keyword():
    result = route_query("Compare ECU-750 and ECU-850")
    assert result.route == "COMPARE"


def test_route_superlative_with_single_model():
    """Superlative + single model should NOT trigger COMPARE."""
    result = route_query("What is the maximum temperature for ECU-750?")
    assert result.route == "ECU_700"


def test_route_series_level():
    result = route_query("Tell me about the 700 series")
    assert result.route == "ECU_700"


def test_route_850b_typo_with_space():
    """'850 b' (space typo) should match ECU-850b, not ECU-850."""
    result = route_query("Tell me about 850 b specs")
    assert "ECU-850b" in result.matched_models


def test_route_ecu_850b_typo_with_space():
    """'ECU-850 b' (space typo) should match ECU-850b, not ECU-850."""
    result = route_query("What is the power of ECU-850 b?")
    assert "ECU-850b" in result.matched_models
