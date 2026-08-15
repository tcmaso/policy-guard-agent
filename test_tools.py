"""Tests for the deterministic evaluator. No AWS credentials, no network, no model."""

import pytest

from agent import TOOL_ARGUMENTS, TOOL_HANDLERS, _run_tool
from tools import evaluate_transaction, get_policy_rules, get_transaction


def test_compliant_transaction_is_approved():
    result = evaluate_transaction("TX-1002", "SOFTWARE_PROCUREMENT")
    assert result["decision"] == "APPROVE"


def test_policy_violation_is_blocked():
    result = evaluate_transaction("TX-1001", "SOFTWARE_PROCUREMENT")
    assert result["decision"] == "BLOCK"
    assert any("security review" in reason for reason in result["reasons"])


def test_missing_information_is_referred_for_review():
    result = evaluate_transaction("TX-1003", "CONSULTING_PROCUREMENT")
    assert result["decision"] == "REVIEW"
    assert any("preferred_supplier" in reason for reason in result["reasons"])


def test_rule_below_threshold_does_not_fire():
    # TX-1004 is not a preferred supplier but sits under the £10,000 threshold.
    result = evaluate_transaction("TX-1004", "HARDWARE_PROCUREMENT")
    assert result["decision"] == "APPROVE"


def test_mismatched_policy_category_is_referred_for_review():
    result = evaluate_transaction("TX-1002", "HARDWARE_PROCUREMENT")
    assert result["decision"] == "REVIEW"


def test_unknown_ids_are_rejected():
    with pytest.raises(LookupError):
        get_transaction("TX-9999")
    with pytest.raises(LookupError):
        get_policy_rules(policy_id="NO_SUCH_POLICY")
    with pytest.raises(ValueError):
        get_policy_rules()


def test_policy_lookup_by_category():
    assert get_policy_rules(category="software")["policy_id"] == "SOFTWARE_PROCUREMENT"


# --- Guardrails on what the model is allowed to ask for --------------------------------


def test_unknown_tool_name_is_refused():
    with pytest.raises(ValueError, match="not an available tool"):
        _run_tool("os.system", {"cmd": "whoami"})


def test_authoritative_values_cannot_be_injected():
    # The foundation model may not smuggle a business value past the evaluator.
    with pytest.raises(ValueError, match="does not accept"):
        _run_tool(
            "evaluate_transaction",
            {
                "transaction_id": "TX-1001",
                "policy_id": "SOFTWARE_PROCUREMENT",
                "security_review_completed": True,
            },
        )


def test_allowlists_stay_in_step():
    assert set(TOOL_HANDLERS) == set(TOOL_ARGUMENTS)
