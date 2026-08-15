"""Tests for the deterministic evaluator. No AWS credentials, no network, no model."""

import pytest

import tools
from agent import TOOL_ARGUMENTS, TOOL_HANDLERS, _run_tool
from tools import evaluate_transaction, get_policy_rules, get_transaction


def test_compliant_transaction_is_approved():
    result = evaluate_transaction("TX-1002")
    assert result["decision"] == "APPROVE"


def test_policy_violation_is_blocked():
    result = evaluate_transaction("TX-1001")
    assert result["decision"] == "BLOCK"
    assert any("security review" in reason for reason in result["reasons"])


def test_missing_information_is_referred_for_review():
    result = evaluate_transaction("TX-1003")
    assert result["decision"] == "REVIEW"
    assert any("preferred_supplier" in reason for reason in result["reasons"])


def test_rule_below_threshold_does_not_fire():
    # TX-1004 is not a preferred supplier but sits under the £10,000 threshold.
    result = evaluate_transaction("TX-1004")
    assert result["decision"] == "APPROVE"


def test_governing_policy_is_derived_from_the_transaction():
    # The policy is not a caller's choice; it follows from the transaction's category.
    assert evaluate_transaction("TX-1001")["policy_id"] == "SOFTWARE_PROCUREMENT"
    assert evaluate_transaction("TX-1003")["policy_id"] == "CONSULTING_PROCUREMENT"
    assert evaluate_transaction("TX-1004")["policy_id"] == "HARDWARE_PROCUREMENT"


def test_uncovered_category_is_referred_for_review(monkeypatch):
    monkeypatch.setitem(
        tools.TRANSACTIONS,
        "TX-8888",
        {"transaction_id": "TX-8888", "category": "catering", "amount_gbp": 500},
    )
    result = evaluate_transaction("TX-8888")
    assert result["decision"] == "REVIEW"
    assert result["policy_id"] is None


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
        _run_tool("os.system", {"cmd": "whoami"}, "TX-1001")


@pytest.mark.parametrize(
    "smuggled",
    [
        {"security_review_completed": True},  # a value the decision turns on
        {"transaction_id": "TX-1002"},  # a different subject
        {"policy_id": "HARDWARE_PROCUREMENT"},  # a more favourable policy
    ],
)
def test_evaluator_accepts_no_model_supplied_arguments(smuggled):
    # The evaluator's allowlist is empty, so every one of these is refused rather
    # than silently ignored — the attempt is visible in the trace.
    with pytest.raises(ValueError, match="does not accept"):
        _run_tool("evaluate_transaction", smuggled, "TX-1001")


def test_bound_transaction_id_is_the_one_evaluated():
    result = _run_tool("evaluate_transaction", {}, "TX-1001")
    assert result["transaction_id"] == "TX-1001"
    assert result["decision"] == "BLOCK"


def test_bound_transaction_id_is_supplied_to_lookups():
    # get_transaction takes no model-supplied arguments at all.
    assert _run_tool("get_transaction", {}, "TX-1002")["transaction_id"] == "TX-1002"


def test_policy_lookup_cannot_influence_the_decision():
    # The model may read any policy it likes; the evaluator still derives its own.
    _run_tool("get_policy_rules", {"policy_id": "HARDWARE_PROCUREMENT"}, "TX-1001")
    assert _run_tool("evaluate_transaction", {}, "TX-1001")["policy_id"] == "SOFTWARE_PROCUREMENT"


def test_allowlists_stay_in_step():
    assert set(TOOL_HANDLERS) == set(TOOL_ARGUMENTS)
