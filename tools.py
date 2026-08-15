"""Canonical data and the three tools the foundation model is allowed to call.

The evaluator in this module is the authoritative compliance decision maker.
The foundation model orchestrates; this file decides.
"""

from typing import Any

# --------------------------------------------------------------------------------------
# Synthetic data. In production these would come from an ERP and a policy service.
# --------------------------------------------------------------------------------------

TRANSACTIONS: dict[str, dict[str, Any]] = {
    "TX-1001": {
        "transaction_id": "TX-1001",
        "supplier": "Northwind Analytics Ltd",
        "category": "software",
        "amount_gbp": 75000,
        "country": "GB",
        "security_review_completed": False,
        "vendor_risk": "medium",
        "preferred_supplier": True,
    },
    "TX-1002": {
        "transaction_id": "TX-1002",
        "supplier": "Kestrel Software Ltd",
        "category": "software",
        "amount_gbp": 12000,
        "country": "GB",
        "security_review_completed": True,
        "vendor_risk": "low",
        "preferred_supplier": True,
    },
    "TX-1003": {
        "transaction_id": "TX-1003",
        "supplier": "Meridian Advisory Partners",
        "category": "consulting",
        "amount_gbp": 60000,
        "country": "IE",
        "security_review_completed": None,
        "vendor_risk": "low",
        # Supplier onboarding is incomplete, so this is genuinely unknown.
        "preferred_supplier": None,
    },
    "TX-1004": {
        "transaction_id": "TX-1004",
        "supplier": "Harbour IT Supplies",
        "category": "hardware",
        "amount_gbp": 8500,
        "country": "GB",
        "security_review_completed": None,
        "vendor_risk": "low",
        "preferred_supplier": False,
    },
}

# Rules are already structured: an upstream system turned policy documents into this.
# `when` is optional; when absent the rule always applies.
POLICIES: dict[str, dict[str, Any]] = {
    "SOFTWARE_PROCUREMENT": {
        "policy_id": "SOFTWARE_PROCUREMENT",
        "category": "software",
        "rules": [
            {
                "rule_id": "SW-01",
                "when": {"field": "amount_gbp", "operator": "gt", "value": 50000},
                "require": {"field": "security_review_completed", "operator": "is_true"},
                "reason": "Software purchases above £50,000 require a completed security review.",
            },
            {
                "rule_id": "SW-02",
                "when": {"field": "vendor_risk", "operator": "eq", "value": "high"},
                "require": {"field": "preferred_supplier", "operator": "is_true"},
                "reason": "High vendor risk is only acceptable for preferred suppliers.",
            },
        ],
    },
    "CONSULTING_PROCUREMENT": {
        "policy_id": "CONSULTING_PROCUREMENT",
        "category": "consulting",
        "rules": [
            {
                "rule_id": "CON-01",
                "when": {"field": "amount_gbp", "operator": "gt", "value": 25000},
                "require": {"field": "preferred_supplier", "operator": "is_true"},
                "reason": "Consulting spend above £25,000 must go to a preferred supplier.",
            },
            {
                "rule_id": "CON-02",
                "require": {
                    "field": "country",
                    "operator": "in",
                    "value": ["GB", "IE", "DE", "FR", "US"],
                },
                "reason": "Consulting must be delivered from an approved jurisdiction.",
            },
        ],
    },
    "HARDWARE_PROCUREMENT": {
        "policy_id": "HARDWARE_PROCUREMENT",
        "category": "hardware",
        "rules": [
            {
                "rule_id": "HW-01",
                "when": {"field": "amount_gbp", "operator": "gt", "value": 10000},
                "require": {"field": "preferred_supplier", "operator": "is_true"},
                "reason": "Hardware purchases above £10,000 must go to a preferred supplier.",
            },
        ],
    },
}

# Deliberately small and closed. This is not a general expression language, and
# nothing here executes model-supplied strings.
OPERATORS = {
    "gt": lambda actual, expected: actual > expected,
    "gte": lambda actual, expected: actual >= expected,
    "lt": lambda actual, expected: actual < expected,
    "eq": lambda actual, expected: actual == expected,
    "in": lambda actual, expected: actual in expected,
    "is_true": lambda actual, expected: actual is True,
}


# --------------------------------------------------------------------------------------
# Tool 1 and 2: canonical lookups. The model may read these but never invent them.
# --------------------------------------------------------------------------------------


def get_transaction(transaction_id: str) -> dict[str, Any]:
    """Return the canonical transaction record."""
    transaction = TRANSACTIONS.get(transaction_id)
    if transaction is None:
        raise LookupError(f"Unknown transaction '{transaction_id}'.")
    return dict(transaction)


def get_policy_rules(
    category: str | None = None, policy_id: str | None = None
) -> dict[str, Any]:
    """Return the canonical structured policy for a category or a policy ID."""
    if policy_id:
        policy = POLICIES.get(policy_id)
        if policy is None:
            raise LookupError(f"Unknown policy '{policy_id}'.")
        return policy

    if category:
        for policy in POLICIES.values():
            if policy["category"] == category:
                return policy
        raise LookupError(f"No policy covers category '{category}'.")

    raise ValueError("Provide either 'category' or 'policy_id'.")


# --------------------------------------------------------------------------------------
# Tool 3: the authoritative deterministic evaluator.
# --------------------------------------------------------------------------------------


def _check(clause: dict[str, Any], transaction: dict[str, Any]) -> bool | None:
    """Evaluate one clause. None means the transaction lacks the information."""
    actual = transaction.get(clause["field"])
    if actual is None:
        return None
    return OPERATORS[clause["operator"]](actual, clause.get("value"))


def evaluate_transaction(transaction_id: str, policy_id: str) -> dict[str, Any]:
    """Decide APPROVE, BLOCK or REVIEW for a transaction under a policy.

    Only IDs are accepted. The canonical transaction and policy are reloaded here,
    so no caller — including the language model — can supply the amount, the
    security-review status, or any other value the decision turns on.
    """
    transaction = get_transaction(transaction_id)
    policy = get_policy_rules(policy_id=policy_id)

    if transaction["category"] != policy["category"]:
        return {
            "transaction_id": transaction_id,
            "policy_id": policy_id,
            "decision": "REVIEW",
            "reasons": [
                f"Policy {policy_id} covers '{policy['category']}' but the transaction "
                f"is '{transaction['category']}'."
            ],
        }

    blocked: list[str] = []
    review: list[str] = []

    for rule in policy["rules"]:
        applies = True if "when" not in rule else _check(rule["when"], transaction)
        if applies is None:
            review.append(
                f"{rule['rule_id']}: cannot tell whether this rule applies — "
                f"'{rule['when']['field']}' is missing."
            )
            continue
        if not applies:
            continue

        satisfied = _check(rule["require"], transaction)
        if satisfied is None:
            review.append(
                f"{rule['rule_id']}: {rule['reason']} '{rule['require']['field']}' is unknown."
            )
        elif not satisfied:
            blocked.append(f"{rule['rule_id']}: {rule['reason']}")

    # Fail safe: a violation blocks, and anything unknown goes to a human.
    if blocked:
        decision, reasons = "BLOCK", blocked + review
    elif review:
        decision, reasons = "REVIEW", review
    else:
        decision, reasons = "APPROVE", ["All applicable policy rules are satisfied."]

    return {
        "transaction_id": transaction_id,
        "policy_id": policy_id,
        "decision": decision,
        "reasons": reasons,
    }
