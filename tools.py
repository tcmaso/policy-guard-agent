"""The three tools the foundation model is allowed to call.

The evaluator in this module is the authoritative compliance decision maker.
The foundation model orchestrates; this file decides. The data it decides over
lives in data.py.
"""

from typing import Any

from data import POLICIES, TRANSACTIONS

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


def evaluate_transaction(transaction_id: str) -> dict[str, Any]:
    """Decide APPROVE, BLOCK or REVIEW for a transaction.

    Takes the transaction ID and nothing else, and that ID is itself bound from the
    caller's request rather than supplied by the language model. Both the transaction
    and the policy that governs it are derived here, so there is no input through
    which any caller can influence the outcome: not the amount, not the
    security-review status, and not the choice of policy to be judged against.
    """
    transaction = get_transaction(transaction_id)

    # The governing policy follows from the transaction's category. It is not a
    # choice, so nothing is gained by letting a caller make it — and allowing one
    # would permit shopping for the most favourable policy covering the category.
    try:
        policy = get_policy_rules(category=transaction["category"])
    except LookupError:
        return {
            "transaction_id": transaction_id,
            "policy_id": None,
            "decision": "REVIEW",
            "reasons": [
                f"No procurement policy covers the '{transaction['category']}' "
                "category, so this transaction cannot be assessed automatically."
            ],
        }

    policy_id = policy["policy_id"]
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
