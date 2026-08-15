"""Synthetic canonical data.

In production these would come from an ERP and a policy service. They are kept
separate from tools.py so that the evaluator's logic can be read, and changed,
without scrolling past the fixtures it happens to run against.

The rules are already structured: an upstream system turned policy documents into
this. `when` is optional; when absent the rule always applies.
"""

from typing import Any

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
