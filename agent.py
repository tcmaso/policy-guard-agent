"""The agent loop: a Bedrock foundation model driven by a hand-written tool-use loop.

The foundation model chooses which tool it needs. This module validates and
executes the call. The compliance decision itself is produced by
tools.evaluate_transaction and is carried back to the caller untouched — it is
never re-read out of the model's prose.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import boto3

from tools import evaluate_transaction, get_policy_rules, get_transaction

MAX_TOOL_ROUNDS = 5

# Explicit allowlist. A name outside this mapping is never executed.
TOOL_HANDLERS = {
    "get_transaction": get_transaction,
    "get_policy_rules": get_policy_rules,
    "evaluate_transaction": evaluate_transaction,
}

# Explicit argument allowlist. This is what stops the model passing an amount or a
# security-review flag into the authoritative evaluator.
#
# Every entry is empty, and that is the design rather than an oversight. Each tool's
# arguments are determined entirely by the transaction under assessment, which is
# fixed by the caller's request — so there is nothing left for the model to supply,
# and anything it does supply is refused rather than silently ignored.
TOOL_ARGUMENTS = {
    "get_transaction": set(),
    "get_policy_rules": set(),
    "evaluate_transaction": set(),
}

TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "get_transaction",
            "description": (
                "Retrieve the canonical record for the transaction under assessment. "
                "Takes no arguments — the transaction is fixed by the request."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "get_policy_rules",
            "description": (
                "Retrieve the canonical structured policy governing the transaction "
                "under assessment, so that you can explain the outcome in terms of its "
                "rules. Takes no arguments — the policy follows from the transaction."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "evaluate_transaction",
            "description": (
                "Authoritative compliance evaluation of the transaction under assessment. "
                "Returns APPROVE, BLOCK or REVIEW. Takes no arguments — the transaction "
                "is fixed by the request and the governing policy follows from it. Call "
                "this for the decision; never state one of your own."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
]

SYSTEM_PROMPT = """You are the orchestration component of a procurement-policy compliance service.

Use the provided tools to retrieve the transaction under assessment, retrieve structured policy rules and request deterministic policy evaluation.

The transaction under assessment is fixed by the request, and the policy governing it follows from that transaction. You choose neither. None of the tools take arguments — call them with an empty object.

Use get_policy_rules to read the applicable rules so that you can explain the outcome in terms of them. It does not influence the decision.

You are not authorised to independently approve or block transactions.

For an assessment request, gather the required information and use evaluate_transaction for the authoritative decision.

Never invent transaction values or policy rules.
Never override the decision returned by evaluate_transaction.

If required information is unavailable, prefer REVIEW rather than assuming compliance.

Explain the authoritative result concisely."""


class AgentError(RuntimeError):
    """The agent could not complete the assessment."""


@dataclass
class AgentResult:
    explanation: str
    tools_used: list[str] = field(default_factory=list)
    # The verbatim output of the deterministic evaluator, if it ran.
    evaluation: dict[str, Any] | None = None
    # One entry per executed tool call: what the model asked for and what it got back.
    # This is the audit record, and what the demo UI renders.
    trace: list[dict[str, Any]] = field(default_factory=list)


def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])


def _bind(name: str, transaction_id: str) -> dict[str, Any]:
    """Build a tool's arguments from the request, without reference to the model.

    Every argument any tool receives is produced here. transaction_id comes from the
    caller, and the governing policy's category is read off that transaction — so
    neither the subject of an assessment nor the rules it is judged against can be
    chosen by the model.
    """
    if name in ("get_transaction", "evaluate_transaction"):
        return {"transaction_id": transaction_id}
    if name == "get_policy_rules":
        return {"category": get_transaction(transaction_id)["category"]}
    return {}


def _run_tool(name: str, arguments: Any, transaction_id: str) -> dict[str, Any]:
    """Validate a model-requested tool call, then execute the allowlisted function.

    The model chooses which capability to invoke and when. It supplies no arguments
    to any of them; `arguments` is validated to be empty and then discarded in favour
    of the bound values.
    """
    if name not in TOOL_HANDLERS:
        raise ValueError(f"'{name}' is not an available tool.")
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for '{name}' must be an object.")

    unexpected = set(arguments) - TOOL_ARGUMENTS[name]
    if unexpected:
        raise ValueError(
            f"'{name}' does not accept {sorted(unexpected)}. "
            "It takes no arguments; the transaction under assessment and everything "
            "that follows from it are supplied server-side."
        )

    return TOOL_HANDLERS[name](**_bind(name, transaction_id))


def assess(transaction_id: str, question: str) -> AgentResult:
    """Run the tool-use loop until the foundation model produces a final answer."""
    client = _bedrock_client()
    model_id = os.environ["BEDROCK_MODEL_ID"]

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"text": f"Transaction {transaction_id}. {question}"}],
        }
    ]
    tools_used: list[str] = []
    evaluation: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={"tools": TOOL_SPECS},
            inferenceConfig={"maxTokens": 1024, "temperature": 0},
        )

        message = response["output"]["message"]
        messages.append(message)

        if response["stopReason"] != "tool_use":
            text = " ".join(
                block["text"].strip() for block in message["content"] if "text" in block
            )
            return AgentResult(
                explanation=text.strip(),
                tools_used=tools_used,
                evaluation=evaluation,
                trace=trace,
            )

        results = []
        for block in message["content"]:
            if "toolUse" not in block:
                continue
            call = block["toolUse"]
            try:
                output = _run_tool(call["name"], call["input"], transaction_id)
                status = "success"
                # Only record a tool as used once it has actually run.
                tools_used.append(call["name"])
                if call["name"] == "evaluate_transaction":
                    evaluation = output
            # TypeError covers a malformed call, e.g. a required argument omitted.
            # Feeding it back as a tool result lets the model correct itself, rather
            # than escaping the loop as an unhandled 500.
            except (LookupError, ValueError, TypeError) as exc:
                output = {"error": str(exc)}
                status = "error"
            trace.append(
                {
                    "round": round_number,
                    "tool": call["name"],
                    "input": call["input"],
                    "status": status,
                    "output": output,
                }
            )
            results.append(
                {
                    "toolResult": {
                        "toolUseId": call["toolUseId"],
                        "content": [{"json": output}],
                        "status": status,
                    }
                }
            )

        messages.append({"role": "user", "content": results})

    raise AgentError(f"No final answer within {MAX_TOOL_ROUNDS} tool rounds.")
