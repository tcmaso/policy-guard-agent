# PolicyGuard Agent

A procurement-policy compliance service for a (fictional) regulated financial
institution. A user asks *"Can transaction TX-1001 be approved?"*; the foundation model decides
what it needs to know, calls approved tools to find out, and a deterministic
Python evaluator makes the actual compliance decision.

## What this demonstrates

- **An agentic loop written from scratch** — no LangChain, no LangGraph, no Bedrock Agents.
- **Manual tool/function calling** over the Amazon Bedrock Converse API via Boto3.
- **Separation of probabilistic orchestration from deterministic decision-making.**
  The foundation model cannot approve or block anything.
- **Deterministic guardrails** — a tool allowlist, an argument allowlist, and a
  fail-safe evaluator, all independently testable without AWS.
- **REST APIs with FastAPI**, and a Pydantic-validated request/response contract.

## Relationship to the existing PolicyGuardAI project

The companion [PolicyGuardAI](https://example.invalid/policy-guard-ai) project covers the
*upstream* half of this domain: retrieval-augmented generation, embeddings, policy
document ingestion, and LLM-based extraction of prose policies into structured rules.

This project deliberately starts where that one finishes. It assumes an upstream
system has already produced validated structured rules, and focuses on the parts
that project does not cover: **agentic orchestration, tool use, REST APIs and
guardrails**. There is no RAG pipeline, no vector database and no embeddings here,
and no code is shared between the two repositories.

## Architecture

```text
Client
  |
  | POST /assess
  v
FastAPI  (app.py)
  |
  v
Agent loop  (agent.py)
  |
  v
Foundation model via Amazon Bedrock (Converse API, Boto3)
  |
  +--> get_transaction()
  |
  +--> get_policy_rules()
  |
  +--> evaluate_transaction()
              |
              v
      Deterministic Python  (tools.py)
              |
              v
    APPROVE / BLOCK / REVIEW
              |
              v
      Foundation model
              |
              v
       Final explanation
```

| File            | Responsibility                                             |
| --------------- | ---------------------------------------------------------- |
| `app.py`        | FastAPI endpoints, request validation, HTTP status mapping |
| `agent.py`      | The Bedrock Converse tool loop, tool validation, dispatch  |
| `tools.py`      | Synthetic data, the three tools, the authoritative evaluator |
| `test_tools.py` | Tests for the evaluator and the guardrails (no AWS needed) |

## The manual tool-use loop

`agent.assess()` is the whole agent. Conceptually:

1. Send the user's question, the system prompt and the three tool schemas to the
   foundation model via `bedrock.converse(...)`.
2. If `stopReason` is not `tool_use`, the model has finished — return its text.
3. Otherwise, for each `toolUse` block the model emitted:
   - check the tool name against the `TOOL_HANDLERS` allowlist;
   - check the argument names against the `TOOL_ARGUMENTS` allowlist;
   - execute the corresponding Python function;
   - append the result as a `toolResult` block.
4. Send the results back and repeat, up to `MAX_TOOL_ROUNDS = 5`.

Nothing the model produces is ever executed as code. There is no `eval()`, no
`exec()`, no shell, and no dynamic import — a tool name outside the allowlist is
an error, not a lookup.

## Why the foundation model does not make the compliance decision

The foundation model decides *what information it needs* and *which capability to
invoke*. Python decides *whether the transaction complies*. That split matters for three reasons:

- **Auditability.** `evaluate_transaction` is an ordinary function. Its output can
  be reproduced, diffed and explained years later; a model's reasoning cannot.
- **Testability.** The compliance logic is covered by `pytest` with no network calls,
  no credentials and no non-determinism.
- **Integrity.** `evaluate_transaction` accepts **identifiers only**. It reloads the
  canonical transaction and policy itself, so the model cannot supply the amount or
  the security-review flag that the decision turns on. The argument allowlist rejects
  any attempt to pass one, and there is a test for exactly that.

The decision returned by `POST /assess` is read from the evaluator's structured
output, never parsed out of the model's prose. If the evaluator never ran, the API
returns `REVIEW` rather than trusting the explanation.

This is also the practical argument for model choice: with the consequential step
pinned to deterministic code, a small, cheap, fast foundation model — Claude Haiku 4.5
here — is doing the only job left: understanding the question and explaining the
result. A larger model free to reason its way through the approval itself would cost
more and audit worse.

## REST API

The service is an API **provider**.

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /assess`

Request:

```json
{ "transaction_id": "TX-1001", "question": "Can this purchase be approved?" }
```

Response:

```json
{
  "transaction_id": "TX-1001",
  "decision": "BLOCK",
  "policy_id": "SOFTWARE_PROCUREMENT",
  "explanation": "The transaction cannot currently be approved because the required security review has not been completed.",
  "tools_used": ["get_transaction", "get_policy_rules", "evaluate_transaction"]
}
```

Set `"include_trace": true` on the request to additionally receive `system_prompt`
and `trace` — one entry per executed tool call, with the arguments the model supplied
and the output Python returned. Omitted entirely when not requested. This is the
audit record, and it is what the demo console renders.

| Status | Meaning                                        |
| ------ | ---------------------------------------------- |
| 200    | Assessment completed                           |
| 404    | Unknown transaction ID                         |
| 422    | Request body failed Pydantic validation        |
| 502    | Bedrock unavailable or rejected the request    |
| 504    | Tool-round limit reached without a final answer |

Raw Bedrock responses are never returned to the client.

## Boto3 and the Bedrock API

The service is also an API **consumer**. Boto3 is the AWS SDK for Python — a client
library that signs and sends HTTPS requests to AWS service APIs and parses the
responses. `boto3.client("bedrock-runtime", ...)` gives a client for the Bedrock
runtime API; `client.converse(...)` calls its Converse operation, which is Bedrock's
model-agnostic messages-and-tools interface.

The Converse API is what makes the manual loop possible: the request carries
`toolConfig` with the tool schemas, and the response carries `stopReason` plus
`toolUse` blocks describing which tool the foundation model wants, and with what
arguments.

Credentials are resolved by Boto3's standard provider chain (environment variables,
shared credentials file, or an IAM role). Nothing is hardcoded:

```python
bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
```

## Installation and running

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure credentials. Copy the template and fill it in:

```bash
cp .env.example .env
```

```ini
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-west-2
BEDROCK_MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
```

`.env` is listed in `.gitignore` and is **never** committed. `app.py` calls
`load_dotenv()`, which populates `os.environ` for local development; in a deployed
environment there is no `.env` file, the call is a no-op, and Boto3 resolves
credentials from the instance role instead. Exported shell variables work equally
well if you prefer not to use a file.

> **Model ID note.** Claude Haiku 4.5 is not available for on-demand throughput
> under its bare model ID; Bedrock requires an *inference profile*. The `global.`
> prefix above routes to the global profile; region-scoped alternatives such as
> `eu.anthropic.claude-haiku-4-5-20251001-v1:0` also work and keep inference within
> that geography. Using the bare `anthropic.claude-haiku-4-5-20251001-v1:0` returns
> a `ValidationException`. The IAM identity needs `bedrock:InvokeModel` on the
> profile and its underlying models.

Run:

```bash
uvicorn app:app --reload
```

Try it:

```bash
curl -X POST http://127.0.0.1:8000/assess \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TX-1001", "question": "Can this purchase be approved?"}'
```

Then open **http://127.0.0.1:8000/ui** for the demo console (below), or
**/docs** for the generated OpenAPI page.

Test:

```bash
pytest -q
```

The tests exercise the evaluator and the guardrails only, so they run with no AWS
credentials and no network access.

## Demo console

`demo.html` is a single self-contained page — no build step, no framework, no CDN —
served at `GET /ui`. Pick a transaction, edit the prompt, and it shows one full
agent run:

- the **system prompt** and the user turn actually sent to the foundation model;
- every **tool call in order**, with the arguments the model chose and the JSON Python
  returned, colour-coded by which side produced it;
- the **authoritative decision**, rendered from `evaluate_transaction`'s output and
  labelled as such, next to the model's explanation labelled *presentation only*.

The point it makes visually: the badge and the prose come from two different places,
and only one of them is authoritative.

`GET /ui` is a demonstration route, not part of the compliance API. Deleting
`demo.html` and the four-line `ui()` handler at the bottom of `app.py` returns the
service to exactly the two endpoints it is specified to expose.

## Synthetic data

Four transactions and three structured policies, all fictional:

| ID      | Category  | Amount   | Situation                                | Decision |
| ------- | --------- | -------- | ---------------------------------------- | -------- |
| TX-1001 | software  | £75,000  | Above threshold, no security review      | BLOCK    |
| TX-1002 | software  | £12,000  | Below threshold, fully compliant         | APPROVE  |
| TX-1003 | consulting | £60,000 | Preferred-supplier status unknown        | REVIEW   |
| TX-1004 | hardware  | £8,500   | Below threshold, so the rule never fires | APPROVE  |

## Production considerations

Deliberately **not** implemented here, but required for a real deployment:

- **Persistent data stores** for transactions and policies, replacing the in-memory dictionaries.
- **Authentication and authorisation** on `/assess`, with per-caller entitlements.
- **Audit logging** of every request, tool call, evaluator input and decision, in immutable storage.
- **Policy versioning**, so a decision can be replayed against the policy in force at the time.
- **Observability** — structured logs, tracing across the tool loop, token and latency metrics.
- **Human review workflows** to route `REVIEW` outcomes to a queue with SLAs and sign-off.
- **Secrets management** via IAM roles or a secrets manager rather than environment variables.
- **Amazon Bedrock Guardrails** for content filtering and prompt-injection defence on the model call.
- **Rate limiting, retries and timeouts** around Bedrock, plus a circuit breaker.
