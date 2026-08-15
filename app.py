"""FastAPI surface for PolicyGuard Agent.

This application is both an API provider (the endpoints below) and an API
consumer (Amazon Bedrock, via Boto3, inside agent.py).
"""

from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import agent
import tools

# Populate os.environ from .env for local development. In deployed environments
# there is no .env file, this is a no-op, and Boto3 resolves credentials from the
# instance role as usual.
load_dotenv()

app = FastAPI(title="PolicyGuard Agent", version="1.0.0")


class AssessRequest(BaseModel):
    transaction_id: str = Field(examples=["TX-1001"])
    question: str = Field(
        default="Can this purchase be approved?", examples=["Can this purchase be approved?"]
    )
    # Off by default so the standard response stays minimal. The demo UI turns it on.
    include_trace: bool = False


class AssessResponse(BaseModel):
    transaction_id: str
    decision: str
    policy_id: str | None
    explanation: str
    tools_used: list[str]
    # Populated only when include_trace is set.
    system_prompt: str | None = None
    trace: list[dict] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# exclude_unset keeps the default response exactly as specified: system_prompt and
# trace are omitted entirely unless the caller opted in.
@app.post("/assess", response_model=AssessResponse, response_model_exclude_unset=True)
def assess(request: AssessRequest) -> AssessResponse:
    try:
        tools.get_transaction(request.transaction_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = agent.assess(request.transaction_id, request.question)
    except (ClientError, BotoCoreError) as exc:
        # Never leak the raw Bedrock response to the client.
        raise HTTPException(status_code=502, detail="Language model unavailable.") from exc
    except agent.AgentError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    extras = (
        {"system_prompt": agent.SYSTEM_PROMPT, "trace": result.trace}
        if request.include_trace
        else {}
    )

    # The decision comes from the deterministic evaluator, not from the model's text.
    # If the evaluator never ran, fail safe rather than trusting the explanation.
    if result.evaluation is None:
        return AssessResponse(
            transaction_id=request.transaction_id,
            decision="REVIEW",
            policy_id=None,
            explanation=(
                "No authoritative policy evaluation was produced, so this transaction "
                "is referred for human review."
            ),
            tools_used=result.tools_used,
            **extras,
        )

    return AssessResponse(
        transaction_id=request.transaction_id,
        decision=result.evaluation["decision"],
        policy_id=result.evaluation["policy_id"],
        explanation=result.explanation,
        tools_used=result.tools_used,
        **extras,
    )


# --- Demo console ---------------------------------------------------------------------
# Not part of the compliance API. Delete this route and demo.html to return the
# service to exactly the two endpoints it is specified to expose.


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")
