from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coding_agents.code_supervisor_agent import CodeSupervisorAgent


# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/code", tags=["Code Analysis"])

_supervisor: CodeSupervisorAgent | None = None


def _get_supervisor() -> CodeSupervisorAgent:
    global _supervisor
    if _supervisor is None:
        _supervisor = CodeSupervisorAgent()
    return _supervisor


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Submission(BaseModel):
    candidate_id: str = Field(..., example="C001")
    code:         str = Field(..., example="def solve(n):\n    return n * 2")


class AnalyzeRequest(BaseModel):
    submissions: list[Submission]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/analyze", summary="Analyse code submissions")
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """
    Accepts a list of submissions.
    Runs AI detection, plagiarism check, and static quality analysis.
    Returns a full report.
    """
    if not req.submissions:
        raise HTTPException(status_code=400, detail="No submissions provided")

    submissions = [s.model_dump() for s in req.submissions]

    try:
        report = _get_supervisor().analyze(submissions=submissions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return report