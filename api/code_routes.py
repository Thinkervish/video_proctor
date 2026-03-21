from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coding_agents.code_supervisor_agent import CodeSupervisorAgent


# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/code", tags=["Code Analysis"])

# Supervisor singleton (reads MongoDB URI from environment variable)
_supervisor: CodeSupervisorAgent | None = None


def _get_supervisor() -> CodeSupervisorAgent:
    global _supervisor
    if _supervisor is None:
        _supervisor = CodeSupervisorAgent(
            mongo_uri       = os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            db_name         = os.getenv("MONGO_DB",  "video_proctor"),
            collection_name = os.getenv("MONGO_COLLECTION", "submissions"),
        )
    return _supervisor


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Submission(BaseModel):
    candidate_id: str = Field(..., example="C001")
    code:         str = Field(..., example="def solve(n):\n    return n * 2")


class TestCase(BaseModel):
    input:           str = Field(default="",  example="5\n")
    expected_output: str = Field(default="",  example="120")
    label:           str = Field(default="",  example="n=5")


class DirectAnalyzeRequest(BaseModel):
    submissions:    list[Submission]
    test_cases:     list[TestCase] = Field(default_factory=list)
    entry_function: str            = Field(default="", example="solve")


class ExamAnalyzeRequest(BaseModel):
    exam_id:        str
    question_id:    str
    test_cases:     list[TestCase] = Field(default_factory=list)
    entry_function: str            = Field(default="")
    write_back:     bool           = Field(default=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/analyze/direct", summary="Analyse code submissions directly (no MongoDB)")
async def analyze_direct(req: DirectAnalyzeRequest) -> dict[str, Any]:
    """
    Accepts a list of submissions in the request body.
    Runs AI detection, plagiarism check, and performance evaluation.
    Returns a full analysis report.
    """
    if not req.submissions:
        raise HTTPException(status_code=400, detail="No submissions provided")

    submissions = [s.model_dump() for s in req.submissions]
    test_cases  = [tc.model_dump() for tc in req.test_cases]

    try:
        report = _get_supervisor().analyze_direct(
            submissions=submissions,
            test_cases=test_cases,
            entry_function=req.entry_function,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return report


@router.post("/analyze/exam", summary="Fetch from MongoDB and analyse")
async def analyze_exam(req: ExamAnalyzeRequest) -> dict[str, Any]:
    """
    Fetches all submissions for a given exam_id + question_id from MongoDB,
    then runs the full analysis pipeline.
    """
    test_cases = [tc.model_dump() for tc in req.test_cases]

    try:
        report = _get_supervisor().analyze_question(
            exam_id        = req.exam_id,
            question_id    = req.question_id,
            test_cases     = test_cases,
            entry_function = req.entry_function,
            write_back     = req.write_back,
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="MongoDB (pymongo) is not installed. Use /analyze/direct instead."
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    return report


@router.get(
    "/result/{exam_id}/{question_id}/{candidate_id}",
    summary="Get analysis result for a single candidate",
)
async def get_candidate_result(
    exam_id:      str,
    question_id:  str,
    candidate_id: str,
) -> dict[str, Any]:
    """
    Returns the stored analysis report for one candidate from MongoDB.
    """
    try:
        from pymongo import MongoClient
        col = MongoClient(
            os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            serverSelectionTimeoutMS=3000,
        )[os.getenv("MONGO_DB", "video_proctor")][
            os.getenv("MONGO_COLLECTION", "submissions")
        ]
        doc = col.find_one(
            {
                "exam_id":      exam_id,
                "question_id":  question_id,
                "candidate_id": candidate_id,
            },
            {"_id": 0},
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="pymongo not installed")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"No result found for candidate {candidate_id}"
        )
    return doc


@router.get(
    "/summary/{exam_id}/{question_id}",
    summary="Aggregated summary for all candidates in a question",
)
async def get_exam_summary(exam_id: str, question_id: str) -> dict[str, Any]:
    """
    Returns a high-level summary: total submissions, flagged AI, flagged plagiarism,
    grade distribution.
    """
    try:
        from pymongo import MongoClient
        col = MongoClient(
            os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            serverSelectionTimeoutMS=3000,
        )[os.getenv("MONGO_DB", "video_proctor")][
            os.getenv("MONGO_COLLECTION", "submissions")
        ]
        docs = list(col.find(
            {"exam_id": exam_id, "question_id": question_id, "analysis": {"$exists": True}},
            {"_id": 0, "candidate_id": 1, "analysis": 1},
        ))
    except ImportError:
        raise HTTPException(status_code=503, detail="pymongo not installed")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not docs:
        raise HTTPException(
            status_code=404,
            detail="No analysed submissions found"
        )

    grade_counts: dict[str, int] = {}
    flagged_ai   = 0
    flagged_plag = 0

    for doc in docs:
        analysis = doc.get("analysis", {})
        grade = analysis.get("final", {}).get("grade", "?")
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        if analysis.get("ai_detection", {}).get("is_ai_generated"):
            flagged_ai += 1
        if analysis.get("is_plagiarised"):
            flagged_plag += 1

    return {
        "exam_id":            exam_id,
        "question_id":        question_id,
        "total_submissions":  len(docs),
        "flagged_ai":         flagged_ai,
        "flagged_plagiarism": flagged_plag,
        "grade_distribution": grade_counts,
    }