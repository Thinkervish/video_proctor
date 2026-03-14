from fastapi import APIRouter
from pydantic import BaseModel
from code_agents.code_supervisor_agent import CodeSupervisorAgent
import time

router     = APIRouter()
_supervisor = CodeSupervisorAgent()

# In-memory store — all submissions for admin to view
_submissions = []

class CodeSubmission(BaseModel):
    code:       str
    student_id: str = "anonymous"
    store:      bool = False

class StoreSubmission(BaseModel):
    student_id: str
    code:       str

@router.post("/analyze_code")
def analyze_code(submission: CodeSubmission):
    result = _supervisor.analyze(submission.code, submission.student_id)

    # Always store in plagiarism DB + submissions list
    _supervisor.store_submission(submission.student_id, submission.code)

    # Save full result + code + timestamp for admin view
    record = dict(result)
    record["code"]         = submission.code
    record["submitted_at"] = time.strftime("%H:%M:%S")
    _submissions.append(record)

    return result

@router.get("/submissions")
def get_submissions():
    """Admin endpoint — returns all submissions with analysis."""
    return {"submissions": _submissions}

@router.post("/store_code")
def store_code(submission: StoreSubmission):
    _supervisor.store_submission(submission.student_id, submission.code)
    return {"status": "stored", "student_id": submission.student_id}

@router.get("/code_db_size")
def db_size():
    return {"count": len(_supervisor.plagiarism_agent.db)}