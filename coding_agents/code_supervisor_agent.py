from __future__ import annotations

import datetime
import traceback
from typing import Any

# Sub-agents
from code_agents.ai_code_detector_agent import AICodeDetectorAgent
from code_agents.plagiarism_agent        import PlagiarismAgent
from code_agents.performance_agent       import PerformanceAgent


# ---------------------------------------------------------------------------
# MongoDB connector (lazy import so the app works without pymongo installed
# during local testing — swap in a mock if needed)
# ---------------------------------------------------------------------------

def _get_mongo_collection(uri: str, db_name: str, collection: str):
    """Return a pymongo Collection.  Raises ImportError if pymongo absent."""
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        return client[db_name][collection]
    except ImportError as exc:
        raise ImportError(
            "pymongo is not installed. Run: pip install pymongo"
        ) from exc


# ---------------------------------------------------------------------------
# Score aggregation weights
# ---------------------------------------------------------------------------
FINAL_SCORE_WEIGHTS = {
    "correctness":   0.50,   # performance_agent test pass rate
    "quality":       0.20,   # performance_agent code quality
    "ai_clean":      0.20,   # inverse of ai_score (lower AI score = better)
    "plagiarism_clean": 0.10,  # inverse of plagiarism score
}


def _final_score(perf: dict, ai: dict, plag_score: float) -> dict[str, Any]:
    """Compute a single composite score from all agent outputs."""
    correctness = perf["test_results"]["score"]
    quality     = perf["quality"]["quality_score"]
    ai_clean    = 1.0 - ai["ai_score"]
    plag_clean  = 1.0 - plag_score

    score = (
        FINAL_SCORE_WEIGHTS["correctness"]      * correctness +
        FINAL_SCORE_WEIGHTS["quality"]          * quality     +
        FINAL_SCORE_WEIGHTS["ai_clean"]         * ai_clean    +
        FINAL_SCORE_WEIGHTS["plagiarism_clean"] * plag_clean
    )
    score = round(score, 4)

    if score >= 0.85:
        grade = "A"
    elif score >= 0.70:
        grade = "B"
    elif score >= 0.55:
        grade = "C"
    elif score >= 0.40:
        grade = "D"
    else:
        grade = "F"

    return {"composite_score": score, "grade": grade}


# ---------------------------------------------------------------------------
# MAIN SUPERVISOR CLASS
# ---------------------------------------------------------------------------

class CodeSupervisorAgent:
    """
    Orchestrates the full code analysis pipeline.

    Usage (with real MongoDB):
        supervisor = CodeSupervisorAgent(
            mongo_uri="mongodb://localhost:27017",
            db_name="video_proctor",
            collection_name="submissions",
        )
        # Analyze all submissions for a specific exam question
        report = supervisor.analyze_question(
            exam_id="EXAM_2025_001",
            question_id="Q3",
            test_cases=[
                {"input": "5\n", "expected_output": "120", "label": "n=5"},
            ],
            entry_function="solve",
        )

    Usage (without MongoDB — pass submissions directly):
        report = supervisor.analyze_direct(
            submissions=[
                {"candidate_id": "C001", "code": "def solve(n): ..."},
            ],
            test_cases=[...],
        )
    """

    def __init__(
        self,
        mongo_uri:        str  = "mongodb://localhost:27017",
        db_name:          str  = "video_proctor",
        collection_name:  str  = "submissions",
        ai_threshold:     float = 0.55,
        plag_threshold:   float = 0.60,
        exec_timeout:     float = 5.0,
    ):
        self.mongo_uri       = mongo_uri
        self.db_name         = db_name
        self.collection_name = collection_name

        # Initialise sub-agents
        self.ai_detector   = AICodeDetectorAgent(threshold=ai_threshold)
        self.plag_agent    = PlagiarismAgent(threshold=plag_threshold)
        self.perf_agent    = PerformanceAgent(timeout=exec_timeout)

        # Lazy MongoDB collection (connected on first use)
        self._collection   = None

    # ------------------------------------------------------------------
    # MongoDB helpers
    # ------------------------------------------------------------------

    def _get_collection(self):
        if self._collection is None:
            self._collection = _get_mongo_collection(
                self.mongo_uri, self.db_name, self.collection_name
            )
        return self._collection

    def _fetch_submissions(
        self,
        exam_id: str,
        question_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch all submissions for a given exam + question from MongoDB."""
        col = self._get_collection()
        cursor = col.find(
            {"exam_id": exam_id, "question_id": question_id},
            {"_id": 0, "candidate_id": 1, "code": 1},
        )
        return list(cursor)

    def _write_analysis(
        self,
        exam_id:      str,
        question_id:  str,
        candidate_id: str,
        analysis:     dict[str, Any],
    ) -> None:
        """Write analysis results back to MongoDB."""
        try:
            col = self._get_collection()
            col.update_one(
                {
                    "exam_id":      exam_id,
                    "question_id":  question_id,
                    "candidate_id": candidate_id,
                },
                {"$set": {"analysis": analysis, "analysed_at": datetime.datetime.utcnow()}},
            )
        except Exception as e:
            print(f"[Supervisor] Warning: could not write analysis to MongoDB: {e}")

    # ------------------------------------------------------------------
    # Core analysis pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        submissions:    list[dict[str, Any]],
        test_cases:     list[dict]  = None,
        entry_function: str         = "",
        exam_id:        str         = "",
        question_id:    str         = "",
        write_back:     bool        = False,
    ) -> dict[str, Any]:
        """
        Internal: run all agents on a list of submissions, return full report.
        """
        test_cases = test_cases or []

        # ---- 1. Plagiarism — batch (cross-compare all submissions) --------
        plag_batch = self.plag_agent.analyze_batch(submissions)

        # Build a lookup: candidate_id → highest plagiarism score against any peer
        plag_scores: dict[str, float] = {s["candidate_id"]: 0.0 for s in submissions}
        for result in plag_batch["results"]:
            id_a, id_b = result["pair"]
            s = result["plagiarism_score"]
            plag_scores[id_a] = max(plag_scores.get(id_a, 0.0), s)
            plag_scores[id_b] = max(plag_scores.get(id_b, 0.0), s)

        # ---- 2. Per-candidate AI detection + performance ------------------
        candidate_reports: list[dict[str, Any]] = []

        for sub in submissions:
            cid  = sub["candidate_id"]
            code = sub.get("code", "")

            # AI detection
            try:
                ai_result = self.ai_detector.analyze(code, candidate_id=cid)
            except Exception:
                ai_result = {
                    "candidate_id": cid,
                    "ai_score": 0.0,
                    "is_ai_generated": False,
                    "confidence": "LOW",
                    "feature_scores": {},
                    "flags": [f"AI detection error: {traceback.format_exc(limit=2)}"],
                    "lines_analyzed": 0,
                }

            # Performance + correctness
            try:
                perf_result = self.perf_agent.analyze(
                    code,
                    candidate_id=cid,
                    test_cases=test_cases,
                    entry_function=entry_function,
                )
            except Exception:
                perf_result = {
                    "candidate_id": cid,
                    "test_results": {"passed": 0, "failed": 0, "total": 0, "score": 0.0, "case_results": []},
                    "complexity":   {"complexity_class": "unknown", "timings_ms": [], "input_sizes": []},
                    "quality":      {"quality_score": 0.0, "issues": [], "has_syntax_error": False},
                    "overall_score": 0.0,
                    "grade": "F",
                }

            # Final composite
            final = _final_score(perf_result, ai_result, plag_scores.get(cid, 0.0))

            report = {
                "candidate_id":        cid,
                "ai_detection":        ai_result,
                "plagiarism_score":    round(plag_scores.get(cid, 0.0), 4),
                "is_plagiarised":      plag_scores.get(cid, 0.0) >= self.plag_agent.threshold,
                "performance":         perf_result,
                "final":               final,
                "risk_flags":          self._build_risk_flags(ai_result, plag_scores.get(cid, 0.0), perf_result),
            }
            candidate_reports.append(report)

            # Write results back to MongoDB if requested
            if write_back and exam_id and question_id:
                self._write_analysis(exam_id, question_id, cid, report)

        # ---- 3. Summary ---------------------------------------------------
        flagged_ai   = [r for r in candidate_reports if r["ai_detection"]["is_ai_generated"]]
        flagged_plag = [r for r in candidate_reports if r["is_plagiarised"]]

        return {
            "exam_id":             exam_id,
            "question_id":         question_id,
            "total_submissions":   len(submissions),
            "flagged_ai":          len(flagged_ai),
            "flagged_plagiarism":  len(flagged_plag),
            "plagiarism_pairs":    plag_batch["flagged"],
            "candidate_reports":   candidate_reports,
            "analysed_at":         datetime.datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_question(
        self,
        exam_id:        str,
        question_id:    str,
        test_cases:     list[dict]  = None,
        entry_function: str         = "",
        write_back:     bool        = True,
    ) -> dict[str, Any]:
        """
        Fetch submissions from MongoDB for one exam question and analyse them.
        """
        submissions = self._fetch_submissions(exam_id, question_id)
        if not submissions:
            return {
                "exam_id":    exam_id,
                "question_id": question_id,
                "error":      "No submissions found for this exam/question",
            }
        return self._run_pipeline(
            submissions,
            test_cases=test_cases,
            entry_function=entry_function,
            exam_id=exam_id,
            question_id=question_id,
            write_back=write_back,
        )

    def analyze_direct(
        self,
        submissions:    list[dict[str, Any]],
        test_cases:     list[dict]  = None,
        entry_function: str         = "",
    ) -> dict[str, Any]:
        """
        Analyse a list of submissions passed directly (no MongoDB needed).
        Useful for testing and API endpoints.
        """
        return self._run_pipeline(
            submissions,
            test_cases=test_cases,
            entry_function=entry_function,
        )

    # ------------------------------------------------------------------
    # Risk flag builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_risk_flags(
        ai:   dict[str, Any],
        plag: float,
        perf: dict[str, Any],
    ) -> list[str]:
        """Build a list of human-readable risk flags for the dashboard."""
        flags: list[str] = []

        if ai["is_ai_generated"]:
            flags.append(
                f"AI-generated code detected (score={ai['ai_score']:.2f}, "
                f"confidence={ai['confidence']})"
            )
            flags.extend(ai.get("flags", []))

        if plag >= 0.85:
            flags.append(f"Confirmed copy detected (plagiarism score={plag:.2f})")
        elif plag >= 0.60:
            flags.append(f"Likely plagiarism detected (score={plag:.2f})")

        if perf["quality"].get("has_syntax_error"):
            flags.append("Submission contains syntax errors")

        for issue in perf["quality"].get("issues", []):
            flags.append(f"Code quality: {issue}")

        if not flags:
            flags.append("No issues detected")

        return flags
