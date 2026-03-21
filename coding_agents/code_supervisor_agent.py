from __future__ import annotations

import traceback
from typing import Any

from code_agents.ai_code_detector_agent import AICodeDetectorAgent
from code_agents.plagiarism_agent        import PlagiarismAgent
from code_agents.performance_agent       import PerformanceAgent


# ---------------------------------------------------------------------------
# Score aggregation weights
# ---------------------------------------------------------------------------
FINAL_SCORE_WEIGHTS = {
    "quality":          0.40,   # static code quality
    "ai_clean":         0.40,   # inverse of ai_score
    "plagiarism_clean": 0.20,   # inverse of plagiarism score
}


def _final_score(perf: dict, ai: dict, plag_score: float) -> dict[str, Any]:
    """Compute a single composite score from all agent outputs."""
    quality    = perf["quality"]["quality_score"]
    ai_clean   = 1.0 - ai["ai_score"]
    plag_clean = 1.0 - plag_score

    score = (
        FINAL_SCORE_WEIGHTS["quality"]          * quality    +
        FINAL_SCORE_WEIGHTS["ai_clean"]         * ai_clean   +
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
    Orchestrates AI detection, plagiarism checking, and static quality analysis.

    Usage:
        supervisor = CodeSupervisorAgent()
        report = supervisor.analyze(
            submissions=[
                {"candidate_id": "C001", "code": "def solve(n): ..."},
                {"candidate_id": "C002", "code": "def solve(n): ..."},
            ]
        )
    """

    def __init__(
        self,
        ai_threshold:   float = 0.55,
        plag_threshold: float = 0.60,
    ):
        self.ai_detector = AICodeDetectorAgent(threshold=ai_threshold)
        self.plag_agent  = PlagiarismAgent(threshold=plag_threshold)
        self.perf_agent  = PerformanceAgent()

    def analyze(
        self,
        submissions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze a list of submissions.

        Args:
            submissions: list of {"candidate_id": str, "code": str}

        Returns full report with per-candidate results and summary.
        """
        if not submissions:
            return {"error": "No submissions provided", "candidate_reports": []}

        # ---- 1. Plagiarism — batch cross-compare all submissions -----------
        plag_batch = self.plag_agent.analyze_batch(submissions)

        plag_scores: dict[str, float] = {s["candidate_id"]: 0.0 for s in submissions}
        for result in plag_batch["results"]:
            id_a, id_b = result["pair"]
            s = result["plagiarism_score"]
            plag_scores[id_a] = max(plag_scores.get(id_a, 0.0), s)
            plag_scores[id_b] = max(plag_scores.get(id_b, 0.0), s)

        # ---- 2. Per-candidate AI detection + quality ----------------------
        candidate_reports: list[dict[str, Any]] = []

        for sub in submissions:
            cid  = sub["candidate_id"]
            code = sub.get("code", "")

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

            try:
                perf_result = self.perf_agent.analyze(code, candidate_id=cid)
            except Exception:
                perf_result = {
                    "candidate_id": cid,
                    "quality": {"quality_score": 0.0, "issues": [], "has_syntax_error": False},
                    "grade": "F",
                }

            final = _final_score(perf_result, ai_result, plag_scores.get(cid, 0.0))

            report = {
                "candidate_id":     cid,
                "ai_detection":     ai_result,
                "plagiarism_score": round(plag_scores.get(cid, 0.0), 4),
                "is_plagiarised":   plag_scores.get(cid, 0.0) >= self.plag_agent.threshold,
                "quality":          perf_result["quality"],
                "final":            final,
                "risk_flags":       self._build_risk_flags(ai_result, plag_scores.get(cid, 0.0), perf_result),
            }
            candidate_reports.append(report)

        # ---- 3. Summary ---------------------------------------------------
        flagged_ai   = [r for r in candidate_reports if r["ai_detection"]["is_ai_generated"]]
        flagged_plag = [r for r in candidate_reports if r["is_plagiarised"]]

        return {
            "total_submissions":  len(submissions),
            "flagged_ai":         len(flagged_ai),
            "flagged_plagiarism": len(flagged_plag),
            "plagiarism_pairs":   plag_batch["flagged"],
            "candidate_reports":  candidate_reports,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _build_risk_flags(
        ai:   dict[str, Any],
        plag: float,
        perf: dict[str, Any],
    ) -> list[str]:
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