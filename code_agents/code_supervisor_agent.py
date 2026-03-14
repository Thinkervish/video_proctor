from code_agents.plagiarism_agent import PlagiarismAgent
from code_agents.ai_code_detector_agent import AICodeDetectorAgent
from code_agents.performance_agent import PerformanceAgent


class CodeSupervisorAgent:
    """
    Orchestrates all 3 code analysis agents and produces
    a unified integrity report with a final score.
    """

    def __init__(self):
        self.plagiarism_agent = PlagiarismAgent()
        self.ai_detector = AICodeDetectorAgent()
        self.performance = PerformanceAgent()

    def analyze(self, code: str, student_id: str = "unknown") -> dict:

        # ── Run agents safely ───────────────────────────────────────────
        try:
            plagiarism = self.plagiarism_agent.analyze(code)
        except Exception:
            plagiarism = {
                "plagiarism_score": 0,
                "flagged": False,
                "matched_student": None
            }

        try:
            ai_result = self.ai_detector.analyze(code)
        except Exception:
            ai_result = {
                "ai_probability": 0,
                "flagged": False,
                "verdict": "unknown",
                "signals": []
            }

        try:
            perf = self.performance.analyze(code)
        except Exception:
            perf = {
                "complexity": 0,
                "complexity_rank": "A",
                "maintainability": 100,
                "mi_rank": "A",
                "quality_score": 100,
                "loc": 0,
                "functions": 0
            }

        # ── Safe value extraction ───────────────────────────────────────
        plag_score = plagiarism.get("plagiarism_score", 0)
        plag_flag = plagiarism.get("flagged", False)

        ai_prob = ai_result.get("ai_probability", 0)
        ai_flag = ai_result.get("flagged", False)
        ai_verdict = ai_result.get("verdict", "unknown")
        ai_signals = ai_result.get("signals", [])

        quality_score = perf.get("quality_score", 0)
        complexity_rank = perf.get("complexity_rank", "A")
        maintainability = perf.get("maintainability", 100)

        # ── Integrity Score (0–100) ─────────────────────────────────────
        plag_penalty = plag_score * 40
        ai_penalty = ai_prob * 35
        quality_bonus = (quality_score / 100) * 25

        integrity = round(
            max(0, 100 - plag_penalty - ai_penalty + quality_bonus - 25), 1
        )

        # ── Flags ───────────────────────────────────────────────────────
        flags = []

        if plag_flag:
            flags.append(f"Plagiarism detected (score: {plag_score:.0%})")

        if ai_flag:
            flags.append(f"AI-generated code suspected ({ai_verdict})")

        if complexity_rank in ["D", "E", "F"]:
            flags.append(f"High complexity (CC rank: {complexity_rank})")

        if maintainability < 40:
            flags.append(f"Poor maintainability (MI: {maintainability})")

        # ── Final report ────────────────────────────────────────────────
        return {
            "student_id": student_id,
            "integrity_score": integrity,
            "flags": flags,
            "clean": len(flags) == 0,

            "plagiarism": {
                "score": plag_score,
                "flagged": plag_flag,
                "matched_student": plagiarism.get("matched_student"),
            },

            "ai_detection": {
                "probability": ai_prob,
                "flagged": ai_flag,
                "verdict": ai_verdict,
                "signals": ai_signals,
            },

            "performance": {
                "complexity": perf.get("complexity"),
                "complexity_rank": complexity_rank,
                "maintainability": maintainability,
                "mi_rank": perf.get("mi_rank"),
                "quality_score": quality_score,
                "loc": perf.get("loc"),
                "functions": perf.get("functions"),
            },
        }

    def store_submission(self, student_id: str, code: str):
        """Store submission in DB for future plagiarism checks."""
        self.plagiarism_agent.store_code(student_id, code)