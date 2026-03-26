import json
from datetime import datetime
from code_agents.code_risk_agent import CodeRiskAgent


class CodeSupervisorAgent:
    def __init__(self, plagiarism_agent, ai_agent):
        self.plagiarism_agent = plagiarism_agent
        self.ai_agent         = ai_agent
        self.code_risk_agent  = CodeRiskAgent()

    def analyze(self, code: str, language: str):
        plagiarism_result = self.plagiarism_agent.check_plagiarism(code, language)
        ai_result         = self.ai_agent.detect(code)

        # ── Update code risk score based on results ───────────────
        if ai_result.get("is_ai_generated", False):
            self.code_risk_agent.update_risk("ai_generated")

        if plagiarism_result.get("is_plagiarized", False):
            self.code_risk_agent.update_risk("plagiarism")

        risk_summary = self.code_risk_agent._summary()

        final_result = {
            "timestamp":    datetime.utcnow().isoformat(),
            "language":     language,
            "plagiarism":   plagiarism_result,
            "ai_detection": ai_result,
            "risk":         risk_summary,
        }

        self._print_output(final_result)
        self._store_output(final_result)

        return final_result

    def update_tab_switch(self):
        """Call this when the candidate switches tabs during the code exam."""
        risk_summary = self.code_risk_agent.update_risk("tab_switched")
        print(f"[CODE SUPERVISOR] Tab switch recorded — {risk_summary}")
        return risk_summary

    def get_risk_summary(self) -> dict:
        """Return the current code risk state."""
        return self.code_risk_agent._summary()

    def _print_output(self, result):
        print(json.dumps(result, indent=4))
        print("================================\n")

    def _store_output(self, result):
        with open("code_analytics.json", "a") as f:
            json.dump(result, f)
            f.write("\n")