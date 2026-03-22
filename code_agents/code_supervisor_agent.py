import json
from datetime import datetime


class CodeSupervisorAgent:
    def __init__(self, plagiarism_agent, ai_agent):
        self.plagiarism_agent = plagiarism_agent
        self.ai_agent = ai_agent

    def analyze(self, code: str, language: str):
        plagiarism_result = self.plagiarism_agent.check_plagiarism(code, language)
        ai_result = self.ai_agent.detect(code)

        final_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "language": language,
            "plagiarism": plagiarism_result,
            "ai_detection": ai_result
        }

        self._print_output(final_result)
        self._store_output(final_result)

        return final_result

    def _print_output(self, result):
        print("\n===== CODE ANALYSIS RESULT =====")
        print(json.dumps(result, indent=4))
        print("================================\n")

    def _store_output(self, result):
        with open("code_analytics.json", "a") as f:
            json.dump(result, f)
            f.write("\n")