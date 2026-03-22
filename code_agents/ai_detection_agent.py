import re

class AIDetectionAgent:
    def detect(self, code: str):
        lines = code.split("\n")
        tokens = re.findall(r"\w+", code)

        # Heuristics
        avg_line_length = sum(len(line) for line in lines) / (len(lines) or 1)
        unique_token_ratio = len(set(tokens)) / (len(tokens) or 1)

        indentation_consistency = self._check_indentation(lines)
        repetitive_patterns = self._check_repetition(tokens)

        score = 0

        # Rule-based scoring
        if avg_line_length > 60:
            score += 0.2

        if unique_token_ratio < 0.4:
            score += 0.2

        if indentation_consistency:
            score += 0.2

        if repetitive_patterns:
            score += 0.2

        if "def " in code and "return" in code:
            score += 0.2

        score = min(score, 1.0)

        return {
            "ai_score": round(score, 2),
            "is_ai_generated": score >= 0.6,
            "features": {
                "avg_line_length": round(avg_line_length, 2),
                "unique_token_ratio": round(unique_token_ratio, 2),
                "indentation_consistent": indentation_consistency,
                "repetitive_patterns": repetitive_patterns
            }
        }

    def _check_indentation(self, lines):
        indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        return len(set(indent_levels)) <= 3  # very uniform → AI-like

    def _check_repetition(self, tokens):
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        repeated = sum(1 for v in freq.values() if v > 5)
        return repeated > 3