from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit
from radon.raw import analyze
import ast

class PerformanceAgent:
    """
    Evaluates code quality using radon metrics:
    - Cyclomatic Complexity (CC)  — how complex is the logic?
    - Maintainability Index (MI)  — how maintainable is the code?
    - Raw metrics                 — LOC, LLOC, comments, blank lines
    """

    def analyze(self, code: str) -> dict:
        result = {
            "complexity":        0,
            "complexity_rank":   "A",
            "maintainability":   0,
            "mi_rank":           "A",
            "loc":               0,
            "lloc":              0,
            "comments":          0,
            "blank_lines":       0,
            "functions":         [],
            "quality_score":     0,
            "note":              "",
        }

        # ── Syntax check ──────────────────────────────────────────────────
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["note"] = f"Syntax error: {e}"
            return result

        # ── Cyclomatic Complexity ─────────────────────────────────────────
        try:
            blocks = cc_visit(code)
            if blocks:
                avg_cc = sum(b.complexity for b in blocks) / len(blocks)
                result["complexity"] = round(avg_cc, 2)
                result["complexity_rank"] = cc_rank(avg_cc)
                result["functions"] = [
                    {
                        "name":       b.name,
                        "complexity": b.complexity,
                        "rank":       cc_rank(b.complexity),
                    }
                    for b in sorted(blocks, key=lambda x: -x.complexity)[:5]
                ]
        except Exception as e:
            result["note"] += f"CC error: {e} "

        # ── Maintainability Index ─────────────────────────────────────────
        try:
            mi = mi_visit(code, multi=True)
            result["maintainability"] = round(mi, 2)
            if   mi >= 85: result["mi_rank"] = "A"
            elif mi >= 65: result["mi_rank"] = "B"
            elif mi >= 40: result["mi_rank"] = "C"
            else:          result["mi_rank"] = "F"
        except Exception as e:
            result["note"] += f"MI error: {e} "

        # ── Raw metrics ───────────────────────────────────────────────────
        try:
            raw = analyze(code)
            result["loc"]        = raw.loc
            result["lloc"]       = raw.lloc
            result["comments"]   = raw.comments
            result["blank_lines"]= raw.blank
        except Exception as e:
            result["note"] += f"Raw error: {e} "

        # ── Overall quality score (0-100) ─────────────────────────────────
        # Combine CC (lower=better) and MI (higher=better)
        cc_score = max(0, 100 - (result["complexity"] * 10))
        mi_score = min(result["maintainability"], 100)
        result["quality_score"] = round((cc_score * 0.4 + mi_score * 0.6), 1)

        return result