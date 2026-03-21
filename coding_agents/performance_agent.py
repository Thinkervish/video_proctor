import ast
from typing import Any


# ---------------------------------------------------------------------------
# Static code quality checks
# ---------------------------------------------------------------------------

def _static_quality_checks(code: str) -> dict[str, Any]:
    """
    AST-based code quality analysis.
    Returns quality_score (0-1) and list of issues.
    """
    issues: list[str] = []
    deductions = 0.0

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "quality_score": 0.0,
            "issues": [f"Syntax error: {e}"],
            "has_syntax_error": True,
        }

    # 1. Bare except clauses
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append("Bare except clause — catches all exceptions including SystemExit")
            deductions += 0.10

    # 2. Use of eval / exec
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                issues.append(f"Use of {node.func.id}() — security risk")
                deductions += 0.15

    # 3. Global variables
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            issues.append("Use of global statement — avoid mutable global state")
            deductions += 0.05

    # 4. Functions without return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
            if not has_return and node.name != "__init__":
                issues.append(f"Function '{node.name}' has no return statement")
                deductions += 0.05

    # 5. Deeply nested code (> 6 levels)
    def _max_depth(node, depth=0):
        child_depths = [_max_depth(c, depth + 1) for c in ast.iter_child_nodes(node)]
        return max(child_depths, default=depth)

    max_d = _max_depth(tree)
    if max_d > 6:
        issues.append(f"Deep nesting detected (depth {max_d}) — consider refactoring")
        deductions += 0.10

    quality_score = max(0.0, 1.0 - deductions)
    return {
        "quality_score":    round(quality_score, 4),
        "issues":           issues,
        "has_syntax_error": False,
    }


# ---------------------------------------------------------------------------
# MAIN AGENT CLASS
# ---------------------------------------------------------------------------

class PerformanceAgent:
    """
    Evaluates submitted code for static code quality only.
    (Test case running and time complexity analysis removed.)
    """

    def analyze(
        self,
        code: str,
        candidate_id: str = "unknown",
    ) -> dict[str, Any]:
        """
        Returns:
            {
              "candidate_id":  str,
              "quality":       dict,   # static quality checks
              "grade":         str,    # A / B / C / D / F
            }
        """
        quality = _static_quality_checks(code)

        score = quality["quality_score"]

        if score >= 0.90:
            grade = "A"
        elif score >= 0.75:
            grade = "B"
        elif score >= 0.60:
            grade = "C"
        elif score >= 0.45:
            grade = "D"
        else:
            grade = "F"

        return {
            "candidate_id": candidate_id,
            "quality":      quality,
            "grade":        grade,
        }