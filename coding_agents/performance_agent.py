import ast
import subprocess
import sys
import time
import textwrap
import math
import resource
import os
import tempfile
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXECUTION_TIMEOUT   = 5       # seconds per test case
MAX_OUTPUT_CHARS    = 2000    # truncate large stdout
MEMORY_LIMIT_MB     = 128     # soft memory limit for sandbox


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

def _run_code_in_sandbox(
    code: str,
    stdin_input: str = "",
    timeout: float = EXECUTION_TIMEOUT,
) -> dict[str, Any]:
    """
    Execute arbitrary Python code in a subprocess sandbox.
    Returns stdout, stderr, return_code, elapsed_time.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return {
            "stdout":       proc.stdout[:MAX_OUTPUT_CHARS],
            "stderr":       proc.stderr[:MAX_OUTPUT_CHARS],
            "return_code":  proc.returncode,
            "elapsed_ms":   round(elapsed * 1000, 2),
            "timed_out":    False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout":       "",
            "stderr":       f"Execution timed out after {timeout}s",
            "return_code":  -1,
            "elapsed_ms":   timeout * 1000,
            "timed_out":    True,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Test case runner
# ---------------------------------------------------------------------------

def _run_test_cases(
    code: str,
    test_cases: list[dict],
) -> dict[str, Any]:
    """
    Run a list of test cases against submitted code.

    Each test case:
        {
          "input":          str,   # fed to stdin
          "expected_output": str,  # compared with stdout (stripped)
          "label":          str,   # optional human-readable label
        }

    Returns summary + per-case results.
    """
    results: list[dict] = []
    passed = 0

    for i, tc in enumerate(test_cases):
        label    = tc.get("label", f"Test {i + 1}")
        expected = tc.get("expected_output", "").strip()
        run      = _run_code_in_sandbox(code, stdin_input=tc.get("input", ""))
        actual   = run["stdout"].strip()

        ok = (not run["timed_out"]) and (run["return_code"] == 0) and (actual == expected)
        if ok:
            passed += 1

        results.append({
            "label":      label,
            "passed":     ok,
            "expected":   expected,
            "actual":     actual,
            "stderr":     run["stderr"],
            "elapsed_ms": run["elapsed_ms"],
            "timed_out":  run["timed_out"],
        })

    total = len(test_cases)
    return {
        "passed":        passed,
        "failed":        total - passed,
        "total":         total,
        "score":         round(passed / total, 4) if total else 0.0,
        "case_results":  results,
    }


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

def _estimate_time_complexity(code: str, entry_function: str = "") -> dict[str, Any]:
    """
    Empirically estimate time complexity by running the code with
    increasing input sizes and fitting the growth curve.

    Requires the submitted code to expose a callable function named
    `entry_function`.  If not provided, wraps code in a generic runner.

    Returns:
        {
          "complexity_class":  str,   # O(1) / O(n) / O(n log n) / O(n²) / O(2ⁿ) / unknown
          "timings_ms":        list,
          "input_sizes":       list,
        }
    """
    sizes = [10, 50, 100, 200, 500]
    timings: list[float] = []

    if not entry_function:
        return {
            "complexity_class": "unknown",
            "timings_ms":       [],
            "input_sizes":      sizes,
        }

    for n in sizes:
        harness = textwrap.dedent(f"""
import time
{code}

_input = list(range({n}))
_start = time.perf_counter()
try:
    {entry_function}(_input)
except Exception:
    pass
print(time.perf_counter() - _start)
""")
        run = _run_code_in_sandbox(harness, timeout=3.0)
        try:
            t = float(run["stdout"].strip())
            timings.append(t * 1000)
        except (ValueError, IndexError):
            timings.append(None)

    # Filter valid timings
    valid = [(s, t) for s, t in zip(sizes, timings) if t is not None and t > 0]
    if len(valid) < 3:
        return {
            "complexity_class": "unknown",
            "timings_ms":       timings,
            "input_sizes":      sizes,
        }

    # Fit ratios: compare T(2n)/T(n) growth
    ratios: list[float] = []
    for i in range(1, len(valid)):
        n1, t1 = valid[i - 1]
        n2, t2 = valid[i]
        if t1 > 0:
            ratios.append(t2 / t1)

    avg_ratio = sum(ratios) / len(ratios)
    n_ratio   = valid[-1][0] / valid[0][0]  # how much n grew total

    # Classify based on average timing ratio per doubling
    if avg_ratio < 1.2:
        complexity = "O(1)"
    elif avg_ratio < 2.5:
        complexity = "O(n)"
    elif avg_ratio < 4.0:
        complexity = "O(n log n)"
    elif avg_ratio < 6.0:
        complexity = "O(n²)"
    else:
        complexity = "O(2ⁿ) or worse"

    return {
        "complexity_class": complexity,
        "timings_ms":       timings,
        "input_sizes":      sizes,
    }


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

    # 3. Global variables (outside module level)
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            issues.append("Use of global statement — avoid mutable global state")
            deductions += 0.05

    # 4. Functions without return (non-trivial functions)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
            if not has_return and node.name != "__init__":
                issues.append(f"Function '{node.name}' has no return statement")
                deductions += 0.05

    # 5. Deeply nested code (> 4 levels)
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
    Evaluates submitted code for correctness, complexity, and quality.

    Usage:
        agent = PerformanceAgent()
        result = agent.analyze(
            code=student_code,
            candidate_id="C001",
            test_cases=[
                {"input": "5\n", "expected_output": "120", "label": "factorial(5)"},
                {"input": "0\n", "expected_output": "1",   "label": "factorial(0)"},
            ],
            entry_function="solve",   # optional, for complexity timing
        )
    """

    def __init__(self, timeout: float = EXECUTION_TIMEOUT):
        self.timeout = timeout

    # ------------------------------------------------------------------
    def analyze(
        self,
        code: str,
        candidate_id: str = "unknown",
        test_cases: list[dict] | None = None,
        entry_function: str = "",
    ) -> dict[str, Any]:
        """
        Full performance analysis.

        Returns:
            {
              "candidate_id":       str,
              "test_results":       dict,   # correctness
              "complexity":         dict,   # time complexity estimate
              "quality":            dict,   # static quality checks
              "overall_score":      float,  # 0-1 composite
              "grade":              str,    # A / B / C / D / F
            }
        """
        test_cases = test_cases or []

        # Run correctness tests
        test_results = _run_test_cases(code, test_cases) if test_cases else {
            "passed": 0, "failed": 0, "total": 0, "score": 0.0, "case_results": []
        }

        # Estimate complexity (if entry function provided)
        complexity = _estimate_time_complexity(code, entry_function)

        # Static quality checks
        quality = _static_quality_checks(code)

        # Composite score
        correctness_weight = 0.60
        quality_weight     = 0.40

        correctness_score = test_results["score"] if test_cases else 0.5
        overall = (
            correctness_weight * correctness_score +
            quality_weight     * quality["quality_score"]
        )
        overall = round(overall, 4)

        if overall >= 0.90:
            grade = "A"
        elif overall >= 0.75:
            grade = "B"
        elif overall >= 0.60:
            grade = "C"
        elif overall >= 0.45:
            grade = "D"
        else:
            grade = "F"

        return {
            "candidate_id":  candidate_id,
            "test_results":  test_results,
            "complexity":    complexity,
            "quality":       quality,
            "overall_score": overall,
            "grade":         grade,
        }
