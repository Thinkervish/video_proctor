import ast
import math
import re
import tokenize
import io
import keyword
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Feature weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "ast_complexity_variance":   0.25,
    "identifier_naming_style":   0.20,
    "comment_density":           0.15,
    "token_entropy_burstiness":  0.15,
    "error_handling_uniformity": 0.10,
    "unused_constructs":         0.08,
    "io_frequency":              0.04,
    "import_usage_ratio":        0.03,
}

# Score >= this threshold  →  flagged as AI-generated
AI_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse(code: str) -> ast.AST | None:
    """Return AST or None if code does not parse."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _get_lines(code: str) -> list[str]:
    return code.splitlines()


# ---------------------------------------------------------------------------
# TIER 1 FEATURES
# ---------------------------------------------------------------------------

def _feature_ast_complexity_variance(tree: ast.AST) -> float:
    """
    AI code has low, uniform AST nesting depth.
    Human exam code has high variance in depth across functions.
    Returns a score 0-1 where HIGH = AI-like (low variance).
    """
    depths: list[int] = []

    def _walk(node, depth):
        depths.append(depth)
        for child in ast.iter_child_nodes(node):
            _walk(child, depth + 1)

    _walk(tree, 0)
    if len(depths) < 5:
        return 0.5  # not enough data

    mean = sum(depths) / len(depths)
    variance = sum((d - mean) ** 2 for d in depths) / len(depths)
    std_dev = math.sqrt(variance)

    # Low std_dev → uniform → AI-like → score close to 1
    # Normalise: std_dev typically 2-8 for human code
    score = max(0.0, 1.0 - (std_dev / 8.0))
    return min(score, 1.0)


def _feature_cyclomatic_complexity(tree: ast.AST) -> float:
    """
    Measure average cyclomatic complexity per function.
    AI code: low, flat (score → 1).  Human code: higher, irregular.
    """
    branch_nodes = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )
    complexities: list[int] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            count = 1  # base complexity
            for child in ast.walk(node):
                if isinstance(child, branch_nodes):
                    count += 1
                elif isinstance(child, ast.BoolOp):
                    count += len(child.values) - 1
            complexities.append(count)

    if not complexities:
        return 0.5

    avg = sum(complexities) / len(complexities)
    # AI code tends to have avg complexity 1-3; human exam code 3-8+
    score = max(0.0, 1.0 - ((avg - 1) / 7.0))
    return min(score, 1.0)


def _feature_identifier_naming_style(tree: ast.AST) -> float:
    """
    AI uses verbose, textbook names (result_list, temp_value).
    Humans under exam pressure use short names (res, tmp, x, ans).
    Returns score 0-1 where HIGH = AI-like (verbose names).
    """
    names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not keyword.iskeyword(node.id):
            names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)

    if not names:
        return 0.5

    # Heuristics for AI-like naming
    verbose_count = 0
    for name in names:
        # snake_case with >= 2 words  →  AI-like
        parts = name.split("_")
        real_parts = [p for p in parts if p]
        if len(real_parts) >= 2 and all(p.isalpha() for p in real_parts):
            verbose_count += 1
        # long single-word names (>= 8 chars without underscore)  →  AI-like
        elif len(name) >= 8 and "_" not in name and name.isalpha():
            verbose_count += 1

    ratio = verbose_count / len(names)
    # ratio > 0.5 → mostly verbose → AI-like
    score = min(ratio * 2.0, 1.0)
    return score


def _feature_comment_density(code: str, tree: ast.AST) -> float:
    """
    AI over-comments trivial lines and adds docstrings everywhere.
    Returns score 0-1 where HIGH = AI-like (high comment density).
    """
    lines = _get_lines(code)
    total_lines = max(len(lines), 1)

    # Count inline comments and blank-line-normalised comment lines
    comment_lines = sum(
        1 for ln in lines
        if ln.strip().startswith("#")
    )

    # Count docstrings
    docstring_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                docstring_count += 1

    code_lines = total_lines - comment_lines
    comment_ratio = (comment_lines + docstring_count * 3) / max(code_lines, 1)

    # ratio > 0.3 (1 comment per 3 code lines) → AI-like
    score = min(comment_ratio / 0.3, 1.0)
    return score


# ---------------------------------------------------------------------------
# TIER 2 FEATURES
# ---------------------------------------------------------------------------

def _feature_error_handling_uniformity(tree: ast.AST) -> float:
    """
    AI wraps everything in try/except unnecessarily.
    Detect: number of try blocks vs total functions.
    HIGH score = AI-like.
    """
    try_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))
    func_count = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )

    if func_count == 0:
        return 0.0

    ratio = try_count / func_count
    # ratio > 0.6 → almost every function has try/except → AI-like
    score = min(ratio / 0.6, 1.0)
    return score


def _feature_unused_constructs(code: str, tree: ast.AST) -> float:
    """
    AI code often contains unused imports and variables.
    Returns score 0-1 where HIGH = AI-like.
    """
    # Collect all imported names
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                imported.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    name = alias.asname if alias.asname else alias.name
                    imported.add(name)

    if not imported:
        return 0.0

    # Check usage of each imported name in the rest of the code
    used = set()
    for name in imported:
        # Simple text search (avoids false negatives from AST scoping)
        pattern = r'\b' + re.escape(name) + r'\b'
        # Count occurrences; import line itself counts as 1
        matches = len(re.findall(pattern, code))
        if matches > 1:
            used.add(name)

    unused_ratio = (len(imported) - len(used)) / len(imported)
    return min(unused_ratio * 2.0, 1.0)


def _feature_token_entropy_burstiness(code: str) -> float:
    """
    AI code has uniform token distribution (low burstiness).
    Human code has high variance — messy sections + clean sections.
    Returns score 0-1 where HIGH = AI-like (low burstiness).
    """
    lines = _get_lines(code)
    if len(lines) < 10:
        return 0.5

    # Split into windows of 10 lines
    window_size = 10
    window_entropies: list[float] = []

    for i in range(0, len(lines), window_size):
        chunk = "\n".join(lines[i: i + window_size])
        tokens = re.findall(r'[A-Za-z_]\w*|[+\-*/=<>!&|]+|[(){}\[\],.:;]|\d+', chunk)
        if not tokens:
            continue
        freq = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        window_entropies.append(entropy)

    if len(window_entropies) < 2:
        return 0.5

    mean_e = sum(window_entropies) / len(window_entropies)
    variance_e = sum((e - mean_e) ** 2 for e in window_entropies) / len(window_entropies)
    std_e = math.sqrt(variance_e)

    # Low std of entropy across windows → uniform → AI-like
    # std typically 0.3-1.5 for human code
    score = max(0.0, 1.0 - (std_e / 1.5))
    return min(score, 1.0)


def _feature_io_frequency(tree: ast.AST, total_lines: int) -> float:
    """
    AI code uses excessive print / input / file read calls.
    Returns score 0-1 where HIGH = AI-like.
    """
    io_calls = 0
    io_names = {"print", "input", "open", "read", "readline", "readlines", "write"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in io_names:
                io_calls += 1
            elif isinstance(node.func, ast.Attribute) and node.func.attr in io_names:
                io_calls += 1

    ratio = io_calls / max(total_lines, 1)
    # ratio > 0.15 (1 I/O call per ~7 lines) → AI-like
    score = min(ratio / 0.15, 1.0)
    return score


# ---------------------------------------------------------------------------
# TIER 3 FEATURES
# ---------------------------------------------------------------------------

def _feature_import_usage_ratio(code: str, tree: ast.AST) -> float:
    """
    AI imports libraries 'just in case'.
    Ratio of unused imports to total imports → HIGH = AI-like.
    (Reuses unused_constructs logic but isolated for weighting.)
    """
    return _feature_unused_constructs(code, tree)


# ---------------------------------------------------------------------------
# MAIN AGENT CLASS
# ---------------------------------------------------------------------------

class AICodeDetectorAgent:
    """
    Analyses a single code submission and returns an AI-detection result.

    Usage:
        agent = AICodeDetectorAgent()
        result = agent.analyze(code_string, candidate_id="C001")
    """

    def __init__(self, threshold: float = AI_THRESHOLD):
        self.threshold = threshold

    # ------------------------------------------------------------------
    def analyze(self, code: str, candidate_id: str = "unknown") -> dict[str, Any]:
        """
        Run all feature extractors and return a structured result.

        Returns:
            {
              "candidate_id": str,
              "ai_score": float,          # 0.0 – 1.0
              "is_ai_generated": bool,
              "confidence": str,          # LOW / MEDIUM / HIGH
              "feature_scores": dict,     # per-feature breakdown
              "flags": list[str],         # human-readable reasons
              "lines_analyzed": int,
            }
        """
        result: dict[str, Any] = {
            "candidate_id": candidate_id,
            "ai_score": 0.0,
            "is_ai_generated": False,
            "confidence": "LOW",
            "feature_scores": {},
            "flags": [],
            "lines_analyzed": len(_get_lines(code)),
        }

        if not code or not code.strip():
            result["flags"].append("empty_submission")
            return result

        tree = _safe_parse(code)
        if tree is None:
            result["flags"].append("parse_error_syntax_invalid")
            result["ai_score"] = 0.0
            return result

        total_lines = max(len(_get_lines(code)), 1)

        # ---- Extract all feature scores --------------------------------
        scores: dict[str, float] = {
            "ast_complexity_variance":   _feature_ast_complexity_variance(tree),
            "cyclomatic_complexity":     _feature_cyclomatic_complexity(tree),
            "identifier_naming_style":   _feature_identifier_naming_style(tree),
            "comment_density":           _feature_comment_density(code, tree),
            "token_entropy_burstiness":  _feature_token_entropy_burstiness(code),
            "error_handling_uniformity": _feature_error_handling_uniformity(tree),
            "unused_constructs":         _feature_unused_constructs(code, tree),
            "io_frequency":              _feature_io_frequency(tree, total_lines),
            "import_usage_ratio":        _feature_import_usage_ratio(code, tree),
        }

        # ---- Weighted final score --------------------------------------
        # cyclomatic_complexity shares the ast_complexity_variance weight
        raw_score = (
            WEIGHTS["ast_complexity_variance"]   * (scores["ast_complexity_variance"] * 0.6 +
                                                     scores["cyclomatic_complexity"] * 0.4) +
            WEIGHTS["identifier_naming_style"]   * scores["identifier_naming_style"] +
            WEIGHTS["comment_density"]           * scores["comment_density"] +
            WEIGHTS["token_entropy_burstiness"]  * scores["token_entropy_burstiness"] +
            WEIGHTS["error_handling_uniformity"] * scores["error_handling_uniformity"] +
            WEIGHTS["unused_constructs"]         * scores["unused_constructs"] +
            WEIGHTS["io_frequency"]              * scores["io_frequency"] +
            WEIGHTS["import_usage_ratio"]        * scores["import_usage_ratio"]
        )

        ai_score = round(min(max(raw_score, 0.0), 1.0), 4)

        # ---- Build flags (human-readable explanation) ------------------
        flags: list[str] = []
        if scores["ast_complexity_variance"] > 0.65:
            flags.append("Unusually uniform AST structure")
        if scores["identifier_naming_style"] > 0.60:
            flags.append("Verbose, textbook-style variable names")
        if scores["comment_density"] > 0.60:
            flags.append("Over-commented code — AI pattern")
        if scores["token_entropy_burstiness"] > 0.65:
            flags.append("Uniform token distribution across sections")
        if scores["error_handling_uniformity"] > 0.60:
            flags.append("Excessive / boilerplate try-except blocks")
        if scores["unused_constructs"] > 0.50:
            flags.append("Unused imports or variables detected")
        if scores["io_frequency"] > 0.60:
            flags.append("High I/O call frequency")

        # ---- Confidence level ------------------------------------------
        if ai_score >= 0.75:
            confidence = "HIGH"
        elif ai_score >= 0.55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        result.update({
            "ai_score":        ai_score,
            "is_ai_generated": ai_score >= self.threshold,
            "confidence":      confidence,
            "feature_scores":  scores,
            "flags":           flags,
        })

        return result
