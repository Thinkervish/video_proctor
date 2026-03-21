import ast
import hashlib
import re
from collections import defaultdict
from itertools import combinations
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KGRAM_SIZE        = 5      # tokens per k-gram (winnowing)
WINDOW_SIZE       = 4      # window for winnowing (pick min hash per window)
PLAGIARISM_THRESHOLD = 0.60   # score >= this → flagged


# ---------------------------------------------------------------------------
# AST Structural Fingerprinting
# ---------------------------------------------------------------------------

class _ASTNormalizer(ast.NodeTransformer):
    """
    Strips all identifier names → replaces with generic tokens.
    After normalisation two structurally identical programs are identical
    even if they use completely different variable names.
    """

    def visit_Name(self, node):
        node.id = "VAR"
        return node

    def visit_arg(self, node):
        node.arg = "ARG"
        return node

    def visit_FunctionDef(self, node):
        node.name = "FUNC"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        node.name = "FUNC"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.name = "CLASS"
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        # Normalise all literals to their type token
        if isinstance(node.value, str):
            node.value = "STR"
        elif isinstance(node.value, (int, float)):
            node.value = 0
        elif isinstance(node.value, bool):
            node.value = False
        return node

    def visit_Import(self, node):
        # Strip import names too
        for alias in node.names:
            alias.name = "MODULE"
            alias.asname = None
        return node

    def visit_ImportFrom(self, node):
        node.module = "MODULE"
        for alias in node.names:
            alias.name = "NAME"
            alias.asname = None
        return node


def _ast_fingerprint(code: str) -> str | None:
    """
    Parse → normalise → dump to string → hash.
    Returns None if code has syntax errors.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    normalizer = _ASTNormalizer()
    normalised = normalizer.visit(tree)
    dumped = ast.dump(normalised, indent=None)
    return hashlib.sha256(dumped.encode()).hexdigest()


def _ast_subtree_similarity(code_a: str, code_b: str) -> float:
    """
    Compare normalised AST dumps at function granularity.
    Returns 0-1 where 1 = identical structure.
    """
    def _get_function_trees(code: str) -> list[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        normalizer = _ASTNormalizer()
        results = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                normalizer.visit(node)
                results.append(ast.dump(node))
        return results

    funcs_a = _get_function_trees(code_a)
    funcs_b = _get_function_trees(code_b)

    if not funcs_a or not funcs_b:
        # Fall back to whole-file AST hash
        ha = _ast_fingerprint(code_a)
        hb = _ast_fingerprint(code_b)
        if ha and hb:
            return 1.0 if ha == hb else 0.0
        return 0.0

    # Jaccard over normalised function dumps
    set_a = set(funcs_a)
    set_b = set(funcs_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Winnowing Token-Fingerprint (MOSS-style)
# ---------------------------------------------------------------------------

def _tokenize_code(code: str) -> list[str]:
    """
    Produce a normalised token stream from source code.
    Strips comments, normalises identifiers and literals.
    """
    # Remove single-line comments
    code = re.sub(r'#[^\n]*', '', code)
    # Remove string literals (keep their type marker)
    code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')', 'STR', code, flags=re.DOTALL)
    # Normalise numbers
    code = re.sub(r'\b\d+(\.\d+)?\b', 'NUM', code)
    # Split on whitespace and punctuation, keep meaningful tokens
    tokens = re.findall(r'[A-Za-z_]\w*|[+\-*/%=<>!&|^~]+|[(){}\[\],.:;]', code)
    # Normalise identifier names (preserve keywords)
    python_keywords = {
        'False','None','True','and','as','assert','async','await',
        'break','class','continue','def','del','elif','else','except',
        'finally','for','from','global','if','import','in','is',
        'lambda','nonlocal','not','or','pass','raise','return',
        'try','while','with','yield'
    }
    return [t if t in python_keywords else 'ID' for t in tokens]


def _kgrams(tokens: list[str], k: int) -> list[tuple]:
    """Generate overlapping k-grams from token list."""
    return [tuple(tokens[i: i + k]) for i in range(len(tokens) - k + 1)]


def _winnow(tokens: list[str], k: int, window: int) -> set[int]:
    """
    Winnowing algorithm: select the minimum hash in each sliding window.
    Returns a set of selected fingerprint hashes.
    """
    grams = _kgrams(tokens, k)
    if not grams:
        return set()

    # Hash each k-gram
    hashes = [hash(g) for g in grams]

    fingerprints: set[int] = set()
    prev_min_idx = -1

    for i in range(len(hashes) - window + 1):
        window_hashes = hashes[i: i + window]
        min_val = min(window_hashes)
        min_idx = i + window_hashes.index(min_val)

        if min_idx != prev_min_idx:
            fingerprints.add(min_val)
            prev_min_idx = min_idx

    return fingerprints


def _winnow_similarity(code_a: str, code_b: str) -> float:
    """
    Compute Jaccard similarity between Winnowing fingerprint sets.
    Returns 0-1 where 1 = identical fingerprints.
    """
    tok_a = _tokenize_code(code_a)
    tok_b = _tokenize_code(code_b)

    fp_a = _winnow(tok_a, KGRAM_SIZE, WINDOW_SIZE)
    fp_b = _winnow(tok_b, KGRAM_SIZE, WINDOW_SIZE)

    if not fp_a or not fp_b:
        return 0.0

    intersection = len(fp_a & fp_b)
    union = len(fp_a | fp_b)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# MAIN AGENT CLASS
# ---------------------------------------------------------------------------

class PlagiarismAgent:
    """
    Compares a set of code submissions and returns pairwise plagiarism scores.

    Usage — batch mode (compare all submissions for one question):
        agent = PlagiarismAgent()
        results = agent.analyze_batch(submissions)
        # submissions = [{"candidate_id": "C001", "code": "..."}, ...]

    Usage — single pair:
        result = agent.compare_pair(code_a, id_a, code_b, id_b)
    """

    def __init__(
        self,
        threshold: float = PLAGIARISM_THRESHOLD,
        ast_weight: float = 0.50,
        winnow_weight: float = 0.50,
    ):
        self.threshold     = threshold
        self.ast_weight    = ast_weight
        self.winnow_weight = winnow_weight

    # ------------------------------------------------------------------
    def compare_pair(
        self,
        code_a: str,
        id_a: str,
        code_b: str,
        id_b: str,
    ) -> dict[str, Any]:
        """
        Compare two submissions.

        Returns:
            {
              "pair": (id_a, id_b),
              "plagiarism_score": float,   # 0.0 – 1.0
              "is_plagiarised": bool,
              "ast_similarity": float,
              "token_similarity": float,
              "verdict": str,
            }
        """
        ast_sim    = _ast_subtree_similarity(code_a, code_b)
        token_sim  = _winnow_similarity(code_a, code_b)

        combined = (
            self.ast_weight    * ast_sim +
            self.winnow_weight * token_sim
        )
        score = round(min(max(combined, 0.0), 1.0), 4)

        if score >= 0.85:
            verdict = "CONFIRMED_COPY"
        elif score >= self.threshold:
            verdict = "LIKELY_PLAGIARISM"
        elif score >= 0.35:
            verdict = "SUSPICIOUS"
        else:
            verdict = "CLEAN"

        return {
            "pair":              (id_a, id_b),
            "plagiarism_score":  score,
            "is_plagiarised":    score >= self.threshold,
            "ast_similarity":    round(ast_sim,   4),
            "token_similarity":  round(token_sim, 4),
            "verdict":           verdict,
        }

    # ------------------------------------------------------------------
    def analyze_batch(
        self,
        submissions: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Cross-compare every pair in a batch of submissions.

        Args:
            submissions: list of {"candidate_id": str, "code": str}

        Returns:
            {
              "total_pairs":     int,
              "flagged_pairs":   int,
              "results":         list[compare_pair results],
              "flagged":         list[compare_pair results],  # only flagged
              "suspect_candidates": list[str],  # IDs involved in any flag
            }
        """
        if len(submissions) < 2:
            return {
                "total_pairs":        0,
                "flagged_pairs":      0,
                "results":            [],
                "flagged":            [],
                "suspect_candidates": [],
            }

        all_results: list[dict] = []

        for sub_a, sub_b in combinations(submissions, 2):
            result = self.compare_pair(
                sub_a["code"], sub_a["candidate_id"],
                sub_b["code"], sub_b["candidate_id"],
            )
            all_results.append(result)

        flagged = [r for r in all_results if r["is_plagiarised"]]
        suspect_ids: set[str] = set()
        for r in flagged:
            suspect_ids.update(r["pair"])

        return {
            "total_pairs":        len(all_results),
            "flagged_pairs":      len(flagged),
            "results":            all_results,
            "flagged":            flagged,
            "suspect_candidates": sorted(suspect_ids),
        }
