import ast
import re
import math
import statistics
from collections import Counter

# ── CodeBERT Global Load ─────────────────────────────────────────────

_codebert_model = None
_codebert_ready = False

def _load_codebert():
    global _codebert_model, _codebert_ready

    if _codebert_ready:
        return True

    try:
        from sentence_transformers import SentenceTransformer

        print("[AIDetector] Loading CodeBERT model...")
        _codebert_model = SentenceTransformer("microsoft/codebert-base")

        _codebert_ready = True
        print("[AIDetector] CodeBERT loaded successfully ✅")

        return True

    except Exception as e:
        print(f"[AIDetector] CodeBERT load failed: {e}")
        _codebert_ready = False
        return False

# ── Reference Code Samples ───────────────────────────────────────────

_AI_REFERENCE_CODES = [
    '''
def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    total = 0
    for number in numbers:
        total += number
    return total
''',
    '''
def find_maximum(lst):
    """Find the maximum value in a list."""
    if not lst:
        return None
    max_value = lst[0]
    for item in lst:
        if item > max_value:
            max_value = item
    return max_value
''',
    '''
def is_palindrome(string):
    """Check if a string is a palindrome."""
    cleaned = string.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
'''
]

_HUMAN_REFERENCE_CODES = [
    '''
def calc(nums):
    s = 0
    for n in nums: 
        s += n
    return s
''',
    '''
def maxval(lst):
    if not lst: 
        return None
    m = lst[0]
    for x in lst:
        if x > m: 
            m = x
    return m
''',
    '''
def chkpal(s):
    s = s.lower().replace(" ","")
    return s == s[::-1]
'''
]

class AICodeDetectorAgent:

    def __init__(self):
        self._ai_embeddings = None
        self._human_embeddings = None

        loaded = _load_codebert()

        if loaded:
            print("[AIDetector] Generating reference embeddings...")

            self._ai_embeddings = _codebert_model.encode(
                _AI_REFERENCE_CODES,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            self._human_embeddings = _codebert_model.encode(
                _HUMAN_REFERENCE_CODES,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            print("[AIDetector] Reference embeddings ready 🚀")

    # ── Main Analyze Function ────────────────────────────────────────
    def analyze(self, code: str) -> dict:
        lines = code.splitlines()
        nonempty = [l for l in lines if l.strip()]

        if len(nonempty) < 3:
            return {
                "ai_probability": 0.0,
                "signals": {},
                "note": "Too short to analyze."
            }

        signals = {}

        signals["perplexity_proxy"] = self._perplexity_proxy(code)
        signals["burstiness"] = self._burstiness(nonempty)
        signals["ast_entropy"] = self._ast_entropy(code)
        signals["ai_style_fingerprint"] = self._ai_style_fingerprint(code)
        signals["comment_density"] = self._comment_density(lines)
        signals["naming_consistency"] = self._naming_consistency(code)
        signals["line_length_variance"] = self._line_length_variance(nonempty)
        signals["structural_symmetry"] = self._structural_symmetry(code)

        using_codebert = _codebert_ready

        if using_codebert:
            signals["codebert_similarity"] = self._codebert_similarity(code)

        if using_codebert:
            weights = {
                "codebert_similarity": 0.35,
                "perplexity_proxy": 0.15,
                "burstiness": 0.12,
                "ast_entropy": 0.12,
                "ai_style_fingerprint": 0.10,
                "naming_consistency": 0.08,
                "comment_density": 0.04,
                "line_length_variance": 0.02,
                "structural_symmetry": 0.02,
            }
        else:
            weights = {
                "perplexity_proxy": 0.28,
                "burstiness": 0.22,
                "ast_entropy": 0.18,
                "ai_style_fingerprint": 0.15,
                "naming_consistency": 0.08,
                "comment_density": 0.05,
                "line_length_variance": 0.02,
                "structural_symmetry": 0.02,
            }

        ai_prob = sum(signals.get(k, 0) * w for k, w in weights.items())
        ai_prob = round(min(max(ai_prob, 0.0), 1.0), 4)

        return {
            "ai_probability": ai_prob,
            "flagged": ai_prob >= 0.65,
            "signals": {k: round(v, 4) for k, v in signals.items()},
            "verdict": self._verdict(ai_prob),
            "codebert_used": using_codebert,
        }

    # ── CodeBERT Similarity ─────────────────────────────────────────
    def _codebert_similarity(self, code: str) -> float:
        try:
            import numpy as np

            code_emb = _codebert_model.encode(
                [code],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]

            def cos_sim(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

            avg_ai = sum(cos_sim(code_emb, e) for e in self._ai_embeddings) / len(self._ai_embeddings)
            avg_human = sum(cos_sim(code_emb, e) for e in self._human_embeddings) / len(self._human_embeddings)

            ai_score = avg_ai / (avg_ai + avg_human + 1e-9)

            return float(min(max(ai_score, 0.0), 1.0))

        except Exception as e:
            print(f"[CodeBERT signal error] {e}")
            return 0.5

    # ── AST Entropy ─────────────────────────────────────────────────
    def _ast_entropy(self, code: str) -> float:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.5

        node_counts = Counter(type(n).__name__ for n in ast.walk(tree))
        total = sum(node_counts.values())

        if total == 0:
            return 0.5

        probs = [c / total for c in node_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        return max(0.0, 1.0 - entropy / 6.0)

    # ── AI Style Fingerprint ────────────────────────────────────────
    def _ai_style_fingerprint(self, code: str) -> float:
        score = 0.0

        try:
            tree = ast.parse(code)
            fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

            if fns:
                doc_ratio = sum(
                    1 for f in fns
                    if f.body
                    and isinstance(f.body[0], ast.Expr)
                    and isinstance(f.body[0].value, ast.Constant)
                    and isinstance(f.body[0].value.value, str)
                ) / len(fns)

                score += min(doc_ratio / 0.7, 1.0) * 0.30

        except SyntaxError:
            pass

        if re.search(r'def main\(\)', code):
            score += 0.15

        if re.search(r"if __name__\s*==\s*['\"]__main__['\"]", code):
            score += 0.10

        if '"""' in code or "'''" in code:
            score += 0.10

        try:
            tree = ast.parse(code)
            names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and len(n.id) > 2]

            if names:
                avg_len = sum(len(n) for n in names) / len(names)
                score += min(avg_len / 12.0, 1.0) * 0.25

        except SyntaxError:
            pass

        return min(score, 1.0)

    # ── Remaining Signals ───────────────────────────────────────────
    def _perplexity_proxy(self, code: str) -> float:
        tokens = re.findall(r'[a-zA-Z_]\w*|[0-9]+|[^\w\s]', code)

        if len(tokens) < 10:
            return 0.5

        counts = Counter(tokens)
        total = len(tokens)

        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        return 1.0 - min(entropy / 6.0, 1.0)

    def _burstiness(self, nonempty_lines):
        lengths = [len(l.rstrip()) for l in nonempty_lines]

        if len(lengths) < 3:
            return 0.5

        mean = statistics.mean(lengths)
        std = statistics.stdev(lengths)

        if mean == 0:
            return 0.5

        cv = std / mean

        return max(0.0, 1.0 - cv * 1.5)

    def _comment_density(self, lines):
        total = len([l for l in lines if l.strip()])
        comments = len([l for l in lines if l.strip().startswith('#')])

        if total == 0:
            return 0.0

        return min((comments / total) / 0.35, 1.0)

    def _naming_consistency(self, code):
        try:
            tree = ast.parse(code)

            names = [
                n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and len(n.id) > 1
            ]

            if not names:
                return 0.5

            def is_snake(n):
                return bool(re.fullmatch(r'[a-z][a-z0-9_]*', n))

            snake_ratio = sum(1 for n in names if is_snake(n)) / len(names)

            return min(snake_ratio / 0.90, 1.0) * snake_ratio

        except SyntaxError:
            return 0.5

    def _line_length_variance(self, nonempty_lines):
        lengths = [len(l.rstrip()) for l in nonempty_lines]

        if len(lengths) < 3:
            return 0.5

        variance = statistics.variance(lengths)

        return max(0.0, 1.0 - variance / 500.0)

    def _structural_symmetry(self, code):
        try_count = len(re.findall(r'\btry\s*:', code))
        except_count = len(re.findall(r'\bexcept\b', code))
        if_count = len(re.findall(r'\bif\s+', code))
        else_count = len(re.findall(r'\belse\s*:', code))

        score = 0.0
        checks = 0

        if try_count > 0:
            score += min(except_count / try_count, 1.0)
            checks += 1

        if if_count > 0:
            score += min((else_count / if_count) * 1.5, 1.0)
            checks += 1

        return (score / checks) if checks > 0 else 0.0

    def _verdict(self, prob: float):
        if prob >= 0.80:
            return "Almost certainly AI-generated"

        if prob >= 0.65:
            return "Likely AI-generated"

        if prob >= 0.45:
            return "Possibly AI-assisted"

        if prob >= 0.25:
            return "Likely human-written"

        return "Almost certainly human-written"
