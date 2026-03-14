import ast
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD    = 0.80   # above this → plagiarism flagged
DB_PATH      = "code_database/stored_codes.json"

class PlagiarismAgent:
    def __init__(self):
        self.db = self._load_db()

    # ── DB helpers ─────────────────────────────────────────────────────────
    def _load_db(self):
        if os.path.exists(DB_PATH):
            with open(DB_PATH) as f:
                return json.load(f)
        return {}

    def _save_db(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, "w") as f:
            json.dump(self.db, f, indent=2)

    def store_code(self, student_id: str, code: str):
        """Store a submission for future comparisons."""
        self.db[student_id] = self._normalize(code)
        self._save_db()

    # ── Normalization — strips boilerplate before comparing ────────────────
    def _normalize(self, code: str) -> str:
        """
        AST-based normalization:
        1. Parse to AST → unparse back (removes comments, normalizes whitespace)
        2. Rename all variable/function/class names → generic placeholders
           so copy-paste-rename is still caught
        3. Remove docstrings
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Not valid Python — fall back to text normalization
            return self._text_normalize(code)

        tree = _RenameTransformer().visit(tree)
        ast.fix_missing_locations(tree)

        try:
            normalized = ast.unparse(tree)
        except Exception:
            normalized = self._text_normalize(code)

        return normalized

    def _text_normalize(self, code: str) -> str:
        """Fallback: strip comments, collapse whitespace."""
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    # ── Core analysis ──────────────────────────────────────────────────────
    def analyze(self, code: str) -> dict:
        normalized = self._normalize(code)

        if not self.db:
            return {
                "plagiarism_score": 0.0,
                "flagged":          False,
                "matched_student":  None,
                "note":             "Database empty — no comparisons possible.",
            }

        corpus      = list(self.db.values())
        student_ids = list(self.db.keys())

        try:
            vectorizer  = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            tfidf       = vectorizer.fit_transform(corpus + [normalized])
            scores      = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
        except Exception as e:
            return {"plagiarism_score": 0.0, "flagged": False,
                    "matched_student": None, "note": str(e)}

        max_score = float(max(scores))
        max_idx   = int(scores.argmax())

        return {
            "plagiarism_score": round(max_score, 4),
            "flagged":          max_score >= THRESHOLD,
            "matched_student":  student_ids[max_idx] if max_score >= THRESHOLD else None,
            "threshold":        THRESHOLD,
        }


# ── AST visitor: renames all identifiers to generic tokens ────────────────
class _RenameTransformer(ast.NodeTransformer):
    """
    Replaces:
      variable names → v0, v1, v2 ...
      function names → f0, f1, f2 ...
      class names    → C0, C1, C2 ...
    This catches plagiarism via rename-and-submit.
    """
    def __init__(self):
        self._var_map  = {}
        self._fn_map   = {}
        self._cls_map  = {}
        self._vc = self._fc = self._cc = 0

    def _var(self, n):
        if n not in self._var_map:
            self._var_map[n] = f"v{self._vc}"; self._vc += 1
        return self._var_map[n]

    def _fn(self, n):
        if n not in self._fn_map:
            self._fn_map[n] = f"f{self._fc}"; self._fc += 1
        return self._fn_map[n]

    def _cls(self, n):
        if n not in self._cls_map:
            self._cls_map[n] = f"C{self._cc}"; self._cc += 1
        return self._cls_map[n]

    def visit_Name(self, node):
        node.id = self._var(node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self._fn(node.name)
        # Remove docstring
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        node.name = self._cls(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        node.arg = self._var(node.arg)
        return node