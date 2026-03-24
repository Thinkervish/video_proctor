import re
import ast
import math
import keyword
from collections import Counter


class AIDetectionAgent:
    """
    Multi-signal AI code detection agent.

    Scoring breakdown (total = 1.0):
      - Docstring quality       0.20
      - Naming conventions      0.20
      - Structural patterns     0.20
      - Comment style           0.15
      - Complexity & entropy    0.15
      - Code style signals      0.10

    Threshold: score >= 0.5 → is_ai_generated = True
    """

    # Common AI-favoured generic names
    _AI_VAR_NAMES = {
        "result", "results", "output", "outputs", "data", "value", "values",
        "item", "items", "element", "elements", "temp", "tmp", "res",
        "num", "nums", "lst", "arr", "obj", "ret", "val", "flag",
        "count", "total", "current", "node", "key", "keys", "pair",
        "left", "right", "mid", "start", "end", "idx", "pos",
    }

    # Phrases AI loves putting in docstrings / comments
    _AI_PHRASES = [
        r"initializes?\s+the",
        r"returns?\s+the\s+(result|value|output|list|dict|string)",
        r"this\s+(function|method|class)\s+(takes|accepts|handles|processes|checks|returns)",
        r"helper\s+(function|method)\s+to",
        r"iterates?\s+(over|through)",
        r"computes?\s+(the\s+)?(sum|product|result|value)",
        r"represents?\s+(a|an|the)",
        r"ensures?\s+that",
        r"note\s*:\s*this",
        r"main\s+(function|entry\s+point)",
        r"example\s+usage\s*:",
        r"args\s*:\s*\n",
        r"returns\s*:\s*\n",
        r"raises\s*:\s*\n",
    ]

    def detect(self, code: str) -> dict:
        lines = [l.rstrip() for l in code.split("\n")]
        non_empty = [l for l in lines if l.strip()]

        scores = {}

        scores["docstring_quality"]   = self._score_docstrings(code, non_empty)
        scores["naming_conventions"]  = self._score_naming(code)
        scores["structural_patterns"] = self._score_structure(code, non_empty)
        scores["comment_style"]       = self._score_comments(lines)
        scores["complexity_entropy"]  = self._score_complexity(code, lines, non_empty)
        scores["code_style"]          = self._score_code_style(lines, non_empty)

        weights = {
            "docstring_quality":   0.20,
            "naming_conventions":  0.20,
            "structural_patterns": 0.20,
            "comment_style":       0.15,
            "complexity_entropy":  0.15,
            "code_style":          0.10,
        }

        ai_score = sum(scores[k] * weights[k] for k in weights)
        ai_score = round(min(ai_score, 1.0), 3)

        return {
            "ai_score": ai_score,
            "is_ai_generated": ai_score >= 0.5,
            "confidence": self._confidence_label(ai_score),
            "feature_scores": {k: round(v, 3) for k, v in scores.items()},
        }

    # ------------------------------------------------------------------
    # 1. DOCSTRING QUALITY  (0.0 – 1.0)
    #    AI almost always writes structured, verbose docstrings with
    #    Args/Returns/Raises sections and overly literal descriptions.
    # ------------------------------------------------------------------
    def _score_docstrings(self, code: str, non_empty: list) -> float:
        if not non_empty:
            return 0.0

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._score_docstrings_regex(code)

        docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                ds = ast.get_docstring(node)
                if ds:
                    docstrings.append(ds)

        if not docstrings:
            return 0.0

        signal = 0.0
        total_funcs = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

        # AI almost always docstrings every function
        if total_funcs > 0:
            docstring_nodes = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and ast.get_docstring(n)
            )
            coverage = docstring_nodes / total_funcs
            if coverage > 0.8:
                signal += 0.4
            elif coverage > 0.5:
                signal += 0.2

        # AI uses structured sections (Args:, Returns:, Raises:, Parameters:)
        structured = sum(
            1 for ds in docstrings
            if re.search(r'\b(Args|Returns|Raises|Parameters|Attributes|Note|Example|Examples)\s*:', ds)
        )
        if structured / max(len(docstrings), 1) > 0.5:
            signal += 0.3

        # AI phrases inside docstrings
        all_ds_text = " ".join(docstrings).lower()
        phrase_hits = sum(
            1 for p in self._AI_PHRASES
            if re.search(p, all_ds_text, re.IGNORECASE)
        )
        signal += min(phrase_hits / 5, 0.3)

        return min(signal, 1.0)

    def _score_docstrings_regex(self, code: str) -> float:
        """Fallback for non-parseable code."""
        ds_blocks = re.findall(r'"""[\s\S]*?"""', code)
        if not ds_blocks:
            return 0.0
        structured = sum(
            1 for b in ds_blocks
            if re.search(r'\b(Args|Returns|Raises|Parameters)\s*:', b)
        )
        return min(structured / max(len(ds_blocks), 1) + 0.1, 1.0)

    # ------------------------------------------------------------------
    # 2. NAMING CONVENTIONS  (0.0 – 1.0)
    #    AI names are generic, consistent, and oddly "textbook perfect".
    # ------------------------------------------------------------------
    def _score_naming(self, code: str) -> float:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._score_naming_regex(code)

        func_names, var_names, arg_names = [], [], []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_names.append(node.name)
                for a in node.args.args:
                    arg_names.append(a.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                var_names.append(node.id)

        all_names = func_names + var_names + arg_names
        if not all_names:
            return 0.0

        signal = 0.0

        # Generic AI variable names
        ai_generic = sum(1 for n in all_names if n.lower() in self._AI_VAR_NAMES)
        generic_ratio = ai_generic / len(all_names)
        signal += min(generic_ratio * 1.5, 0.4)

        # Perfect snake_case everywhere (AI rarely breaks the style guide)
        valid_snake = sum(
            1 for n in all_names
            if re.match(r'^[a-z][a-z0-9_]*$', n) or n.startswith('_')
        )
        snake_ratio = valid_snake / len(all_names)
        if snake_ratio > 0.95:
            signal += 0.2

        # Single-letter vars are rare in AI (it prefers verbose names)
        single_letter = sum(1 for n in var_names if len(n) == 1 and n not in ('i', 'j', 'k', 'x', 'y', 'n'))
        if len(var_names) > 0 and single_letter / len(var_names) < 0.05:
            signal += 0.1

        # Functions named exactly what they do: get_, set_, check_, is_, has_, calculate_, compute_
        ai_func_prefixes = ('get_', 'set_', 'check_', 'is_', 'has_', 'calculate_', 'compute_', 'process_', 'handle_', 'create_', 'build_', 'validate_', 'update_', 'find_', 'load_', 'save_', 'parse_', 'format_', 'convert_', 'extract_')
        if func_names:
            prefixed = sum(1 for f in func_names if any(f.startswith(p) for p in ai_func_prefixes))
            if prefixed / len(func_names) > 0.5:
                signal += 0.3

        return min(signal, 1.0)

    def _score_naming_regex(self, code: str) -> float:
        names = re.findall(r'\b([a-zA-Z_]\w*)\b', code)
        names = [n for n in names if not keyword.iskeyword(n)]
        if not names:
            return 0.0
        generic = sum(1 for n in names if n.lower() in self._AI_VAR_NAMES)
        return min(generic / len(names) * 2, 1.0)

    # ------------------------------------------------------------------
    # 3. STRUCTURAL PATTERNS  (0.0 – 1.0)
    #    AI writes textbook-clean, symmetric structures.
    # ------------------------------------------------------------------
    def _score_structure(self, code: str, non_empty: list) -> float:
        signal = 0.0

        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None

        if tree:
            funcs = [
                n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]

            # Every function has an explicit return statement (AI habit)
            if funcs:
                returns_all = sum(
                    1 for f in funcs
                    if any(isinstance(n, ast.Return) for n in ast.walk(f))
                )
                if returns_all / len(funcs) > 0.85:
                    signal += 0.2

            # AI loves type annotations
            total_args = sum(len(f.args.args) for f in funcs)
            annotated_args = sum(
                sum(1 for a in f.args.args if a.annotation) for f in funcs
            )
            if total_args > 0 and annotated_args / total_args > 0.7:
                signal += 0.25

            # Return type annotations
            annotated_returns = sum(1 for f in funcs if f.returns)
            if funcs and annotated_returns / len(funcs) > 0.7:
                signal += 0.15

        # AI over-uses list/dict comprehensions
        comprehensions = len(re.findall(r'\[.+\s+for\s+\w+\s+in\s+', code))
        if comprehensions >= 2:
            signal += min(comprehensions * 0.05, 0.2)

        # AI loves enumerate(), zip(), isinstance(), isinstance checks, .items()
        ai_builtins = len(re.findall(
            r'\b(enumerate|zip|isinstance|hasattr|getattr|setattr|any|all|map|filter|sorted|reversed)\s*\(',
            code
        ))
        if ai_builtins >= 3:
            signal += min(ai_builtins * 0.03, 0.2)

        return min(signal, 1.0)

    # ------------------------------------------------------------------
    # 4. COMMENT STYLE  (0.0 – 1.0)
    #    AI comments are frequent, full-sentence, and redundant.
    # ------------------------------------------------------------------
    def _score_comments(self, lines: list) -> float:
        comment_lines = [l for l in lines if l.strip().startswith("#")]
        code_lines    = [l for l in lines if l.strip() and not l.strip().startswith("#")]

        if not code_lines or not comment_lines:
            return 0.0

        signal = 0.0

        comment_ratio = len(comment_lines) / len(code_lines)

        # AI comments heavily — roughly 1 comment per 3-5 lines of code
        if 0.15 < comment_ratio < 0.6:
            signal += 0.3
        elif comment_ratio >= 0.6:
            signal += 0.5  # extremely dense comments = strong AI signal

        # AI comments are full sentences starting with capital letter
        full_sentence = sum(
            1 for l in comment_lines
            if re.match(r'#\s+[A-Z]', l.strip())
        )
        if full_sentence / len(comment_lines) > 0.5:
            signal += 0.3

        # AI comments state the obvious: "# Initialize the list", "# Return the result"
        obvious_patterns = [
            r'#\s*(initialize|initialise)',
            r'#\s*return\s+the',
            r'#\s*(create|creating)\s+(a|an|the)',
            r'#\s*(check|checking)\s+(if|whether)',
            r'#\s*(iterate|loop)\s+(over|through)',
            r'#\s*(calculate|compute|get)\s+the',
            r'#\s*(add|append|insert)\s+(the|to)',
            r'#\s*(print|display|show)\s+the',
            r'#\s*(define|defining)\s+(a|the)',
            r'#\s*step\s+\d+',
        ]
        obvious_hits = sum(
            1 for l in comment_lines
            if any(re.search(p, l, re.IGNORECASE) for p in obvious_patterns)
        )
        if obvious_hits / len(comment_lines) > 0.2:
            signal += 0.2

        return min(signal, 1.0)

    # ------------------------------------------------------------------
    # 5. COMPLEXITY & ENTROPY  (0.0 – 1.0)
    #    AI code is mid-complexity, high lexical diversity, very uniform.
    # ------------------------------------------------------------------
    def _score_complexity(self, code: str, lines: list, non_empty: list) -> float:
        if not non_empty:
            return 0.0

        tokens = re.findall(r'\w+', code)
        if not tokens:
            return 0.0

        signal = 0.0

        # Shannon entropy of token distribution
        # AI generates tokens from a learned distribution — slightly more uniform
        freq = Counter(tokens)
        total = sum(freq.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        # Normalise: typical code entropy 3.5–5.5; AI sits 4.0–5.0
        norm_entropy = (entropy - 3.5) / 2.0
        if 0.2 < norm_entropy < 0.8:
            signal += 0.3

        # Line length uniformity (AI lines are suspiciously consistent)
        lengths = [len(l) for l in non_empty]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        if std_dev < 18:  # very uniform → AI
            signal += 0.3
        elif std_dev < 25:
            signal += 0.15

        # Nesting depth: AI rarely goes deeper than 3 levels
        max_indent = max((len(l) - len(l.lstrip())) for l in non_empty) // 4
        if max_indent <= 3:
            signal += 0.2
        elif max_indent >= 6:
            signal -= 0.1  # deeply nested = probably human

        # Magic numbers: humans use them, AI tends to assign them to named constants
        magic_numbers = re.findall(r'(?<!["\'\'\w])(?<!=)\s*\b([2-9]\d{1,4})\b(?!\s*["\'\'])', code)
        if len(magic_numbers) == 0 and len(non_empty) > 10:
            signal += 0.2  # no magic numbers in substantial code = AI signal

        return min(max(signal, 0.0), 1.0)

    # ------------------------------------------------------------------
    # 6. CODE STYLE SIGNALS  (0.0 – 1.0)
    #    Fine-grained style habits that are very AI-specific.
    # ------------------------------------------------------------------
    def _score_code_style(self, lines: list, non_empty: list) -> float:
        if not non_empty:
            return 0.0

        signal = 0.0
        code = "\n".join(lines)

        # AI always puts spaces around operators (PEP-8 perfect)
        # Humans often write x=1, x+=1 without spaces
        operator_spaced = len(re.findall(r'\w\s[+\-*/]=?\s\w', code))
        operator_unspaced = len(re.findall(r'\w[+\-*/]=\w', code))
        total_ops = operator_spaced + operator_unspaced
        if total_ops > 0 and operator_spaced / total_ops > 0.95:
            signal += 0.3

        # AI uses f-strings correctly and consistently
        fstrings = len(re.findall(r'f["\']', code))
        old_format = len(re.findall(r'\.format\(', code))
        percent_fmt = len(re.findall(r'%[sdf]', code))
        if fstrings > 0 and old_format == 0 and percent_fmt == 0:
            signal += 0.2  # consistent modern style = AI

        # AI almost never has trailing whitespace
        trailing_ws = sum(1 for l in lines if l != l.rstrip())
        if len(lines) > 5 and trailing_ws == 0:
            signal += 0.2

        # AI adds `if __name__ == "__main__":` blocks in scripts
        if '__name__' in code and '__main__' in code:
            signal += 0.15

        # AI uses `pass` in empty exception handlers (try/except pass)
        silent_excepts = len(re.findall(r'except[^:]*:\s*(?:#[^\n]*\n\s*)*pass', code))
        if silent_excepts >= 1:
            signal += 0.15

        return min(signal, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _confidence_label(self, score: float) -> str:
        if score >= 0.80:
            return "high"
        elif score >= 0.60:
            return "medium"
        elif score >= 0.40:
            return "low"
        else:
            return "unlikely"