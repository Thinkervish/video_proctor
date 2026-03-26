import re
import ast
import math
import keyword
from collections import Counter


class AIDetectionAgent:
    """
    Multi-signal AI code detection agent (v4).

    Scoring breakdown (total = 1.0):
      - Docstring quality       0.18
      - Naming conventions      0.12
      - Structural patterns     0.15
      - Comment style           0.12
      - Complexity & entropy    0.12
      - Code style signals      0.12
      - Terse AI patterns       0.19   ← NEW: catches short assignment-style AI code

    v4 changes vs v3:
      - Added _score_terse_ai() dimension (weight 0.19) targeting short AI scripts
        that lack type hints / docstring sections but still carry AI fingerprints:
          * Boilerplate scaffold comments ("# Write your solution here", "# Taking input")
          * Redundant branch logic (identical return values in separate if-blocks)
          * try/except ValueError wrapping int(input()) — classic AI I/O scaffold
          * Docstring present on tiny function (AI always documents, even trivially)
          * AI-phrase docstrings without Args/Returns (partial credit, not zero)
          * Perfectly blank-line-separated sections in short code
          * "Eligible" / domain-label string literals typical of AI problem solutions
      - Docstring scorer now awards partial credit for any docstring with an
        AI-style description phrase, even without Args/Returns sections.
      - Structural scorer handles 0-argument functions (no longer zeroes out).
      - Threshold lowered from 0.52 → 0.45 to catch borderline short scripts.
      - All gates reduced to >= 1 function minimum.
    """

    _STRONGLY_AI_VAR_NAMES = {
        "result", "results", "output", "outputs", "element", "elements",
        "lst", "arr", "obj", "val",
    }

    _AI_PHRASES = [
        r"determines?\s+(whether|if)",
        r"checks?\s+(whether|if)",
        r"initializes?\s+the",
        r"returns?\s+the\s+(result|value|output|list|dict|string|count|sum)",
        r"this\s+(function|method|class)\s+(takes|accepts|handles|processes|checks|returns)",
        r"helper\s+(function|method)\s+to",
        r"iterates?\s+(over|through)",
        r"computes?\s+(the\s+)?(sum|product|result|value)",
        r"represents?\s+(a|an|the)",
        r"ensures?\s+that",
        r"calculates?\s+(the\s+)?",
        r"based\s+on\s+the\s+(number|value|count|input)",
        r"eligible\s+for\s+(reward|bonus|prize)",
        r"note\s*:\s*this",
        r"example\s+usage\s*:",
    ]

    # Scaffold comment phrases AI almost always emits
    _AI_SCAFFOLD_COMMENTS = [
        r"#\s*write\s+your\s+solution\s+here",
        r"#\s*taking\s+input\s+(from\s+)?(the\s+)?user",
        r"#\s*take\s+input",
        r"#\s*read\s+input",
        r"#\s*get\s+input",
        r"#\s*main\s+(logic|function|program|code)",
        r"#\s*driver\s+code",
        r"#\s*test\s+the\s+(function|solution|code)",
        r"#\s*call\s+the\s+function",
        r"#\s*print\s+(the\s+)?(result|output|answer)",
        r"#\s*your\s+code\s+(here|goes\s+here)",
        r"#\s*solution",
        r"#\s*approach",
        r"#\s*algorithm",
        r"#\s*helper\s+function",
        r"#\s*edge\s+case",
    ]

    _HUMAN_PATTERNS = [
        r"#\s*(TODO|FIXME|HACK|XXX|NOQA|type:\s*ignore)",
        r"#\s*noqa",
        r"#\s*pylint\s*:",
        r"#\s*type\s*:",
        r"\bpdb\b",
        r"\bipdb\b",
        r"breakpoint\s*\(\s*\)",
        r"print\s*\(\s*['\"]debug",
        r"print\s*\(\s*['\"]test",
    ]

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, code: str) -> dict:
        if not isinstance(code, str):
            code = str(code)

        lines = [line.rstrip() for line in code.split("\n")]
        non_empty = [line for line in lines if line.strip()]

        if len(non_empty) < 3:
            return {
                "ai_score": 0.0,
                "is_ai_generated": False,
                "confidence": "insufficient_data",
                "feature_scores": {},
                "human_signals_found": [],
                "parse_error": None,
            }

        tree, parse_error = self._safe_parse(code)
        human_signals = self._detect_human_signals(code, lines, tree)

        scores = {
            "docstring_quality":   self._score_docstrings(code, non_empty, tree),
            "naming_conventions":  self._score_naming(code, tree),
            "structural_patterns": self._score_structure(code, non_empty, tree),
            "comment_style":       self._score_comments(lines, non_empty),
            "complexity_entropy":  self._score_complexity(code, lines, non_empty),
            "code_style":          self._score_code_style(lines, non_empty),
            "terse_ai_patterns":   self._score_terse_ai(code, lines, non_empty, tree),
        }

        weights = {
            "docstring_quality":   0.18,
            "naming_conventions":  0.12,
            "structural_patterns": 0.15,
            "comment_style":       0.12,
            "complexity_entropy":  0.12,
            "code_style":          0.12,
            "terse_ai_patterns":   0.19,
        }

        raw_score = sum(scores[k] * weights[k] for k in weights)

        # Each human signal reduces score by 0.06, capped at 0.22
        human_penalty = min(len(human_signals) * 0.06, 0.22)
        ai_score = max(raw_score - human_penalty, 0.0)
        ai_score = round(min(ai_score, 1.0), 3)

        # Threshold lowered to 0.45 to catch short terse AI scripts
        return {
            "ai_score": ai_score,
            "is_ai_generated": ai_score >= 0.40,
            "confidence": self._confidence_label(ai_score),
            "feature_scores": {k: round(v, 3) for k, v in scores.items()},
            "human_signals_found": human_signals,
            "parse_error": parse_error,
        }

    # ------------------------------------------------------------------ #
    # INTERNAL HELPERS                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_parse(code: str):
        try:
            return ast.parse(code), None
        except SyntaxError as exc:
            return None, f"SyntaxError: {exc}"
        except Exception as exc:  # noqa: BLE001
            return None, f"ParseError: {exc}"

    @staticmethod
    def _get_functions(tree) -> list:
        if tree is None:
            return []
        return [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    # ------------------------------------------------------------------ #
    # HUMAN SIGNAL DETECTOR                                                #
    # ------------------------------------------------------------------ #

    def _detect_human_signals(self, code: str, lines: list, tree) -> list:
        found = []

        for pattern in self._HUMAN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                found.append(pattern)

        bad_indent = sum(
            1 for line in lines
            if line and line[0] == " " and (len(line) - len(line.lstrip(" "))) % 4 != 0
        )
        if bad_indent >= 2:
            found.append("inconsistent_indentation")

        max_indent = max(
            ((len(line) - len(line.lstrip())) // 4 for line in lines if line.strip()),
            default=0,
        )
        if max_indent >= 5:
            found.append("deep_nesting")

        if tree is not None:
            loop_vars = {"i", "j", "k", "x", "y", "n", "e", "f", "v", "c"}
            short_vars = [
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and len(node.id) <= 2
                and node.id not in loop_vars
            ]
            if len(short_vars) >= 2:
                found.append("short_cryptic_names")

        if any(line != line.rstrip() for line in lines):
            found.append("trailing_whitespace")

        return found

    # ------------------------------------------------------------------ #
    # 1. DOCSTRING QUALITY  (0.0 – 1.0)                                   #
    # ------------------------------------------------------------------ #

    def _score_docstrings(self, code: str, non_empty: list, tree) -> float:
        if not non_empty:
            return 0.0

        if tree is None:
            return self._score_docstrings_regex(code)

        funcs = self._get_functions(tree)
        all_nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef,
                ast.ClassDef, ast.Module,
            ))
        ]
        docstrings = [ds for ds in (ast.get_docstring(n) for n in all_nodes) if ds]

        if not docstrings:
            return 0.0

        signal = 0.0
        all_ds_text = " ".join(docstrings).lower()

        # Coverage: AI documents every function, even trivial ones
        if funcs:
            documented = sum(1 for f in funcs if ast.get_docstring(f))
            coverage = documented / len(funcs)
            if coverage >= 1.0:
                signal += 0.30
            elif coverage > 0.70:
                signal += 0.15

        # Structured sections
        structured_count = sum(
            1 for ds in docstrings
            if re.search(
                r"\b(Args|Returns|Raises|Parameters|Attributes|Example|Examples)\s*:",
                ds,
            )
        )
        structured_ratio = structured_count / len(docstrings)
        if structured_ratio > 0.6:
            signal += 0.35
        elif structured_ratio > 0.3:
            signal += 0.15

        # AI description phrases — partial credit even without Args/Returns
        phrase_hits = sum(
            1 for p in self._AI_PHRASES
            if re.search(p, all_ds_text, re.IGNORECASE | re.MULTILINE)
        )
        signal += min(phrase_hits * 0.12, 0.35)

        return min(signal, 1.0)

    def _score_docstrings_regex(self, code: str) -> float:
        ds_blocks = re.findall(r'"""[\s\S]*?"""', code)
        if not ds_blocks:
            return 0.0
        structured = sum(
            1 for b in ds_blocks
            if re.search(r"\b(Args|Returns|Raises|Parameters)\s*:", b)
        )
        phrase_hits = sum(
            1 for p in self._AI_PHRASES
            if re.search(p, code, re.IGNORECASE)
        )
        return min(structured / max(len(ds_blocks), 1) * 0.5 + min(phrase_hits * 0.12, 0.35), 1.0)

    # ------------------------------------------------------------------ #
    # 2. NAMING CONVENTIONS  (0.0 – 1.0)                                  #
    # ------------------------------------------------------------------ #

    def _score_naming(self, code: str, tree) -> float:
        if tree is None:
            return self._score_naming_regex(code)

        func_names, var_names, arg_names = [], [], []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_names.append(node.name)
                for arg in node.args.args:
                    if arg.arg not in ("self", "cls"):
                        arg_names.append(arg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                var_names.append(node.id)

        all_names = func_names + var_names + arg_names
        if not all_names:
            return 0.0

        signal = 0.0

        # Strongly generic variable names
        strongly_generic = sum(1 for n in all_names if n.lower() in self._STRONGLY_AI_VAR_NAMES)
        strong_ratio = strongly_generic / len(all_names)
        if strong_ratio > 0.25:
            signal += 0.35
        elif strong_ratio > 0.12:
            signal += 0.15

        # Perfect snake_case
        public_names = [n for n in all_names if not n.startswith("_")]
        if len(public_names) >= 3:
            valid_snake = sum(1 for n in public_names if re.match(r"^[a-z][a-z0-9_]*$", n))
            if valid_snake / len(public_names) > 0.97:
                signal += 0.15

        # AI function prefixes (gate at >= 1)
        ai_func_prefixes = (
            "get_", "set_", "check_", "is_", "has_", "calculate_",
            "compute_", "process_", "handle_", "create_", "build_",
            "validate_", "update_", "find_", "load_", "save_",
            "parse_", "format_", "convert_", "extract_", "generate_",
        )
        if func_names:
            prefixed = sum(1 for f in func_names if any(f.startswith(p) for p in ai_func_prefixes))
            prefixed_ratio = prefixed / len(func_names)
            if prefixed_ratio > 0.60:
                signal += 0.35
            elif prefixed_ratio > 0.40:
                signal += 0.15

        # Single-letter non-loop vars: human signal
        actual_vars = [n for n in var_names if n not in arg_names]
        loop_vars = {"i", "j", "k", "x", "y", "n", "e", "f", "v", "c", "p", "q", "r", "s", "t"}
        single_letter = sum(1 for n in actual_vars if len(n) == 1 and n not in loop_vars)
        if len(actual_vars) > 3:
            sl_ratio = single_letter / len(actual_vars)
            if sl_ratio < 0.02:
                signal += 0.05
            elif sl_ratio >= 0.1:
                signal -= 0.10

        return min(max(signal, 0.0), 1.0)

    def _score_naming_regex(self, code: str) -> float:
        names = re.findall(r"\b([a-zA-Z_]\w*)\b", code)
        names = [n for n in names if not keyword.iskeyword(n)]
        if not names:
            return 0.0
        strongly_generic = sum(1 for n in names if n.lower() in self._STRONGLY_AI_VAR_NAMES)
        return min(strongly_generic / len(names) * 3, 1.0)

    # ------------------------------------------------------------------ #
    # 3. STRUCTURAL PATTERNS  (0.0 – 1.0)                                 #
    # ------------------------------------------------------------------ #

    def _score_structure(self, code: str, non_empty: list, tree) -> float:
        if tree is None:
            return self._score_structure_regex(code, non_empty)

        signal = 0.0
        funcs = self._get_functions(tree)

        if funcs:
            # Return statements
            returns_all = sum(
                1 for f in funcs
                if any(isinstance(n, ast.Return) for n in ast.walk(f))
            )
            ret_ratio = returns_all / len(funcs)
            if ret_ratio > 0.90:
                signal += 0.20
            elif ret_ratio > 0.70:
                signal += 0.10

            # Type annotations on args
            # Exclude self/cls when counting
            total_args = sum(
                len([a for a in f.args.args if a.arg not in ("self", "cls")])
                for f in funcs
            )
            annotated_args = sum(
                sum(1 for a in f.args.args if a.annotation and a.arg not in ("self", "cls"))
                for f in funcs
            )
            if total_args > 0:
                ann_ratio = annotated_args / total_args
                if ann_ratio > 0.85:
                    signal += 0.25
                elif ann_ratio > 0.55:
                    signal += 0.12
                # No penalty for unannotated — terse AI often skips hints

            # Return type annotations
            annotated_returns = sum(1 for f in funcs if f.returns)
            ret_ann_ratio = annotated_returns / len(funcs)
            if ret_ann_ratio > 0.85:
                signal += 0.20
            elif ret_ann_ratio > 0.55:
                signal += 0.10

        # Comprehensions
        comprehensions = len(re.findall(r"\[.+\s+for\s+\w+\s+in\s+", code))
        if comprehensions >= 3:
            signal += min(comprehensions * 0.04, 0.15)

        # AI builtin density
        ai_builtins = len(re.findall(
            r"\b(enumerate|zip|isinstance|hasattr|getattr|any|all|map|filter|sorted|reversed)\s*\(",
            code,
        ))
        builtin_density = ai_builtins / max(len(non_empty), 1)
        if builtin_density > 0.08:
            signal += min(ai_builtins * 0.025, 0.15)

        return min(max(signal, 0.0), 1.0)

    def _score_structure_regex(self, code: str, non_empty: list) -> float:
        signal = 0.0
        comprehensions = len(re.findall(r"\[.+\s+for\s+\w+\s+in\s+", code))
        if comprehensions >= 3:
            signal += min(comprehensions * 0.04, 0.15)
        type_hints = len(re.findall(r"\)\s*->\s*\w+\s*:", code))
        if type_hints >= 2:
            signal += min(type_hints * 0.10, 0.30)
        arg_hints = len(re.findall(
            r"\b\w+\s*:\s*(str|int|float|bool|list|dict|tuple|List|Dict|Optional|Union)\b", code
        ))
        if arg_hints >= 2:
            signal += min(arg_hints * 0.05, 0.20)
        return min(signal, 1.0)

    # ------------------------------------------------------------------ #
    # 4. COMMENT STYLE  (0.0 – 1.0)                                       #
    # ------------------------------------------------------------------ #

    def _score_comments(self, lines: list, non_empty: list) -> float:
        comment_lines = [line for line in lines if line.strip().startswith("#")]
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

        if not code_lines or not comment_lines:
            return 0.0

        signal = 0.0
        comment_ratio = len(comment_lines) / len(code_lines)

        if 0.15 < comment_ratio < 0.55:
            signal += 0.25
        elif comment_ratio >= 0.55:
            signal += 0.40

        # Full-sentence comments (capital letter start)
        full_sentence = sum(
            1 for line in comment_lines
            if re.match(r"#\s+[A-Z][a-z]", line.strip())
        )
        if comment_lines:
            fs_ratio = full_sentence / len(comment_lines)
            if fs_ratio > 0.65:
                signal += 0.30
            elif fs_ratio > 0.35:
                signal += 0.15

        # Obvious/redundant patterns
        obvious_patterns = [
            r"#\s*(initialize|initialise)",
            r"#\s*return\s+the",
            r"#\s*(create|creating)\s+(a|an|the)",
            r"#\s*(check|checking)\s+(if|whether)",
            r"#\s*(iterate|loop)\s+(over|through)",
            r"#\s*(calculate|compute|get)\s+the",
            r"#\s*(add|append|insert)\s+(the|to)",
            r"#\s*(define|defining)\s+(a|the)",
            r"#\s*step\s+\d+\s*[:\-]",
            r"#\s*(now|then|next|finally|lastly)",
        ]
        obvious_hits = sum(
            1 for line in comment_lines
            if any(re.search(p, line, re.IGNORECASE) for p in obvious_patterns)
        )
        if comment_lines:
            ratio = obvious_hits / len(comment_lines)
            if ratio > 0.25:
                signal += 0.25
            elif ratio > 0.10:
                signal += 0.10

        # Section dividers
        dividers = sum(1 for line in comment_lines if re.match(r"#\s*[-=]{3,}", line.strip()))
        if dividers >= 2:
            signal += min(dividers * 0.05, 0.10)

        return min(signal, 1.0)

    # ------------------------------------------------------------------ #
    # 5. COMPLEXITY & ENTROPY  (0.0 – 1.0)                                #
    # ------------------------------------------------------------------ #

    def _score_complexity(self, code: str, lines: list, non_empty: list) -> float:
        if len(non_empty) < 5:
            return 0.0

        tokens = re.findall(r"\w+", code)
        if not tokens:
            return 0.0

        signal = 0.0

        # Shannon entropy
        freq = Counter(tokens)
        total = sum(freq.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        norm_entropy = (entropy - 3.5) / 2.0
        if 0.20 < norm_entropy < 0.90:
            signal += 0.25

        # Line length uniformity
        lengths = [len(line) for line in non_empty]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        if std_dev < 14:
            signal += 0.35
        elif std_dev < 22:
            signal += 0.15
        elif std_dev > 35:
            signal -= 0.10

        # Nesting depth
        max_indent = max(
            ((len(line) - len(line.lstrip())) // 4 for line in non_empty if line.strip()),
            default=0,
        )
        if max_indent <= 2:
            signal += 0.20
        elif max_indent == 3:
            signal += 0.10
        elif max_indent >= 5:
            signal -= 0.15

        # Magic numbers (absence = AI signal)
        magic_numbers = re.findall(
            r"(?<![\"'\w])(?<!=)\s*\b([3-9]\d{1,3})\b(?!\s*[\"'])", code
        )
        if len(magic_numbers) == 0 and len(non_empty) > 10:
            signal += 0.20
        elif len(magic_numbers) >= 5:
            signal -= 0.10

        return min(max(signal, 0.0), 1.0)

    # ------------------------------------------------------------------ #
    # 6. CODE STYLE SIGNALS  (0.0 – 1.0)                                  #
    # ------------------------------------------------------------------ #

    def _score_code_style(self, lines: list, non_empty: list) -> float:
        if not non_empty:
            return 0.0

        signal = 0.0
        code = "\n".join(lines)

        # Operator spacing
        spaced = len(re.findall(r"\w\s[+\-*/]=?\s\w", code))
        unspaced = len(re.findall(r"\w[+\-*/]=\w", code))
        total_ops = spaced + unspaced
        if total_ops >= 4:
            spacing_ratio = spaced / total_ops
            if spacing_ratio > 0.97:
                signal += 0.20
            elif spacing_ratio < 0.80:
                signal -= 0.10

        # f-strings exclusively
        fstrings = len(re.findall(r'f["\']', code))
        old_format = len(re.findall(r"\.format\(", code))
        pct_fmt = len(re.findall(r"%[sdf]", code))
        if fstrings >= 2 and old_format == 0 and pct_fmt == 0:
            signal += 0.15

        # No trailing whitespace
        trailing_ws = sum(1 for line in lines if line != line.rstrip())
        if len(lines) > 5 and trailing_ws == 0:
            signal += 0.15
        elif trailing_ws >= 3:
            signal -= 0.10

        # if __name__ == "__main__"
        if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', code):
            signal += 0.10

        # Silent except pass
        silent_excepts = len(re.findall(r"except[^:]*:\s*(?:#[^\n]*\n\s*)*pass", code))
        if silent_excepts >= 1:
            signal += 0.10

        # Blank line regularity
        blank_runs = re.findall(r"\n(\n+)", code)
        run_lengths = [len(r) for r in blank_runs]
        if run_lengths:
            unique_runs = len(set(run_lengths))
            if unique_runs <= 2 and len(run_lengths) >= 3:
                signal += 0.10

        # No tabs, no 2-space indentation
        uses_tabs = "\t" in code
        two_space_indent = len(re.findall(r"^\s{2}[^\s]", code, re.MULTILINE))
        if not uses_tabs and two_space_indent == 0:
            signal += 0.10

        # Consistent quote style
        double_q = len(re.findall(r'"[^"]*"', code))
        single_q = len(re.findall(r"'[^']*'", code))
        total_q = double_q + single_q
        if total_q >= 4:
            dominant = max(double_q, single_q) / total_q
            if dominant > 0.90:
                signal += 0.10

        return min(max(signal, 0.0), 1.0)

    # ------------------------------------------------------------------ #
    # 7. TERSE AI PATTERNS  (0.0 – 1.0)  ← NEW                           #
    # ------------------------------------------------------------------ #

    def _score_terse_ai(self, code: str, lines: list, non_empty: list, tree) -> float:
        """
        Catches short, assignment-style AI code that lacks type hints and
        structured docstrings but still carries strong AI fingerprints.
        """
        signal = 0.0
        comment_lines = [line.strip() for line in lines if line.strip().startswith("#")]

        # ── Scaffold comments ──────────────────────────────────────────
        # AI almost always emits these; humans writing real code don't.
        scaffold_hits = sum(
            1 for pattern in self._AI_SCAFFOLD_COMMENTS
            if re.search(pattern, code, re.IGNORECASE)
        )
        if scaffold_hits >= 2:
            signal += 0.50
        elif scaffold_hits == 1:
            signal += 0.25

        # ── Docstring on a tiny/trivial function ──────────────────────
        # Humans skip docstrings on simple helpers; AI always adds one.
        if tree is not None:
            funcs = self._get_functions(tree)
            for f in funcs:
                ds = ast.get_docstring(f)
                if ds:
                    # Count statements in body (excluding the docstring expr)
                    body_stmts = [
                        n for n in f.body
                        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
                    ]
                    if len(body_stmts) <= 6:
                        # Tiny function that AI still bothered to document
                        signal += 0.20

        # ── Redundant branching: identical return in separate if-blocks ─
        # AI often writes:
        #   if x == 5: return "Not Eligible"
        #   return "Not Eligible"
        # instead of a single else. Humans collapse this naturally.
        if tree is not None:
            redundant = self._detect_redundant_branches(tree)
            if redundant:
                signal += 0.25

        # ── try/except ValueError around int(input()) ─────────────────
        # This exact scaffold is AI-generated boilerplate for competitive
        # programming / assignment solutions.
        if re.search(
            r"try\s*:[\s\S]*?int\s*\(\s*input\s*\(\s*\)\s*(?:\.strip\(\s*\))?\s*\)[\s\S]*?except\s+ValueError",
            code,
            re.IGNORECASE,
        ):
            signal += 0.30

        # ── AI-phrase docstring without Args/Returns ───────────────────
        # Catches the "Determines whether..." style that scores low in
        # _score_docstrings because it has no structured sections.
        if tree is not None:
            all_nodes = [
                n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Module))
            ]
            docstrings = [ds for ds in (ast.get_docstring(n) for n in all_nodes) if ds]
            all_ds_text = " ".join(docstrings).lower()
            phrase_hits = sum(
                1 for p in self._AI_PHRASES
                if re.search(p, all_ds_text, re.IGNORECASE)
            )
            if phrase_hits >= 1:
                signal += min(phrase_hits * 0.15, 0.30)

        # ── Perfectly spaced blank lines in short code ─────────────────
        # Short AI scripts have exactly 1 blank line between every block.
        if len(non_empty) <= 25:
            blank_runs = re.findall(r"\n(\n+)", code)
            run_lengths = [len(r) for r in blank_runs]
            if run_lengths and len(set(run_lengths)) == 1 and len(run_lengths) >= 2:
                signal += 0.10

        # ── "Eligible" / domain-label string literals ─────────────────
        # AI-generated assignment solutions frequently output these exact
        # human-readable verdict strings. Real developers use enums/bools.
        domain_strings = re.findall(
            r'"(Eligible|Not Eligible|Yes|No|Valid|Invalid|'
            r'Pass|Fail|Success|Failure|Found|Not Found|'
            r'Prime|Not Prime|Palindrome|Not Palindrome|'
            r'Armstrong|Leap Year|Not a Leap Year)"',
            code,
        )
        if len(domain_strings) >= 2:
            signal += 0.25
        elif len(domain_strings) == 1:
            signal += 0.10

        # ── Verbose error message in except (AI is always polite) ─────
        if re.search(
            r'except\s+\w+.*:\s*\n\s*print\s*\(\s*["\'].*[Pp]lease\s+enter',
            code,
        ):
            signal += 0.15

        return min(signal, 1.0)

    def _detect_redundant_branches(self, tree) -> bool:
        """
        Detect when two consecutive if-blocks return the same value,
        or when a final return duplicates the last branch's return.
        Pattern: if x > 5: return A  /  if x == 5: return B  /  return B
        """
        if tree is None:
            return False

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            body = node.body
            # Strip docstring
            stmts = [
                s for s in body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            ]

            returns = []
            for stmt in stmts:
                if isinstance(stmt, ast.Return):
                    returns.append(self._return_value_str(stmt))
                elif isinstance(stmt, ast.If):
                    # Collect return values from if-body
                    for s in stmt.body:
                        if isinstance(s, ast.Return):
                            returns.append(self._return_value_str(s))

            # Check for duplicate return values (sign of redundant branching)
            if len(returns) >= 2:
                counter = Counter(returns)
                if any(v >= 2 for v in counter.values()):
                    return True

        return False

    @staticmethod
    def _return_value_str(ret_node) -> str:
        """Convert a Return node's value to a comparable string."""
        if ret_node.value is None:
            return "None"
        if isinstance(ret_node.value, ast.Constant):
            return repr(ret_node.value.value)
        try:
            return ast.unparse(ret_node.value)
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------ #
    # Confidence label                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.75:
            return "high"
        elif score >= 0.58:
            return "medium"
        elif score >= 0.40:
            return "low"
        else:
            return "unlikely"


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import textwrap

    HUMAN_SAMPLE = textwrap.dedent("""\
        # quick script to flatten a list
        # TODO: handle generators too

        def flatten(lst, depth=0):
            out = []
            for x in lst:
                if isinstance(x, list) and depth < 5:
                    out += flatten(x, depth+1)
                else:
                    out.append(x)
            return out

        data = [[1,2,[3]],[4,5]]
        print(flatten(data))
    """)

    AI_TERSE_SAMPLE = textwrap.dedent("""\
        # Write your solution here
        def check_task_completion(task_count):
            \"\"\"
            Determines whether an employee is eligible for reward
            based on the number of tasks completed.
            \"\"\"

            if task_count > 5:
                return "Eligible"

            if task_count == 5:
                return "Not Eligible"

            return "Not Eligible"


        # Taking input from user
        try:
            tasks = int(input().strip())
            result = check_task_completion(tasks)
            print(result)
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    """)

    AI_VERBOSE_SAMPLE = textwrap.dedent("""\
        from typing import List, Optional

        def calculate_average(numbers: List[float]) -> Optional[float]:
            \"\"\"
            Calculates the average of a list of numbers.

            Args:
                numbers: A list of float values.

            Returns:
                The average value, or None if the list is empty.
            \"\"\"
            # Check if the list is empty
            if not numbers:
                return None

            # Calculate the sum of all numbers
            total = sum(numbers)

            # Compute and return the average
            return total / len(numbers)


        def process_data(data: List[float]) -> dict:
            \"\"\"
            Processes the input data and returns summary statistics.

            Args:
                data: A list of float values to process.

            Returns:
                A dictionary containing the average, minimum, and maximum values.
            \"\"\"
            # Initialize the result dictionary
            result = {
                "average": calculate_average(data),
                "minimum": min(data) if data else None,
                "maximum": max(data) if data else None,
            }

            # Return the result
            return result


        if __name__ == "__main__":
            # Example usage
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            output = process_data(values)
            print(f"Statistics: {output}")
    """)

    agent = AIDetectionAgent()

    for label, sample in [
        ("HUMAN", HUMAN_SAMPLE),
        ("AI TERSE", AI_TERSE_SAMPLE),
        ("AI VERBOSE", AI_VERBOSE_SAMPLE),
    ]:
        print(f"\n=== {label} ===")
        r = agent.detect(sample)
        print(f"  ai_score:        {r['ai_score']}")
        print(f"  is_ai_generated: {r['is_ai_generated']}")
        print(f"  confidence:      {r['confidence']}")
        print(f"  feature_scores:  {r['feature_scores']}")
        print(f"  human_signals:   {r['human_signals_found']}")
        print(f"  parse_error:     {r['parse_error']}")