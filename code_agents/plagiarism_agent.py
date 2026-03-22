from difflib import SequenceMatcher
from code_agents.code_utils import (
    get_python_ast,
    normalize_python_ast,
    get_java_ast,
    get_cpp_ast
)


class PlagiarismAgent:
    def __init__(self):
        self.past_representations = []

    def check_plagiarism(self, code: str, language: str):
        representation = self._get_representation(code, language)

        if representation is None:
            return {
                "plagiarism_score": 0,
                "is_plagiarized": False,
                "error": "Parsing failed"
            }

        max_similarity = 0

        for past in self.past_representations:
            similarity = SequenceMatcher(None, representation, past).ratio()
            max_similarity = max(max_similarity, similarity)

        result = {
            "plagiarism_score": round(max_similarity, 3),
            "is_plagiarized": max_similarity > 0.8
        }

        self.past_representations.append(representation)

        return result

    def _get_representation(self, code: str, language: str):
        language = language.lower()

        if language == "python":
            ast_tree = get_python_ast(code)
            normalized = normalize_python_ast(ast_tree)
            return str(normalized)

        elif language == "java":
            return get_java_ast(code)

        elif language in ["cpp", "c++"]:
            return get_cpp_ast(code)

        else:
            # fallback → raw text
            return code