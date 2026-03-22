import ast
import subprocess
import tempfile
import os


# ---------------- PYTHON AST ----------------
def get_python_ast(code: str):
    try:
        return ast.parse(code)
    except:
        return None


def normalize_python_ast(node):
    if node is None:
        return None

    if isinstance(node, ast.AST):
        fields = []
        for field, value in ast.iter_fields(node):
            if field in ("id", "arg", "name"):
                continue
            fields.append(normalize_python_ast(value))
        return (type(node).__name__, tuple(fields))

    elif isinstance(node, list):
        return [normalize_python_ast(x) for x in node]

    else:
        return str(node)


# ---------------- JAVA AST (JavaParser) ----------------
def get_java_ast(code: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".java") as f:
            f.write(code.encode())
            file_path = f.name

        result = subprocess.run(
            ["java", "-jar", "tools/javaparser.jar", file_path],
            capture_output=True,
            text=True
        )

        os.unlink(file_path)

        return result.stdout

    except Exception as e:
        return None


# ---------------- C++ AST (Clang) ----------------
def get_cpp_ast(code: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cpp") as f:
            f.write(code.encode())
            file_path = f.name

        result = subprocess.run(
            ["clang++", "-Xclang", "-ast-dump", "-fsyntax-only", file_path],
            capture_output=True,
            text=True
        )

        os.unlink(file_path)

        return result.stdout

    except Exception:
        return None