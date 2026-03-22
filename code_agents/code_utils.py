import ast
import subprocess
import tempfile
import os
import re

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
        import subprocess
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".java") as f:
            f.write(code.encode())
            file_path = f.name

        result = subprocess.run(
            [
                "java",
                "-cp",
                "tools/javaparser-core-3.25.8.jar;tools",
                "ASTPrinter",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        os.unlink(file_path)

        return result.stdout

    except Exception as e:
        print("Java AST Error:", e)
        return None
    
def normalize_java_ast(ast_text: str):
    if not ast_text:
        return None

    lines = ast_text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)

    return "\n".join(cleaned)

#C++ AST (Clang) with filtering and normalization
def get_cpp_ast(code: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cpp") as f:
            f.write(code.encode())
            file_path = f.name

        result = subprocess.run(
            ["clang++", "-Xclang", "-ast-dump", "-fsyntax-only", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # 🔥 suppress errors
            text=True
        )

        os.unlink(file_path)

        raw_ast = result.stdout

        # 🔥 Filter important lines
        filtered_lines = []
        for line in raw_ast.split("\n"):
            if any(keyword in line for keyword in [
                "FunctionDecl",
                "IfStmt",
                "ForStmt",
                "WhileStmt",
                "ReturnStmt",
                "BinaryOperator",
                "CallExpr",
                "DeclStmt",
                "VarDecl"
            ]):
                filtered_lines.append(line.strip())

        filtered_ast = "\n".join(filtered_lines)

        # 🔥 Normalize (NEW STEP)
        return normalize_cpp_ast(filtered_ast)

    except Exception as e:
        print("C++ AST Error:", e)
        return None    
def normalize_cpp_ast(ast_text: str):
    if not ast_text:
        return None

    lines = ast_text.split("\n")
    cleaned = []

    for line in lines:
        # Remove memory addresses (0x...)
        line = re.sub(r'0x[0-9a-fA-F]+', '', line)

        # Remove source location info <...>
        line = re.sub(r'<.*?>', '', line)

        # Remove extra spaces
        line = re.sub(r'\s+', ' ', line).strip()

        cleaned.append(line)

    return "\n".join(cleaned)