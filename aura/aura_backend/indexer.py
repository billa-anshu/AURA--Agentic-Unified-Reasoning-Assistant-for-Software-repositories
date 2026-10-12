"""indexer.py - builds the symbol index AURA's RepoRetriever expects:
    {"relative/path.py::Class.method": {"line_start": 10, "line_end": 42}, ...}
Python uses `ast` (exact ranges); other languages use regex + brace matching.
"""
import ast
import os
import re

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
             ".idea", ".vscode", ".next", "target", ".gradle"}
CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".php", ".cs",
             ".kt", ".rs", ".c", ".cpp", ".h", ".hpp", ".swift", ".vue"}
MAX_FILE_BYTES = 400_000
MAX_SYMBOL_LINES = 150
MODULE_CHUNK_LINES = 120

_GENERIC = [
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:class|interface|enum|struct|trait|impl)\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("),
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:public|private|protected|internal|static|final|abstract|override|async|\s)+"
               r"[\w<>\[\],.?\s]+?\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:throws [\w, .]+)?\s*\{?\s*$"),
    re.compile(r"^\s*def\s+([A-Za-z_]\w*[?!]?)"),                 # ruby
    re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+)*function\s+([A-Za-z_]\w*)"),  # php
]
_NOT_SYMBOLS = {"if", "for", "while", "switch", "catch", "return", "else", "new", "elif", "with"}


def _py_symbols(src):
    tree = ast.parse(src)
    out = []

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{ch.name}"
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append((name, start, ch.end_lineno or ch.lineno))
                if isinstance(ch, ast.ClassDef):
                    visit(ch, name + ".")
    visit(tree, "")
    return out


def _generic_symbols(lines):
    starts = []
    for i, line in enumerate(lines):
        if len(line) > 300:
            continue
        for rx in _GENERIC:
            m = rx.match(line)
            if m and m.group(1) not in _NOT_SYMBOLS:
                starts.append((m.group(1), i + 1))
                break
    out = []
    for k, (name, s) in enumerate(starts):
        end = None
        depth, seen = 0, False
        for j in range(s - 1, min(len(lines), s - 1 + MAX_SYMBOL_LINES)):
            for ch in lines[j]:
                if ch == "{":
                    depth, seen = depth + 1, True
                elif ch == "}":
                    depth -= 1
            if seen and depth <= 0:
                end = j + 1
                break
            if not seen and j - (s - 1) >= 3:     # no brace nearby: not brace-delimited
                break
        if end is None:
            nxt = starts[k + 1][1] - 1 if k + 1 < len(starts) else len(lines)
            end = min(nxt, s + MAX_SYMBOL_LINES - 1)
        out.append((name, s, max(end, s)))
    return out


def build_index(repo_root):
    index = {}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            full = os.path.join(dirpath, fn)
            if ext not in CODE_EXTS or os.path.islink(full):
                continue
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    src = f.read()
            except OSError:
                continue
            lines = src.splitlines(keepends=True)
            if not lines:
                continue
            rel = os.path.relpath(full, repo_root).replace("\\", "/")
            syms = None
            if ext == ".py":
                try:
                    syms = _py_symbols(src)
                except (SyntaxError, ValueError, RecursionError):
                    syms = None
            if syms is None:
                syms = _generic_symbols(lines)
            if not syms:   # script-style file: index the top of it as one chunk
                syms = [("<module>", 1, min(len(lines), MODULE_CHUNK_LINES))]
            for name, s, e in syms:
                key = f"{rel}::{name}"
                if key in index:
                    key = f"{key}#{s}"
                index[key] = {"line_start": s, "line_end": e}
    return index
