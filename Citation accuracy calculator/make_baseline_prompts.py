"""
make_baseline_prompts.py
Builds INDEX-AIDED prompts for the baseline chat models (ChatGPT / Claude / DeepSeek ...).

Each prompt gives the model the same kind of evidence AURA uses:
  A. README + dependency file          (background)
  B. file tree
  C. SYMBOL INDEX from your parser     (every symbol as a ready-made, validator-compatible tag)
  D. source code WITH line numbers
and asks for the SAME four views AURA writes (plain English / student / recruiter / developer).

Output: prompts_indexed/<repo_key>.txt   -> paste each into a NEW chat, save the answer as
        Model_outputs/<system>_idx/<repo_key>.txt   (e.g. chatgpt_idx, claude_idx, deepseek_idx)

Usage (PowerShell):
    python make_baseline_prompts.py
    $env:AURA_REPOS="01_...,04_..."; python make_baseline_prompts.py     # subset
    $env:PROMPT_MAX_CHARS="120000"                                       # shrink for small chat windows
"""

import os
import json

ROOT_INDEX_PATH = "metadata_index.json"
REPOSITORIES_DIR = "repositories"
OUT_DIR = "prompts_indexed"
MAX_CHARS = int(os.environ.get("PROMPT_MAX_CHARS", "150000"))

DEFAULT_SELECTED = [
    "01_Attendance-Management-System",
    "04_ResumeRise",
    "05_bookreview-api",
    "06_Virtual-Tour-Creater",
    "08_Ai_Health_Recommedation",
    "10_HomeServicesBooking",
    "15_whatsappchatanalyzer",
    "19_AI-COACHING-PLATFORM",
]
_env = os.environ.get("AURA_REPOS", "")
SELECTED = [k.strip() for k in _env.split(",") if k.strip()] or DEFAULT_SELECTED

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "poetry.lock"}


def file_tree(repo_root, limit=150):
    lines = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in sorted(filenames):
            if fname.startswith(".") or fname in SKIP_FILES:
                continue
            lines.append(os.path.relpath(os.path.join(dirpath, fname), repo_root).replace("\\", "/"))
    lines.sort()
    if len(lines) > limit:
        lines = lines[:limit] + [f"... ({len(lines) - limit} more files)"]
    return "\n".join(lines)


def background(repo_root):
    parts = []
    for name in sorted(os.listdir(repo_root)):
        if name.lower().startswith("readme"):
            with open(os.path.join(repo_root, name), "r", encoding="utf-8", errors="ignore") as f:
                parts.append("README:\n" + f.read()[:1200])
            break
    for name in ("requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml"):
        p = os.path.join(repo_root, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                parts.append(f"{name}:\n" + f.read()[:600])
            break
    return "\n\n".join(parts) or "(none)"


def numbered_source(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    return "\n".join(f"{i:>4} | {ln}" for i, ln in enumerate(lines, 1))


def tag(file_path, symbol, e):
    return f"[cite:file={file_path};symbol={symbol};lines={e['line_start']}-{e['line_end']}]"


TEMPLATE = """You are analyzing a software repository. You are given:
A. background (README and dependency file), B. the file tree, C. a SYMBOL INDEX produced by a code
parser, D. the source code with line numbers (shown as "N | code").

=== A. BACKGROUND ===
{background}

=== B. FILE TREE ===
{tree}

=== C. SYMBOL INDEX (each line is a ready-made citation tag) ===
{symbols}

=== D. SOURCE CODE WITH LINE NUMBERS ===
{code}
=== END OF REPOSITORY ===

TASK: Write ONE report with exactly these four parts, about 700-900 words in total (tags not counted):

## 1. In plain English
For someone who has never programmed: 2-3 sentences on what the project is and who would use it,
one line starting "Analogy:", and "What you can do with it:" with 3 short bullets. No jargon.

## 2. For students: how to read this project
A numbered reading order of 4-5 files with one line each on why to read it; a Mermaid diagram in a
```mermaid code fence (flowchart LR, at most 7 nodes, plain-word labels); "Key terms:" 3 short glossary entries.

## 3. For recruiters and interviewers
"What it demonstrates:" 3 bullets of concrete skills; "Stack:" one line; "Most interesting part:" 1-2 sentences.

## 4. For developers
Short bullets grouped under: Purpose, Main modules, How data flows, Tech stack, How to run it,
Worth a closer look. Explain what the code does and how or why.

CITATION RULES (important):
- To cite, COPY a tag from section C exactly, character for character, and put it right after the claim
  it supports. Never type a tag from memory and never edit its file, symbol or line numbers.
- Every bullet in part 4 that describes code must carry at least one tag. Parts 1-3 may carry tags where natural.
- Cite only when that code really supports the claim. If no tag in section C fits, either leave the
  claim out or base it on the README and give no tag.
- Do not invent files, functions, or behaviour that are not in the material above.

Write the report now.
"""


def build(repo_key, entry):
    repo_root = os.path.join(REPOSITORIES_DIR, entry["slug"])
    with open(entry["index_path"], "r", encoding="utf-8") as f:
        index = json.load(f)

    by_file = {}
    for key, e in index.items():
        if "::" not in key:
            continue
        fp, sym = key.split("::", 1)
        if os.path.exists(os.path.join(repo_root, fp)):
            by_file.setdefault(fp, []).append((sym, e))

    # include source files smallest-first until the character budget is used
    sizes = {fp: os.path.getsize(os.path.join(repo_root, fp)) for fp in by_file}
    budget = MAX_CHARS - 12000          # room for background, tree, symbol list and instructions
    included, omitted, used = [], [], 0
    for fp in sorted(by_file, key=lambda x: sizes[x]):
        est = int(sizes[fp] * 1.15)
        if used + est <= budget:
            included.append(fp)
            used += est
        else:
            omitted.append(fp)
    included.sort()

    symbols = []
    for fp in included:                  # only symbols whose code is actually shown
        for sym, e in sorted(by_file[fp], key=lambda x: x[1]["line_start"]):
            symbols.append(tag(fp, sym, e))

    code = "\n\n".join(f"--- {fp} ---\n{numbered_source(os.path.join(repo_root, fp))}" for fp in included)
    prompt = TEMPLATE.format(background=background(repo_root), tree=file_tree(repo_root),
                             symbols="\n".join(symbols), code=code)
    return prompt, len(included), len(omitted), len(symbols)


def main():
    if not os.path.exists(ROOT_INDEX_PATH):
        print(f"ERROR: {ROOT_INDEX_PATH} not found.")
        return
    with open(ROOT_INDEX_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    for key in SELECTED:
        entry = manifest.get(key)
        if not entry or entry.get("skipped"):
            print(f"{key}: not in {ROOT_INDEX_PATH}")
            continue
        if not os.path.isdir(os.path.join(REPOSITORIES_DIR, entry["slug"])):
            print(f"{key}: repo folder missing")
            continue
        prompt, n_in, n_out, n_sym = build(key, entry)
        with open(os.path.join(OUT_DIR, f"{key}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        note = f"  WARNING: {n_out} files omitted for length" if n_out else ""
        print(f"{key}: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens), "
              f"{n_in} files, {n_sym} symbols{note}")

    print(f"\nPrompts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()