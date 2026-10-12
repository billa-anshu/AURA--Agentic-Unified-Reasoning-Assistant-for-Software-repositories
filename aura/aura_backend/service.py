"""service.py - runs the existing aura.py pipeline on an arbitrary directory.
No CrewAI: aura.py already is the agent loop. We only (1) build the symbol index it needs,
(2) call aura.summarize_repo, (3) report stage progress and repo stats.
Groq rate limits are per key, so runs are serialised with a lock.
"""
import builtins
import json
import os
import threading
from pathlib import Path

import aura                       # aura.py must sit next to this file
from indexer import build_index

_run_lock = threading.Lock()

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
             ".idea", ".vscode", ".next", "target", ".gradle"}
LANGS = {".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
         ".tsx": "TypeScript", ".java": "Java", ".html": "HTML", ".css": "CSS", ".md": "Markdown",
         ".json": "JSON", ".go": "Go", ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".cpp": "C++",
         ".c": "C", ".h": "C/C++", ".rs": "Rust", ".kt": "Kotlin", ".swift": "Swift",
         ".vue": "Vue", ".ipynb": "Jupyter", ".sql": "SQL", ".yml": "YAML", ".yaml": "YAML"}
_STAGES = (("[1]", 1), ("[2]", 2), ("[3-4]", 3), ("[5]", 4))


class NoCodeError(Exception):
    pass


def repo_stats(root: Path):
    files = lines = 0
    langs = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.startswith(".") or fn in {"package-lock.json", "yarn.lock", "poetry.lock"}:
                continue
            files += 1
            ext = os.path.splitext(fn)[1].lower()
            langs[LANGS.get(ext, "Other")] = langs.get(LANGS.get(ext, "Other"), 0) + 1
            if ext in LANGS:
                p = os.path.join(dp, fn)
                try:
                    if os.path.getsize(p) < 1_000_000:
                        with open(p, "rb") as f:
                            lines += f.read().count(b"\n") + 1
                except OSError:
                    pass
    return {"files": files, "lines": lines, "languages": langs}


def _stage_print(cb):
    """aura.py reports its stages with print('    [1] planner...'). Hook that for progress."""
    def _p(*a, **k):
        builtins.print(*a, **k)
        s = " ".join(str(x) for x in a).strip()
        for marker, n in _STAGES:
            if s.startswith(marker):
                cb(n)
                break
    return _p


def run_aura(job_id: str, repo_root: Path, workdir: Path, display_name: str, on_stage=lambda n: None):
    index = build_index(str(repo_root))
    if not index:
        raise NoCodeError("no supported source files found (py, js, ts, java, go, rb, php, cs, ...)")
    index_path = workdir / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    extra = repo_stats(repo_root)

    with _run_lock:
        aura._cache = {}                 # in-memory only; no shared cache file between users
        aura.save_cache = lambda: None
        aura.print = _stage_print(on_stage)
        try:
            retriever = aura.RepoRetriever(job_id, str(index_path), str(repo_root))
            if not retriever.chunks or retriever.vectorizer is None:
                raise NoCodeError("source files were found but nothing could be indexed")
            last = None
            for _ in range(aura.MAX_REPO_ATTEMPTS):
                try:
                    report, facts = aura.summarize_repo(job_id, str(repo_root), retriever)
                    break
                except aura.DailyQuotaExhausted:
                    raise
                except Exception as e:   # stages stay in aura._cache, so a retry resumes
                    last = e
            else:
                raise RuntimeError(f"pipeline failed: {last}")
        finally:
            if hasattr(aura, "print"):
                del aura.print

    report = report.replace(f"# {job_id.split('_', 1)[-1]} - AURA report", f"# {display_name} - AURA report", 1)
    return {
        "name": display_name,
        "report_markdown": report,
        "claims": [{"n": n, "topic": c["t"], "claim": c["c"], "citations": c["tags"], "files": c["labels"]}
                   for n, c in enumerate(facts["claims"], 1)],
        "stats": {
            **extra,
            "symbols_indexed": len(retriever.chunks),
            "verified_claims": len(facts["claims"]),
            "citations": report.count(aura.CITE_TAG_PREFIX),
            "first_pass_faithfulness": facts["first_pass_faithfulness"],
            "llm_calls": facts["llm_calls_this_run"],
        },
        "agent_trace": facts["agent_trace"],
        "verifier_rounds": facts["verifier_rounds"],
    }
