"""
aura.py - AURA v5 (no-tool-loop, citation-grounded repo summaries)

What changed vs v4
  1. More evidence per repo (bigger code/arch groups, more chunks per file, anchors that
     spread across top-level directories) -> more material to cite, better completeness.
  2. The writer is split into TWO calls (sections 1-2, then 3-5) and sees short evidence
     snippets, not only the analysts' prose -> longer, deeper reports with more citations.
  3. SentenceCiter: a deterministic post-processing step that attaches the best-matching
     REAL chunk tag to code-describing sentences that still have no citation. Tags always
     come from the parsed index, so every tag still points at real evidence.
  4. Adjacent duplicate tags are collapsed.

Citation grounding (unchanged idea)
  * Every retrieved chunk gets a short ID (C1, C2 ...). The model only writes IDs.
  * Python swaps each ID for the real tag built from the metadata index:
        [cite:file=<path>;symbol=<name>;lines=<start>-<end>]
  * Unknown/invented IDs are dropped.

LLM calls per repo: 5 (code analyst, architecture analyst, writer parts A, B, C).
Output tokens per call are capped so they fit Groq's separate OUTPUT-tokens-per-minute
limit (set AURA_OTPM, default 1000; AURA_OTPM=0 disables the cap for models without it).
All per-stage progress lives in ONE file (aura_cache.json), not hundreds of small files.

Install:
  pip install openai scikit-learn python-dotenv

Usage:
  export GROQ_API_KEY=...
  python aura.py
"""

import os
import re
import json
import time
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import (
    OpenAI,
    RateLimitError,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Override without editing code:  export AURA_MODEL=<model id>
LLM_MODEL = os.environ.get("AURA_MODEL", "qwen/qwen3.8-27b")
REASONING_EFFORT = "none"            # dropped automatically if the model rejects it
REQUEST_TIMEOUT = 90                 # seconds per LLM call (writer calls are longer now)

MODEL_OUTPUTS_DIR = "Model_outputs"
AURA_OUTPUT_DIR = os.path.join(MODEL_OUTPUTS_DIR, "AURA")
CACHE_PATH = "aura_cache.json"       # ONE file holds all per-stage progress
LEGACY_PARTIAL_DIR = "aura_partial"  # old per-stage files; only read once to migrate "done" flags
REPOSITORIES_DIR = "repositories"
ROOT_INDEX_PATH = "metadata_index.json"

# Output budgets (tokens per call). Every call is also capped to fit OTPM_LIMIT (below).
CODE_MAX_TOKENS = 800
ARCH_MAX_TOKENS = 700
WRITER_MAX_TOKENS = 800   # per writer part (3 parts: sections 1-2, 3, 4-5)

# Groq enforces a separate OUTPUT tokens-per-minute limit on some models (yours: 1000/min).
# 0 = no output limit (e.g. a model/tier that doesn't have one).
OTPM_LIMIT = int(os.environ.get("AURA_OTPM", "1000"))
OUT_CAP_FRACTION = 0.7    # max_tokens per call = 70% of the per-minute limit

# Retrieval
VECTOR_MAX_FEATURES = 2000
PER_QUERY_K = 4           # chunks kept per query
CHUNK_CHARS = 500         # chars of code shown to the analysts per chunk
WRITER_SNIPPET_CHARS = 180  # chars of code shown to the writer per chunk
SNIPPET_CONTEXT_LINES = 2
CODE_GROUP_MAX = 18       # max chunks given to the Code Analyst
ARCH_GROUP_MAX = 14       # max chunks given to the Architecture Analyst
CODE_GROUP_MAX_LIMITED = 10   # used when OTPM_LIMIT is small (analysts must fit their output budget)
ARCH_GROUP_MAX_LIMITED = 8
MAX_PER_FILE = 3          # spread evidence across files (better coverage)
ANCHOR_SYMBOLS = 8        # biggest symbols (one per top-level dir first, then per file)
README_CHARS = 1200       # project context for the writer (not citable)
MANIFEST_CHARS = 500

# Sentence citer (deterministic extra citations)
SENT_CITER_THRESHOLD = 0.20   # min TF-IDF cosine between sentence and chunk. Tune on 2-3 repos.
SENT_CITER_MAX_USES = 3       # one chunk tag may be auto-attached at most this many times

# Bump this string to regenerate EVERY repo with fresh LLM calls. Old outputs and old cached
# stages are ignored automatically.
PROMPT_VERSION = "v5"

CODE_QUERIES = [
    "main entry point application startup",
    "core classes models entities",
    "api routes endpoints handlers controllers",
    "service business logic processing",
    "utility helper functions",
    "ui component page view",
    "form input validation",
    "data processing transformation",
]
ARCH_QUERIES = [
    "database access repository storage",
    "request routing flow middleware",
    "configuration settings environment",
    "authentication authorization security",
    "external api client integration",
    "model inference pipeline",
    "file upload download handling",
    "error handling logging",
]

# Pacing: Groq free tier is ~8k tokens/min per model; stay under it.
TPM_BUDGET = 6500
TPM_WINDOW_SECONDS = 60
MAX_LLM_RETRIES = 5
INTER_REPO_PAUSE = 5
MAX_REPO_ATTEMPTS = 2

CITE_TAG_PREFIX = "[cite:file="
TAG_RE = re.compile(r"\[cite:file=[^\]]*?;lines=\d+-\d+\]")


class DailyQuotaExhausted(Exception):
    """Groq's per-day token cap is gone; stop the whole run."""


# ---------------------------------------------------------------------------
# Pacing: track real token usage in a rolling window
# ---------------------------------------------------------------------------

class AdaptivePacer:
    # Tracks BOTH total tokens and output tokens in a rolling 60s window.
    def __init__(self, tpm=TPM_BUDGET, otpm=OTPM_LIMIT, window=TPM_WINDOW_SECONDS):
        self.tpm = tpm
        self.otpm = otpm
        self.window = window
        self.log = []  # (timestamp, total_tokens, output_tokens)

    def _prune(self):
        now = time.time()
        self.log = [e for e in self.log if now - e[0] < self.window]

    def wait_for(self, total_needed, out_needed=0):
        while True:
            self._prune()
            used_t = sum(e[1] for e in self.log)
            used_o = sum(e[2] for e in self.log)
            fits_t = used_t + total_needed <= self.tpm
            fits_o = (not self.otpm) or (used_o + out_needed <= self.otpm)
            if not self.log or (fits_t and fits_o):
                return
            wait = max(1.0, self.log[0][0] + self.window - time.time() + 0.5)
            print(f"    ...pacing: waiting {wait:.0f}s for token window")
            time.sleep(wait)

    def record(self, total, out=0):
        self.log.append((time.time(), total, out))


pacer = AdaptivePacer()
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)
_send_reasoning = True


# ---------------------------------------------------------------------------
# LLM call (no tools, no agent loop)
# ---------------------------------------------------------------------------

def _clean(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def _retry_after_seconds(msg):
    """Parse Groq's 'try again in 1m12.5s' / '6.52s' hint."""
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", msg)
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1


_out_cap = int(OTPM_LIMIT * OUT_CAP_FRACTION) if OTPM_LIMIT else 10 ** 6
_otpm_hits = 0


def _is_output_limit(low):
    return "output tokens" in low or "otpm" in low


def call_llm(system, user, max_tokens, model=LLM_MODEL):
    global _send_reasoning, _out_cap, _otpm_hits
    last_err = None

    for attempt in range(MAX_LLM_RETRIES):
        max_tokens = min(max_tokens, _out_cap)
        estimate = (len(system) + len(user)) // 3 + max_tokens
        pacer.wait_for(min(estimate, pacer.tpm), max_tokens)

        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        effort = "low" if "gpt-oss" in model else REASONING_EFFORT  # gpt-oss rejects "none"
        if _send_reasoning and effort:
            kwargs["extra_body"] = {"reasoning_effort": effort}

        try:
            resp = client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            total = usage.total_tokens if usage else estimate
            out = usage.completion_tokens if usage else max_tokens
            pacer.record(total, out)
            return _clean(resp.choices[0].message.content)

        except RateLimitError as e:
            last_err = e
            msg = str(e)
            low = msg.lower()
            if "per day" in low or "tpd" in low:
                raise DailyQuotaExhausted(msg)
            if _is_output_limit(low):
                # Output-token limit: trimming the PROMPT does not help; lower max_tokens instead.
                m = re.search(r"limit\s+(\d+)", low)
                limit = int(m.group(1)) if m else (OTPM_LIMIT or 1000)
                pacer.otpm = limit
                _otpm_hits += 1
                new_cap = int(limit * OUT_CAP_FRACTION)
                _out_cap = int(min(_out_cap, new_cap) * (0.85 if _otpm_hits > 1 else 1.0))
                print(f"    ...output-token limit ({limit}/min): capping max_tokens at {_out_cap}")
                time.sleep(3)
                continue
            if "too large" in low:
                user = user[: int(len(user) * 0.7)]
                print("    ...request too large, trimming prompt")
                continue
            wait = _retry_after_seconds(msg) or 20 * (attempt + 1)
            print(f"    ...rate limited, waiting {wait:.0f}s")
            time.sleep(wait)

        except APIStatusError as e:
            last_err = e
            msg = str(e)
            low = msg.lower()
            if e.status_code == 413 or "too large" in low:
                user = user[: int(len(user) * 0.7)]
                print("    ...request too large, trimming prompt")
                continue
            if e.status_code == 400 and "reasoning" in low and _send_reasoning:
                _send_reasoning = False
                print("    ...model rejected reasoning_effort, retrying without it")
                continue
            if e.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            raise

        except (APITimeoutError, APIConnectionError) as e:
            last_err = e
            print(f"    ...network/timeout error, retrying ({attempt + 1}/{MAX_LLM_RETRIES})")
            time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"LLM call failed after {MAX_LLM_RETRIES} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Retriever (TF-IDF over indexed symbols, citation tags built from metadata)
# ---------------------------------------------------------------------------

def _norm(path):
    return path.replace("\\", "/")


class RepoRetriever:
    def __init__(self, repo_key, index_path, repo_root):
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        self.repo_key = repo_key
        self.repo_root = repo_root
        self.chunks = []
        self.chunk_meta = []
        self._cache = {}
        self._file_cache = {}

        for key, entry in self.index.items():
            try:
                file_path, symbol = key.split("::", 1)
                full_path = os.path.join(repo_root, file_path)
                if not os.path.exists(full_path):
                    continue

                if full_path not in self._file_cache:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        self._file_cache[full_path] = f.readlines()
                lines = self._file_cache[full_path]

                start = max(0, entry["line_start"] - SNIPPET_CONTEXT_LINES)
                end = min(len(lines), entry["line_end"] + SNIPPET_CONTEXT_LINES)
                snippet = "".join(lines[start:end])
                if not snippet.strip():
                    continue

                self.chunks.append(f"{symbol}\n{snippet}")
                self.chunk_meta.append({
                    "file": file_path,
                    "symbol": symbol,
                    "line_start": entry["line_start"],
                    "line_end": entry["line_end"],
                })
            except Exception:
                continue

        self._file_cache = {}  # free memory
        self.vectorizer = None
        self.matrix = None

        if self.chunks:
            # Fall back to looser settings for tiny repos (strict min_df can empty the vocabulary).
            for params in (
                dict(min_df=2, max_df=0.7),
                dict(min_df=1, max_df=0.95),
                dict(min_df=1, max_df=1.0),
            ):
                try:
                    vec = TfidfVectorizer(
                        stop_words="english",
                        max_features=VECTOR_MAX_FEATURES,
                        ngram_range=(1, 2),
                        **params,
                    )
                    self.matrix = vec.fit_transform(self.chunks)
                    self.vectorizer = vec
                    break
                except ValueError:
                    continue

    def search(self, query, top_k=PER_QUERY_K):
        key = (query, top_k)
        if key in self._cache:
            return self._cache[key]
        if self.vectorizer is None:
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        results = []
        for i in sims.argsort()[::-1][:top_k]:
            if sims[i] <= 0.03:
                continue
            results.append(self._result(i, float(sims[i])))
        self._cache[key] = results
        return results

    def _result(self, i, score):
        meta = self.chunk_meta[i]
        tag = (
            f"[cite:file={meta['file']};symbol={meta['symbol']};"
            f"lines={meta['line_start']}-{meta['line_end']}]"
        )
        return {
            "text": self.chunks[i][:CHUNK_CHARS],
            "citation_tag": tag,
            "meta": meta,
            "score": score,
        }

    def top_symbols(self, n=ANCHOR_SYMBOLS):
        """Anchor evidence picked directly (no TF-IDF): the largest symbol in each top-level
        directory first (breadth across the project), then the largest remaining per file."""
        order = sorted(
            range(len(self.chunk_meta)),
            key=lambda i: self.chunk_meta[i]["line_end"] - self.chunk_meta[i]["line_start"],
            reverse=True,
        )
        out, files, dirs = [], set(), set()

        for i in order:                                   # pass 1: one per top-level dir
            f = _norm(self.chunk_meta[i]["file"])
            top = f.split("/")[0] if "/" in f else "."
            if top in dirs:
                continue
            dirs.add(top)
            files.add(f)
            out.append(self._result(i, 1.0))
            if len(out) >= n:
                return out

        for i in order:                                   # pass 2: one per remaining file
            f = _norm(self.chunk_meta[i]["file"])
            if f in files:
                continue
            files.add(f)
            out.append(self._result(i, 1.0))
            if len(out) >= n:
                break
        return out


# ---------------------------------------------------------------------------
# Evidence pool: short IDs (C1, C2...) <-> real citation tags
# ---------------------------------------------------------------------------

class EvidencePool:
    def __init__(self):
        self.by_key = OrderedDict()   # (file, symbol) -> cid
        self.by_id = {}               # cid -> result dict

    def collect(self, retriever, queries, per_query=PER_QUERY_K, max_items=10, anchors=None):
        """Register chunks (anchors first, then query hits, max MAX_PER_FILE per file).
        Returns this group's items [(cid, result)]."""
        group, seen, per_file = [], set(), {}

        def add(r):
            k = (r["meta"]["file"], r["meta"]["symbol"])
            f = r["meta"]["file"]
            if k in seen or len(group) >= max_items or per_file.get(f, 0) >= MAX_PER_FILE:
                return
            seen.add(k)
            per_file[f] = per_file.get(f, 0) + 1
            if k not in self.by_key:
                cid = f"C{len(self.by_key) + 1}"
                self.by_key[k] = cid
                self.by_id[cid] = r
            group.append((self.by_key[k], self.by_id[self.by_key[k]]))

        for r in anchors or []:
            add(r)
        for q in queries:
            for r in retriever.search(q, top_k=per_query):
                add(r)
        return group

    @staticmethod
    def render(items, chars=None):
        return "\n\n".join(
            f"[{cid}] {r['meta']['file']} :: {r['meta']['symbol']}\n"
            f"{r['text'][:chars] if chars else r['text']}"
            for cid, r in items
        )

    @staticmethod
    def id_index(items):
        return "\n".join(f"{cid}: {r['meta']['file']} :: {r['meta']['symbol']}" for cid, r in items)

    def resolve(self, text):
        """Replace [C3] / [C1, C4] with real citation tags. Unknown IDs are dropped."""
        def sub(m):
            ids = re.findall(r"C\d+", m.group(0))
            tags = [self.by_id[i]["citation_tag"] for i in ids if i in self.by_id]
            return " ".join(tags)

        return re.sub(r"\[\s*C\d+(?:\s*[,;]\s*C\d+)*\s*\]", sub, text)


# ---------------------------------------------------------------------------
# Symbol linker: deterministic extra citations (no LLM call)
# ---------------------------------------------------------------------------

GENERIC_NAMES = {
    "main", "init", "test", "tests", "data", "name", "user", "users", "index", "model",
    "models", "setup", "config", "utils", "helper", "handler", "request", "response",
    "result", "value", "values", "items", "item", "list", "dict", "string", "number",
    "error", "start", "close", "load", "save", "create", "update", "delete", "build",
    "render", "submit", "click", "change", "login", "logout", "format",
}


def _tag_from_meta(meta):
    return (f"[cite:file={meta['file']};symbol={meta['symbol']};"
            f"lines={meta['line_start']}-{meta['line_end']}]")


def _append_tags(sent, tags):
    """Insert tags before the sentence's final punctuation."""
    end = re.search(r"[.!?]+\s*$", sent)
    joined = " ".join(tags)
    if end:
        return sent[: end.start()] + " " + joined + end.group(0)
    return sent + " " + joined


class SymbolLinker:
    """If a report sentence names a function/class that exists ONCE in the parsed index and
    carries no citation yet, attach that symbol's real tag."""

    def __init__(self, retriever):
        by_name = {}
        for meta in retriever.chunk_meta:
            sym = meta["symbol"]
            last = re.split(r"[.:#/\\]", sym)[-1]
            for n in {sym, last}:
                if len(n) >= 5 and re.fullmatch(r"\w+", n) and n.lower() not in GENERIC_NAMES:
                    by_name.setdefault(n, []).append(meta)
        self.unique = {}
        for n, metas in by_name.items():
            if len({(m["file"], m["symbol"]) for m in metas}) == 1:   # skip ambiguous names
                self.unique[n] = metas[0]
        self.regex = None
        if self.unique:
            names = sorted(self.unique, key=len, reverse=True)
            self.regex = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)")

    def link(self, text, max_per_sentence=2):
        if self.regex is None:
            return text, 0
        added = 0
        out_lines = []
        for line in text.split("\n"):
            if not line.strip() or line.lstrip().startswith("#"):
                out_lines.append(line)
                continue
            parts = re.split(r"(?<=[.!?])\s+(?!\[cite:)", line)   # never split a tag from its sentence
            new_parts = []
            for sent in parts:
                if len(sent) >= 25 and "[cite:" not in sent:
                    metas, seen = [], set()
                    for m in self.regex.finditer(sent):
                        meta = self.unique[m.group(1)]
                        k = (meta["file"], meta["symbol"])
                        if k not in seen:
                            seen.add(k)
                            metas.append(meta)
                        if len(metas) >= max_per_sentence:
                            break
                    if metas:
                        sent = _append_tags(sent, [_tag_from_meta(x) for x in metas])
                        added += len(metas)
                new_parts.append(sent)
            out_lines.append(" ".join(new_parts))
        return "\n".join(out_lines), added


# ---------------------------------------------------------------------------
# Sentence citer: attach the best-matching REAL chunk to uncited code sentences
# ---------------------------------------------------------------------------

class SentenceCiter:
    """For each still-uncited, code-describing sentence, find the most similar indexed chunk
    (same TF-IDF space as retrieval). If similarity >= threshold and that tag has not been
    reused too often, attach its real tag. Tags always come from the index, so they are
    valid; the threshold keeps them relevant. Raise the threshold for stricter relevance."""

    def __init__(self, retriever, threshold=SENT_CITER_THRESHOLD, max_uses=SENT_CITER_MAX_USES):
        self.r = retriever
        self.threshold = threshold
        self.max_uses = max_uses
        self.uses = {}

    def attach(self, text):
        if self.r.vectorizer is None:
            return text, 0
        # tags already in the report count toward the reuse cap
        for tag in TAG_RE.findall(text):
            self.uses[tag] = self.uses.get(tag, 0) + 1

        added, out_lines = 0, []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(line)
                continue
            parts = re.split(r"(?<=[.!?])\s+(?!\[cite:)", line)
            new_parts = []
            for sent in parts:
                if len(sent) >= 40 and "[cite:" not in sent:
                    sims = cosine_similarity(self.r.vectorizer.transform([sent]), self.r.matrix)[0]
                    best = int(sims.argmax())
                    if sims[best] >= self.threshold:
                        tag = _tag_from_meta(self.r.chunk_meta[best])
                        if self.uses.get(tag, 0) < self.max_uses:
                            self.uses[tag] = self.uses.get(tag, 0) + 1
                            sent = _append_tags(sent, [tag])
                            added += 1
                new_parts.append(sent)
            out_lines.append(" ".join(new_parts))
        return "\n".join(out_lines), added


def dedupe_adjacent_tags(text):
    """[tagA] [tagA] [tagB] -> [tagA] [tagB]"""
    pattern = re.compile(r"(?:\[cite:file=[^\]]*?;lines=\d+-\d+\][ \t]*){2,}")

    def sub(m):
        seen, uniq = set(), []
        for t in TAG_RE.findall(m.group(0)):
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        trail = " " if m.group(0).endswith((" ", "\t")) else ""
        return " ".join(uniq) + trail

    return pattern.sub(sub, text)


# ---------------------------------------------------------------------------
# Structure analysis (pure Python, no LLM)
# ---------------------------------------------------------------------------

def get_file_tree_and_languages(repo_root):
    ext_counts = {}
    tree_lines = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}

    for dirpath, _, filenames in os.walk(repo_root):
        if any(skip in dirpath.split(os.sep) for skip in skip_dirs):
            continue
        for fname in filenames:
            if fname.startswith(".") or fname in {"package-lock.json", "yarn.lock", "poetry.lock"}:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fname), repo_root)
            tree_lines.append(rel)
            ext = os.path.splitext(fname)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    total = sum(ext_counts.values()) or 1
    lang_summary = ", ".join(
        f"{ext or '(no ext)'}: {round(100 * c / total)}%"
        for ext, c in sorted(ext_counts.items(), key=lambda x: -x[1])[:6]
    )
    if len(tree_lines) > 60:
        tree_lines = tree_lines[:30] + ["..."] + tree_lines[-20:]
    return "\n".join(tree_lines), lang_summary


def read_project_context(repo_root):
    """README + dependency manifest: background only, NOT citable (no IDs)."""
    readme = ""
    for name in sorted(os.listdir(repo_root)):
        if name.lower().startswith("readme"):
            try:
                with open(os.path.join(repo_root, name), "r", encoding="utf-8", errors="ignore") as f:
                    lines = [
                        ln for ln in f.read().splitlines()
                        if not ln.strip().startswith("![") and "badge" not in ln.lower()
                    ]
                readme = "\n".join(lines).strip()[:README_CHARS]
            except Exception:
                pass
            break

    manifest = ""
    for name in ("requirements.txt", "package.json", "pom.xml", "build.gradle",
                 "pyproject.toml", "Pipfile"):
        path = os.path.join(repo_root, name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    manifest = f"{name}:\n" + f.read().strip()[:MANIFEST_CHARS]
            except Exception:
                pass
            break
    return readme, manifest


# ---------------------------------------------------------------------------
# The "agents" = sequential no-tool prompts
# ---------------------------------------------------------------------------

CITE_RULES = (
    "CITATION RULES: after every claim about specific code, write the ID of the supporting "
    "evidence chunk in square brackets, like [C3] or [C2, C5]. Use ONLY IDs that appear in the "
    "evidence. Never invent IDs. Never write file paths, line numbers or tags yourself."
)


def run_code_analyst(items):
    system = (
        "You are a Code Analyst. You explain what key functions and classes do, "
        "strictly from the evidence provided. " + CITE_RULES
    )
    user = (
        "EVIDENCE (code chunks):\n\n" + EvidencePool.render(items) + "\n\n"
        "TASK: Cover EVERY chunk above (one entry each, in order of importance). For each: "
        "name the function/class, say what it does, mention its key inputs/outputs or the "
        "libraries/services it calls, and why it matters to the app. at most 2 short sentences (about 45 words) each, "
        "ending with its evidence ID."
    )
    return call_llm(system, user, CODE_MAX_TOKENS)


def run_architecture_analyst(items):
    system = (
        "You are an Architecture Analyst. You describe how components interact and what design "
        "patterns appear, strictly from the evidence provided. " + CITE_RULES
    )
    user = (
        "EVIDENCE (code chunks):\n\n" + EvidencePool.render(items) + "\n\n"
        "TASK: Describe the end-to-end data/control flow step by step (request/input -> "
        "processing -> storage/output), naming the concrete functions/classes involved, then "
        "name design patterns you can actually see (e.g. MVC, repository pattern, middleware "
        "chain) and which chunk shows each. 2-3 short paragraphs (under 350 words in total); support each claim with evidence IDs."
    )
    return call_llm(system, user, ARCH_MAX_TOKENS)


WRITER_SYSTEM = (
    "You are a Senior Technical Writer producing part of a final repository report. "
    "Use only the information given. " + CITE_RULES + " README and dependency text is "
    "background only: use it for purpose and tech stack, but never attach an ID to it."
)

PART_A_TASK = (
    "TASK: Write ONLY these two sections in markdown (about 300-350 words):\n"
    "## 1. Overall purpose and architecture  (what problem it solves, who uses it, the tech "
    "stack and languages, how the folders are organized)\n"
    "## 2. Key components/modules  (a bullet per module or top-level folder: its responsibility, "
    "main functions/classes, and how it connects to other modules)\n"
    "Be specific: use real function, file and library names. Put evidence IDs on every sentence "
    "that describes code. Do not mention missing information."
)

PART_B_TASK = (
    "TASK: Write ONLY this section in markdown (about 350-400 words). Sections 1-2 are already "
    "written; do not repeat them.\n"
    "## 3. Notable functions/classes  (name each one, explain HOW it works step by step and "
    "WHY it matters; go beyond restating names)\n"
    "Put evidence IDs on every sentence that describes code, spread over many different chunks. "
    "Do not mention missing information."
)

PART_C_TASK = (
    "TASK: Write ONLY these two sections in markdown (about 300-350 words). Sections 1-3 are "
    "already written; do not repeat them.\n"
    "## 4. Data flow and design patterns  (trace one request/input end to end through concrete "
    "functions, then name each design pattern with the chunk that shows it)\n"
    "## 5. Technology stack and dependencies  (libraries/frameworks actually used and what each "
    "is used for)\n"
    "Put evidence IDs on every sentence that describes code. Do not mention missing information."
)

WRITER_PARTS = [("A (sections 1-2)", "writer_a", PART_A_TASK),
                ("B (section 3)", "writer_b", PART_B_TASK),
                ("C (sections 4-5)", "writer_c", PART_C_TASK)]


def _writer_context(tree_summary, lang_summary, code_analysis, arch_analysis, all_items,
                    readme, manifest):
    return (
        f"README (background):\n{readme or '(none)'}\n\n"
        f"DEPENDENCIES (background):\n{manifest or '(none)'}\n\n"
        f"FILE TREE (sample):\n{tree_summary[:800]}\n\n"
        f"LANGUAGE COMPOSITION: {lang_summary}\n\n"
        f"CODE ANALYSIS:\n{code_analysis}\n\n"
        f"ARCHITECTURE ANALYSIS:\n{arch_analysis}\n\n"
        f"EVIDENCE SNIPPETS (cite these by ID):\n"
        f"{EvidencePool.render(all_items, chars=WRITER_SNIPPET_CHARS)}\n\n"
    )


def run_writer_part(ctx, task, max_tokens):
    return call_llm(WRITER_SYSTEM, ctx + task, max_tokens)


_cache = {}   # repo_key -> {stage_name: text, "done": bool}


def load_cache():
    global _cache
    _cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == PROMPT_VERSION:
                _cache = data.get("repos", {})
        except Exception:
            _cache = {}
    # one-time migration: repos finished by the old per-file version stay finished
    if os.path.isdir(LEGACY_PARTIAL_DIR):
        suffix = f"_{PROMPT_VERSION}_DONE.flag"
        migrated = 0
        for fn in os.listdir(LEGACY_PARTIAL_DIR):
            if fn.endswith(suffix):
                key = fn[: -len(suffix)]
                if not _cache.get(key, {}).get("done"):
                    _cache.setdefault(key, {})["done"] = True
                    migrated += 1
        if migrated:
            print(f"Migrated {migrated} finished repos from {LEGACY_PARTIAL_DIR}/ "
                  f"(you can delete that folder now).")
            save_cache()


def save_cache():
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": PROMPT_VERSION, "repos": _cache}, f)
    os.replace(tmp, CACHE_PATH)


def cached_stage(repo_key, stage, fn):
    # Run a stage once; reuse its saved text if the repo run is retried or resumed.
    repo = _cache.setdefault(repo_key, {})
    if repo.get(stage) and len(repo[stage]) > 20:
        return repo[stage]
    out = fn()
    repo[stage] = out
    save_cache()
    return out


def mark_done(repo_key):
    _cache.setdefault(repo_key, {})["done"] = True
    save_cache()


def summarize_repo(repo_key, repo_root, retriever, tree_summary, lang_summary):
    pool = EvidencePool()

    limited = bool(OTPM_LIMIT) and OTPM_LIMIT <= 1500
    code_max = CODE_GROUP_MAX_LIMITED if limited else CODE_GROUP_MAX
    arch_max = ARCH_GROUP_MAX_LIMITED if limited else ARCH_GROUP_MAX

    readme, manifest = read_project_context(repo_root)
    code_items = pool.collect(retriever, CODE_QUERIES, max_items=code_max,
                              anchors=retriever.top_symbols())
    arch_items = pool.collect(retriever, ARCH_QUERIES, max_items=arch_max)

    all_by_id = OrderedDict()
    for cid, r in code_items + arch_items:
        all_by_id.setdefault(cid, r)
    all_items = list(all_by_id.items())

    print(f"    evidence: {len(code_items)} code chunks, {len(arch_items)} architecture chunks "
          f"({len(all_items)} unique)")

    print("    [1/5] code analyst...")
    code_analysis = cached_stage(repo_key, "code", lambda: run_code_analyst(code_items))

    print("    [2/5] architecture analyst...")
    arch_analysis = cached_stage(repo_key, "arch", lambda: run_architecture_analyst(arch_items))

    ctx = _writer_context(tree_summary, lang_summary, code_analysis, arch_analysis,
                          all_items, readme, manifest)

    parts = []
    for n, (label, stage, task) in enumerate(WRITER_PARTS, 3):
        print(f"    [{n}/5] writer part {label}...")
        parts.append(cached_stage(
            repo_key, stage,
            lambda task=task: run_writer_part(ctx, task, WRITER_MAX_TOKENS)).strip())

    raw_report = "\n\n".join(parts)

    resolved = pool.resolve(raw_report)
    linked, n_link = SymbolLinker(retriever).link(resolved)
    final, n_sent = SentenceCiter(retriever).attach(linked)
    final = dedupe_adjacent_tags(final)
    print(f"    symbol linker +{n_link}, sentence citer +{n_sent} extra citations")
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def already_done(out_path, repo_key):
    # Done = cache says finished AND the output file has real citation tags.
    if not _cache.get(repo_key, {}).get("done"):
        return False
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 100):
        return False
    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        return CITE_TAG_PREFIX in f.read()


def check_model():
    """Fail fast: ask Groq which models THIS key can use, instead of 404-ing on every repo."""
    print(f"Using API key ending ...{GROQ_API_KEY[-4:]}  |  model: {LLM_MODEL}")
    try:
        ids = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        print(f"Could not list models ({e}). Check GROQ_API_KEY.")
        return False
    if LLM_MODEL in ids:
        return True
    skip = ("whisper", "orpheus", "guard", "safeguard", "tts")
    usable = [i for i in ids if not any(s in i for s in skip)]
    print(f"\nERROR: model '{LLM_MODEL}' is not available to this API key.")
    print("Models this key CAN use:")
    for i in usable:
        print(f"  - {i}")
    print("\nPick one and run:  export AURA_MODEL=<model id>   (then python aura.py)")
    return False


def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set.")
        return
    if not check_model():
        return

    os.makedirs(AURA_OUTPUT_DIR, exist_ok=True)
    load_cache()

    if not os.path.exists(ROOT_INDEX_PATH):
        print(f"ERROR: {ROOT_INDEX_PATH} not found. Run the metadata/index build step first.")
        return

    with open(ROOT_INDEX_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    repo_keys = list(manifest.keys())
    total = len(repo_keys)
    print(f"\nFound {total} repositories to process\n")

    for idx, repo_key in enumerate(repo_keys, 1):
        entry = manifest[repo_key]
        if entry.get("skipped"):
            continue

        out_path = os.path.join(AURA_OUTPUT_DIR, f"{repo_key}.txt")
        if already_done(out_path, repo_key):
            print(f"[{idx}/{total}] {repo_key}: already done, skipping")
            continue

        repo_root = os.path.join(REPOSITORIES_DIR, entry["slug"])
        if not os.path.isdir(repo_root):
            print(f"[{idx}/{total}] {repo_key}: repo folder missing at {repo_root}")
            continue

        start = time.time()
        print(f"[{idx}/{total}] {repo_key}: building retriever...")
        retriever = RepoRetriever(repo_key, entry["index_path"], repo_root)
        if not retriever.chunks or retriever.vectorizer is None:
            print(f"[{idx}/{total}] {repo_key}: no retrievable chunks")
            continue
        print(f"[{idx}/{total}] {repo_key}: {len(retriever.chunks)} symbols indexed")

        tree_summary, lang_summary = get_file_tree_and_languages(repo_root)

        final_text = None
        for attempt in range(MAX_REPO_ATTEMPTS):
            try:
                final_text = summarize_repo(repo_key, repo_root, retriever, tree_summary, lang_summary)
                break
            except DailyQuotaExhausted as e:
                print(f"\n[STOP] Groq daily token quota exhausted: {e}")
                print(f"Finished stages are cached in {CACHE_PATH}. Re-run tomorrow to resume.")
                return
            except Exception as e:
                print(f"[{idx}/{total}] {repo_key}: ERROR (attempt {attempt + 1}): {e}")

        if final_text is None:
            print(f"[{idx}/{total}] {repo_key}: giving up (stages already done are cached)")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        mark_done(repo_key)

        n_cites = final_text.count(CITE_TAG_PREFIX)
        print(f"[{idx}/{total}] {repo_key}: NEW output saved -> {out_path} "
              f"({len(final_text.split())} words, {n_cites} citations, {time.time() - start:.1f}s)")

        if idx < total:
            time.sleep(INTER_REPO_PAUSE)

    print("\n" + "=" * 60)
    print("AURA pipeline complete!")
    print(f"Output directory: {AURA_OUTPUT_DIR}")
    print("Run citation_validator.py and summary_quality_judge.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()