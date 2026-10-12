"""
aura.py - AURA v6: bounded AGENTIC, citation-grounded repository summaries

Agents (each one is a role prompt + the Python tools it may use):

  1. PLANNER       reads tree/README/symbol overview, decides what kind of repo this is and
                   what questions must be answered. (1 call)
  2. INVESTIGATOR  a real tool loop. Each step the model CHOOSES an action as JSON:
                       {"action":"search","query":"..."}   find symbols in the repo index
                       {"action":"read","id":"C7"}         read more of a chunk it already has
                       {"action":"done"}                   stop, evidence is enough
                   It runs at most INVESTIGATOR_STEPS steps (hard budget). (2-4 small calls)
  3. EXTRACTOR     turns the evidence into short structured claims, each tied to chunk IDs. (1 call)
  4. VERIFIER      critic. Checks every claim against ONLY its cited code. Unsupported claims
                   are dropped, partly-right ones are rewritten. If too few claims survive, the
                   Extractor is sent back for one revision round. (1 call per round)
  5. WRITERS       three audience views (beginner / student / interviewer) written ONLY from
                   verified claims. The developer view is built in Python (no LLM call).

Citations: the model only ever writes chunk IDs (C3) or fact numbers ([3]). Python converts
them to real [cite:file=..;symbol=..;lines=..] tags built from the parsed index, so a tag can
never point at code that does not exist. The readable report uses footnotes [n]; the real tags
sit in a Sources section at the end (this is what citation_validator.py reads).

Per repo: about 9-12 LLM calls (planner 1 + investigator <=4 + extractor/verifier 2-4 + writers 3).

Install:  pip install openai scikit-learn python-dotenv
Run (PowerShell):
    $env:GROQ_API_KEY="..."
    python aura.py
or put GROQ_API_KEY=... in a .env file next to aura.py.

Optional env vars:  AURA_MODEL, AURA_FAST_MODEL, AURA_OTPM, AURA_REPOS (comma separated keys)
"""

import os
import re
import json
import time
import warnings
from collections import OrderedDict, Counter

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
LLM_MODEL = os.environ.get("AURA_MODEL", "qwen/qwen3.8-27b")          # writer / extractor
FAST_MODEL = os.environ.get("AURA_FAST_MODEL", LLM_MODEL)             # planner / investigator / verifier
# TIP: set AURA_FAST_MODEL to a different Groq model. Every model has its OWN rate limits,
# so splitting the agents across two models roughly doubles your throughput.
REASONING_EFFORT = "none"
REQUEST_TIMEOUT = 90

MODEL_OUTPUTS_DIR = "Model_outputs"
AURA_OUTPUT_DIR = os.path.join(MODEL_OUTPUTS_DIR, "AURA")
FACTS_DIR = "AURA_facts"              # verified claims + agent trace per repo (for your report)
CACHE_PATH = "aura_agent_cache.json"
REPOSITORIES_DIR = "repositories"
ROOT_INDEX_PATH = "metadata_index.json"

# The evaluation subset (chosen by size/stack, see chat). Override: $env:AURA_REPOS="01_...,04_..."
DEFAULT_REPOS = [
    "01_Attendance-Management-System",
    "04_ResumeRise",
    "05_bookreview-api",
    "06_Virtual-Tour-Creater",
    "08_Ai_Health_Recommedation",
    "10_HomeServicesBooking",
    "15_whatsappchatanalyzer",
    "19_AI-COACHING-PLATFORM",
]

# Output budgets (tokens per call); every call is also capped to fit OTPM_LIMIT.
PLAN_MAX_TOKENS = 450
STEP_MAX_TOKENS = 80
EXTRACT_MAX_TOKENS = 700
VERIFY_MAX_TOKENS = 350
BEGINNER_MAX_TOKENS = 330
STUDENT_MAX_TOKENS = 600
INTERVIEWER_MAX_TOKENS = 380

OTPM_LIMIT = int(os.environ.get("AURA_OTPM", "1000"))   # 0 = model has no output-token limit
OUT_CAP_FRACTION = 0.7

# Agent budgets
INVESTIGATOR_STEPS = 4        # hard cap on tool-loop steps
EVIDENCE_MAX = 24             # max chunks in the evidence pool
MAX_REVISION_ROUNDS = 2       # extra extract+verify rounds if too few claims survive
MIN_CLAIMS = 16               # fewer verified claims than this triggers a revision round

# Retrieval / prompt sizes
VECTOR_MAX_FEATURES = 2000
PER_QUERY_K = 4
SNIPPET_CONTEXT_LINES = 2
ANCHOR_SYMBOLS = 8
OBS_CHARS = 150               # per hit shown to the investigator
READ_CHARS = 700              # what a "read" action returns
EXTRACT_CHARS = 280           # per chunk shown to the extractor
VERIFY_CHARS = 300            # per chunk shown to the verifier
README_CHARS = 1000
MANIFEST_CHARS = 400

# Pacing (Groq free tier is ~8k tokens/min per model)
TPM_BUDGET = 6500
TPM_WINDOW_SECONDS = 60
MAX_LLM_RETRIES = 5
INTER_REPO_PAUSE = 5
MAX_REPO_ATTEMPTS = 2

PROMPT_VERSION = "agent-v2"   # bump to regenerate everything
CITE_TAG_PREFIX = "[cite:file="

TOPIC_TITLES = OrderedDict([
    ("purpose", "Purpose"),
    ("module", "Main modules"),
    ("flow", "How data flows"),
    ("stack", "Tech stack"),
    ("run", "How to run it"),
    ("highlight", "Worth a closer look"),
])
NO_CODE_TOPICS = {"purpose", "stack", "run"}   # may come from README alone (no chunk IDs)

LLM_CALLS = 0


class DailyQuotaExhausted(Exception):
    """Groq's per-day token cap is gone; stop the whole run."""


# ---------------------------------------------------------------------------
# Pacing + LLM call (per-model state)
# ---------------------------------------------------------------------------

class AdaptivePacer:
    def __init__(self, tpm=TPM_BUDGET, otpm=OTPM_LIMIT, window=TPM_WINDOW_SECONDS):
        self.tpm, self.otpm, self.window = tpm, otpm, window
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


class ModelState:
    def __init__(self):
        self.pacer = AdaptivePacer()
        self.out_cap = int(OTPM_LIMIT * OUT_CAP_FRACTION) if OTPM_LIMIT else 10 ** 6
        self.otpm_hits = 0
        self.send_reasoning = True


_states = {}
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)


def state_for(model):
    return _states.setdefault(model, ModelState())


def _clean(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _retry_after_seconds(msg):
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", msg)
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1


def call_llm(system, user, max_tokens, model=None):
    global LLM_CALLS
    model = model or LLM_MODEL
    st = state_for(model)
    last_err = None

    for attempt in range(MAX_LLM_RETRIES):
        max_tokens = min(max_tokens, st.out_cap)
        estimate = (len(system) + len(user)) // 3 + max_tokens
        st.pacer.wait_for(min(estimate, st.pacer.tpm), max_tokens)

        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        effort = "low" if "gpt-oss" in model else REASONING_EFFORT
        if st.send_reasoning and effort:
            kwargs["extra_body"] = {"reasoning_effort": effort}

        try:
            resp = client.chat.completions.create(**kwargs)
            LLM_CALLS += 1
            usage = getattr(resp, "usage", None)
            st.pacer.record(usage.total_tokens if usage else estimate,
                            usage.completion_tokens if usage else max_tokens)
            return _clean(resp.choices[0].message.content)

        except RateLimitError as e:
            last_err = e
            msg = str(e)
            low = msg.lower()
            if "per day" in low or "tpd" in low:
                raise DailyQuotaExhausted(msg)
            if "output tokens" in low or "otpm" in low:
                m = re.search(r"limit\s+(\d+)", low)
                limit = int(m.group(1)) if m else (OTPM_LIMIT or 1000)
                st.pacer.otpm = limit
                st.otpm_hits += 1
                new_cap = int(limit * OUT_CAP_FRACTION)
                st.out_cap = int(min(st.out_cap, new_cap) * (0.85 if st.otpm_hits > 1 else 1.0))
                print(f"    ...output-token limit ({limit}/min): capping max_tokens at {st.out_cap}")
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
            low = str(e).lower()
            if e.status_code == 413 or "too large" in low:
                user = user[: int(len(user) * 0.7)]
                print("    ...request too large, trimming prompt")
                continue
            if e.status_code == 400 and "reasoning" in low and st.send_reasoning:
                st.send_reasoning = False
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


def parse_json(text):
    """Tolerant JSON parser: strips code fences, finds the first object/array, and salvages
    arrays that were cut off by the token limit."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    a, b = t.find("{"), t.find("[")
    pairs = [("{", "}"), ("[", "]")]
    if b != -1 and (a == -1 or b < a):
        pairs.reverse()
    for o, c in pairs:
        s, e = t.find(o), t.rfind(c)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except Exception:
                pass
    s, e = t.find("["), t.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(t[s:e + 1] + "]")
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Retriever (TF-IDF over indexed symbols) = the tools the agents use
# ---------------------------------------------------------------------------

def _norm(path):
    return path.replace("\\", "/")


def tag_from_meta(meta):
    return (f"[cite:file={meta['file']};symbol={meta['symbol']};"
            f"lines={meta['line_start']}-{meta['line_end']}]")


class RepoRetriever:
    def __init__(self, repo_key, index_path, repo_root):
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.repo_key, self.repo_root = repo_key, repo_root
        self.chunks, self.chunk_meta = [], []
        self.key_to_idx = {}
        self._cache = {}
        file_cache = {}

        for key, entry in self.index.items():
            try:
                file_path, symbol = key.split("::", 1)
                full_path = os.path.join(repo_root, file_path)
                if not os.path.exists(full_path):
                    continue
                if full_path not in file_cache:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_cache[full_path] = f.readlines()
                lines = file_cache[full_path]
                start = max(0, entry["line_start"] - SNIPPET_CONTEXT_LINES)
                end = min(len(lines), entry["line_end"] + SNIPPET_CONTEXT_LINES)
                snippet = "".join(lines[start:end])
                if not snippet.strip():
                    continue
                self.key_to_idx[(file_path, symbol)] = len(self.chunks)
                self.chunks.append(f"{symbol}\n{snippet}")
                self.chunk_meta.append({"file": file_path, "symbol": symbol,
                                        "line_start": entry["line_start"],
                                        "line_end": entry["line_end"]})
            except Exception:
                continue

        self.vectorizer = self.matrix = None
        if self.chunks:
            for params in (dict(min_df=2, max_df=0.7), dict(min_df=1, max_df=0.95),
                           dict(min_df=1, max_df=1.0)):
                try:
                    vec = TfidfVectorizer(stop_words="english", max_features=VECTOR_MAX_FEATURES,
                                          ngram_range=(1, 2), **params)
                    self.matrix = vec.fit_transform(self.chunks)
                    self.vectorizer = vec
                    break
                except ValueError:
                    continue

    def _result(self, i, score):
        return {"idx": i, "meta": self.chunk_meta[i], "score": score}

    def search(self, query, top_k=PER_QUERY_K):
        key = (query, top_k)
        if key in self._cache:
            return self._cache[key]
        if self.vectorizer is None:
            return []
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        out = [self._result(int(i), float(sims[i])) for i in sims.argsort()[::-1][:top_k]
               if sims[i] > 0.03]
        self._cache[key] = out
        return out

    def largest_in_file(self, path):
        path = _norm(path).strip()
        best, best_size = None, -1
        for i, m in enumerate(self.chunk_meta):
            f = _norm(m["file"])
            if f == path or f.endswith("/" + path) or path.endswith("/" + f):
                size = m["line_end"] - m["line_start"]
                if size > best_size:
                    best, best_size = i, size
        return best

    def top_symbols(self, n=ANCHOR_SYMBOLS):
        """Largest symbol per top-level directory first (breadth), then per remaining file."""
        order = sorted(range(len(self.chunk_meta)),
                       key=lambda i: self.chunk_meta[i]["line_end"] - self.chunk_meta[i]["line_start"],
                       reverse=True)
        out, files, dirs = [], set(), set()
        for i in order:
            f = _norm(self.chunk_meta[i]["file"])
            top = f.split("/")[0] if "/" in f else "."
            if top in dirs:
                continue
            dirs.add(top)
            files.add(f)
            out.append(self._result(i, 1.0))
            if len(out) >= n:
                return out
        for i in order:
            f = _norm(self.chunk_meta[i]["file"])
            if f in files:
                continue
            files.add(f)
            out.append(self._result(i, 1.0))
            if len(out) >= n:
                break
        return out


class EvidencePool:
    """Short IDs (C1, C2...) for chunks the agents have seen. The model only writes IDs."""

    def __init__(self, retriever):
        self.r = retriever
        self.idx_to_cid = OrderedDict()
        self.cid_to_idx = {}

    def __len__(self):
        return len(self.idx_to_cid)

    def add(self, idx):
        if idx in self.idx_to_cid:
            return self.idx_to_cid[idx], False
        cid = f"C{len(self.idx_to_cid) + 1}"
        self.idx_to_cid[idx] = cid
        self.cid_to_idx[cid] = idx
        return cid, True

    def meta(self, cid):
        return self.r.chunk_meta[self.cid_to_idx[cid]]

    def label(self, cid):
        m = self.meta(cid)
        return f"{m['file']} :: {m['symbol']}"

    def tag(self, cid):
        return tag_from_meta(self.meta(cid))

    def render(self, cids, chars):
        return "\n\n".join(f"[{c}] {self.label(c)}\n{self.r.chunks[self.cid_to_idx[c]][:chars]}"
                           for c in cids if c in self.cid_to_idx)

    def keys(self):
        return [[m["file"], m["symbol"]] for m in
                (self.r.chunk_meta[i] for i in self.idx_to_cid)]

    def restore(self, keys):
        for f, s in keys:
            i = self.r.key_to_idx.get((f, s))
            if i is not None:
                self.add(i)


# ---------------------------------------------------------------------------
# Repo structure (pure Python)
# ---------------------------------------------------------------------------

def get_file_tree_and_languages(repo_root):
    ext_counts, tree_lines = {}, []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    for dirpath, _, filenames in os.walk(repo_root):
        if any(skip in dirpath.split(os.sep) for skip in skip_dirs):
            continue
        for fname in filenames:
            if fname.startswith(".") or fname in {"package-lock.json", "yarn.lock", "poetry.lock"}:
                continue
            tree_lines.append(os.path.relpath(os.path.join(dirpath, fname), repo_root))
            ext = os.path.splitext(fname)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    total = sum(ext_counts.values()) or 1
    lang = ", ".join(f"{e or '(no ext)'}: {round(100 * c / total)}%"
                     for e, c in sorted(ext_counts.items(), key=lambda x: -x[1])[:6])
    if len(tree_lines) > 60:
        tree_lines = tree_lines[:30] + ["..."] + tree_lines[-20:]
    return "\n".join(tree_lines), lang


def read_project_context(repo_root):
    readme = ""
    for name in sorted(os.listdir(repo_root)):
        if name.lower().startswith("readme"):
            try:
                with open(os.path.join(repo_root, name), "r", encoding="utf-8", errors="ignore") as f:
                    lines = [ln for ln in f.read().splitlines()
                             if not ln.strip().startswith("![") and "badge" not in ln.lower()]
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
# Agent 1: PLANNER
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS = [
    "What is the entry point and how does the application start?",
    "What are the main modules and what does each one do?",
    "How does data or a request flow through the system?",
    "Which external libraries or services does it rely on?",
]


def run_planner(retriever, tree, lang, readme, manifest):
    overview = "\n".join(f"{r['meta']['file']} :: {r['meta']['symbol']}"
                         for r in retriever.top_symbols(30))
    system = ("You are a Planner agent. You decide how to investigate a code repository so it "
              "can be explained to beginners, students and interviewers. Reply with ONE JSON "
              "object and nothing else.")
    user = (
        f"README:\n{readme or '(none)'}\n\nDEPENDENCIES:\n{manifest or '(none)'}\n\n"
        f"FILE TREE:\n{tree[:900]}\n\nLANGUAGES: {lang}\n\n"
        f"LARGEST SYMBOLS (file :: symbol):\n{overview}\n\n"
        'Return JSON: {"repo_type": "<web app | ML pipeline | bot | library | ...>", '
        '"purpose": "<one plain sentence: what it does and for whom>", '
        '"questions": ["<up to 5 specific things to find out about THIS repo>"], '
        '"focus_files": ["<up to 4 file paths from the tree that matter most>"]}'
    )
    plan = parse_json(call_llm(system, user, PLAN_MAX_TOKENS, model=FAST_MODEL))
    if not isinstance(plan, dict):
        plan = {}
    qs = [str(q)[:140] for q in plan.get("questions", []) if isinstance(q, str)][:5]
    ff = [str(f) for f in plan.get("focus_files", []) if isinstance(f, str)][:4]
    return {
        "repo_type": str(plan.get("repo_type") or "software project")[:60],
        "purpose": str(plan.get("purpose") or "")[:240],
        "questions": qs or DEFAULT_QUESTIONS,
        "focus_files": ff,
    }


# ---------------------------------------------------------------------------
# Agent 2: INVESTIGATOR (tool loop with a hard step budget)
# ---------------------------------------------------------------------------

INVESTIGATOR_SYSTEM = (
    "You are an Investigator agent exploring a code repository to answer questions. Each turn "
    "reply with ONE JSON object and nothing else. Allowed actions:\n"
    '{"action":"search","query":"<2-5 keywords>"}   find code symbols matching keywords\n'
    '{"action":"read","id":"C7"}                    read more of a chunk you already know\n'
    '{"action":"done"}                              stop: every question has evidence\n'
    "Search for what is still unanswered. Every query must use keywords not used in earlier "
    "queries. Choose done as soon as the questions are covered."
)


def _snip(text, n=OBS_CHARS):
    body = text.split("\n", 1)[1] if "\n" in text else text
    return re.sub(r"\s+", " ", body).strip()[:n]


def run_investigator(retriever, pool, plan):
    for r in retriever.top_symbols(ANCHOR_SYMBOLS):
        pool.add(r["idx"])
    for f in plan["focus_files"]:
        i = retriever.largest_in_file(f)
        if i is not None:
            pool.add(i)

    trace, queries = [], []
    last_obs = "(none yet - the known evidence below is the starting set)"
    for step in range(1, INVESTIGATOR_STEPS + 1):
        known = "\n".join(f"{c}: {pool.label(c)}" for c in list(pool.cid_to_idx)[:EVIDENCE_MAX])
        user = (
            "QUESTIONS TO ANSWER:\n" + "\n".join(f"- {q}" for q in plan["questions"]) + "\n\n"
            f"KNOWN EVIDENCE ({len(pool)}/{EVIDENCE_MAX}):\n{known}\n\n"
            f"QUERIES ALREADY RUN: {queries or 'none'}\n\n"
            f"LAST OBSERVATION:\n{last_obs}\n\n"
            f"Step {step} of {INVESTIGATOR_STEPS}. Choose your next action."
        )
        act = parse_json(call_llm(INVESTIGATOR_SYSTEM, user, STEP_MAX_TOKENS, model=FAST_MODEL))
        if not isinstance(act, dict):
            trace.append({"step": step, "action": "unparseable"})
            break
        action = str(act.get("action", "")).lower().strip()

        if action == "done":
            trace.append({"step": step, "action": "done"})
            break

        if action == "search":
            q = str(act.get("query", "")).strip()[:80]
            if not q or q in queries:
                last_obs = "Query empty or repeated. Use different keywords."
                trace.append({"step": step, "action": "search", "query": q, "note": "rejected"})
                continue
            queries.append(q)
            hits, new_ids = [], []
            for r in retriever.search(q, top_k=PER_QUERY_K):
                if len(pool) >= EVIDENCE_MAX and r["idx"] not in pool.idx_to_cid:
                    continue
                cid, is_new = pool.add(r["idx"])
                if is_new:
                    new_ids.append(cid)
                hits.append(f"{cid} {pool.label(cid)}: {_snip(retriever.chunks[r['idx']])}")
            last_obs = "\n".join(hits) or "No matching symbols."
            trace.append({"step": step, "action": "search", "query": q, "new": new_ids})
            if len(pool) >= EVIDENCE_MAX:
                break
            continue

        if action == "read":
            cid = str(act.get("id", "")).strip().upper()
            if cid in pool.cid_to_idx:
                last_obs = pool.render([cid], READ_CHARS)
                trace.append({"step": step, "action": "read", "id": cid})
            else:
                last_obs = f"Unknown id '{cid}'."
                trace.append({"step": step, "action": "read", "id": cid, "note": "unknown"})
            continue

        last_obs = "Invalid action. Use search, read or done."
        trace.append({"step": step, "action": "invalid"})
    return trace


# ---------------------------------------------------------------------------
# Agents 3+4: EXTRACTOR and VERIFIER (with a revision loop)
# ---------------------------------------------------------------------------

def _words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _is_dup(text, existing, thr=0.5):
    w = _words(text)
    for e in existing:
        ew = _words(e)
        if w and ew and len(w & ew) / len(w | ew) >= thr:
            return True
    return False


EXTRACTOR_SYSTEM = (
    "You are a Fact Extractor. You write short, plain, factual claims about a code repository, "
    "each backed by evidence chunk IDs. You never state anything the evidence does not show. "
    "Reply with a JSON array and nothing else."
)


def extract_claims(plan, pool, readme, manifest, verified):
    cids = list(pool.cid_to_idx)
    done = "\n".join(f"- {c['c']}" for c in verified) or "(none)"
    user = (
        f"PROJECT: {plan['repo_type']} - {plan['purpose']}\n\n"
        f"README (background, may support purpose/stack/run claims only):\n{readme[:600] or '(none)'}\n"
        f"DEPENDENCIES:\n{manifest or '(none)'}\n\n"
        f"EVIDENCE:\n{pool.render(cids, EXTRACT_CHARS)}\n\n"
        f"ALREADY VERIFIED (do not repeat):\n{done}\n\n"
        'TASK: Return a JSON array of up to 10 items: {"t": topic, "c": claim, "ids": ["C3"], "w": 1-3}\n'
        "topic is one of: purpose, module, flow, stack, run, highlight. w = importance (3 = key).\n"
        "Rules: each claim is at most 30 words, plain wording, says WHAT the code does and "
        "HOW it works or WHY it matters. Every claim about code lists 1-3 IDs that directly show it. "
        "Only purpose, stack and run claims taken from the README/dependencies may have empty ids. "
        "Never write claims about what is missing, not shown or not listed. Do not repeat or "
        "rephrase a claim already verified; cover different code.\n"
        "Cover, as far as the evidence allows: 1 purpose, 3-4 module, 2 flow, 1 stack, 1 run, "
        "2 highlight."
    )
    raw = parse_json(call_llm(EXTRACTOR_SYSTEM, user, EXTRACT_MAX_TOKENS))
    out = []
    seen = [c["c"] for c in verified]
    used = Counter(tuple(c["ids"]) for c in verified if c["ids"])
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict):
            continue
        text = str(it.get("c", "")).strip()
        topic = str(it.get("t", "")).strip().lower()
        ids = [i for i in (it.get("ids") or []) if isinstance(i, str) and i in pool.cid_to_idx][:3]
        if topic not in TOPIC_TITLES or len(text) < 15 or _is_dup(text, seen):
            continue
        if ids and used[tuple(ids)] >= 2:      # at most 2 claims per identical evidence
            continue
        if not ids and topic not in NO_CODE_TOPICS:
            continue
        try:
            w = max(1, min(3, int(it.get("w", 2))))
        except (TypeError, ValueError):
            w = 2
        seen.append(text)
        used[tuple(ids)] += 1
        out.append({"t": topic, "c": text, "ids": ids, "w": w})
    return out[:10]


VERIFIER_SYSTEM = (
    "You are a strict Verifier agent. For each claim, judge it ONLY against the SOURCE shown "
    "under it. Verdicts: S = fully supported by the source; P = partly supported (give a "
    "corrected claim in 'fix', at most 30 words, using only what the source shows); U = not "
    "supported or invented. A fix must state only what the source shows. Never mention what is "
    "missing or not shown. Reply with a JSON array and nothing else."
)

BAD_FIX = re.compile(
    r"\b(not (?:specifically|listed|mentioned|shown|provided)|no evidence|"
    r"provided (?:source|package|evidence))\b", re.I)


def verify_claims(claims, pool, readme, manifest):
    blocks = []
    for n, cl in enumerate(claims, 1):
        src = (pool.render(cl["ids"], VERIFY_CHARS) if cl["ids"]
               else f"README/DEPENDENCIES:\n{readme[:400]}\n{manifest[:300]}")
        blocks.append(f"CLAIM {n}: {cl['c']}\nSOURCE:\n{src}")
    user = ("\n\n".join(blocks) +
            '\n\nReturn a JSON array with one item per claim: {"n": 1, "v": "S"} '
            'or {"n": 2, "v": "P", "fix": "..."} or {"n": 3, "v": "U"}.')
    for _ in range(2):   # one retry if the answer is not parseable
        parsed = parse_json(call_llm(VERIFIER_SYSTEM, user, VERIFY_MAX_TOKENS, model=FAST_MODEL))
        if isinstance(parsed, list):
            res = {}
            for v in parsed:
                try:
                    res[int(v["n"])] = v
                except Exception:
                    continue
            return res
    return None


def build_verified_claims(plan, pool, readme, manifest):
    verified, rounds = [], []
    for rnd in range(MAX_REVISION_ROUNDS + 1):
        new = extract_claims(plan, pool, readme, manifest, verified)
        if not new:
            rounds.append({"round": rnd, "extracted": 0})
            break
        verdicts = verify_claims(new, pool, readme, manifest)
        s = p = u = 0
        kept = []
        for n, cl in enumerate(new, 1):
            if verdicts is None:            # verifier failed twice: keep but flag
                cl["verdict"] = "unchecked"
                kept.append(cl)
                continue
            v = str(verdicts.get(n, {}).get("v", "U")).upper()[:1]
            if v == "S":
                s += 1
                cl["verdict"] = "S"
                kept.append(cl)
            elif v == "P":
                p += 1
                fix = str(verdicts[n].get("fix", "")).strip()
                if len(fix) >= 15 and not BAD_FIX.search(fix):
                    cl["c"], cl["verdict"] = fix, "P-fixed"
                    kept.append(cl)
            else:
                u += 1
        rounds.append({"round": rnd, "extracted": len(new), "supported": s, "partial": p,
                       "unsupported": u, "verifier_failed": verdicts is None, "kept": len(kept)})
        verified.extend(kept)
        if len(verified) >= MIN_CLAIMS:
            break
    return verified, rounds


# ---------------------------------------------------------------------------
# Agent 5: AUDIENCE WRITERS (+ deterministic developer view)
# ---------------------------------------------------------------------------

WRITER_SYSTEM = (
    "You explain a code repository to a specific audience. Use ONLY the numbered FACTS given "
    "(plus the project line). Never add technical details that are not in the facts. After a "
    "sentence that relies on facts, add the fact numbers like [3] or [2][5], at most once per "
    "bullet or paragraph. Never write file paths, line numbers or other tags. An everyday "
    "analogy is allowed but must be introduced with the word 'Analogy:'."
)

VIEW_TASKS = {
    "beginner": (
        "TASK: Write in markdown, at most 130 words, for someone who has never programmed:\n"
        "## In plain English\n"
        "- 2-3 sentences: what this project is and who would use it (no jargon)\n"
        "- One line starting 'Analogy:'\n"
        "- 'What you can do with it:' then 3 short bullets"
    ),
    "student": (
        "TASK: Write in markdown, at most 230 words, for a CS student reading this code for the "
        "first time:\n"
        "## For students: how to read this project\n"
        "1. A numbered reading order of 4-5 files, one line each saying why to read it. Use "
        "ONLY these files: {files}\n"
        "2. A Mermaid diagram in a ```mermaid code fence: 'flowchart LR' with at most 7 nodes "
        "like A[Browser] --> B[Server]. Node labels: plain words only, no brackets or quotes inside.\n"
        "3. 'Key terms:' 3 short glossary entries for terms used in the facts."
    ),
    "interviewer": (
        "TASK: Write in markdown, at most 160 words, for a recruiter or interviewer:\n"
        "## For recruiters and interviewers\n"
        "- **What it demonstrates:** 3 bullets of concrete skills shown by the facts\n"
        "- **Stack:** one line\n"
        "- **Most interesting part:** 1-2 sentences"
    ),
}


def clean_footnotes(text, n_max):
    text = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]",
                  lambda m: "".join(f"[{x.strip()}]" for x in m.group(1).split(",")), text)
    parts = text.split("```")
    for i in range(0, len(parts), 2):   # only outside code fences
        parts[i] = re.sub(r"\[(\d+)\]",
                          lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_max else "", parts[i])
    return "```".join(parts).strip()


def write_views(plan, lang, numbered, files, n_max):
    facts = "\n".join(f"[{n}] {cl['c']}" for n, cl in numbered)
    head = f"PROJECT: {plan['repo_type']} - {plan['purpose']}\nLANGUAGES: {lang}\n\nFACTS:\n{facts}\n\n"
    caps = {"beginner": BEGINNER_MAX_TOKENS, "student": STUDENT_MAX_TOKENS,
            "interviewer": INTERVIEWER_MAX_TOKENS}
    views = {}
    for kind in ("beginner", "student", "interviewer"):
        task = VIEW_TASKS[kind].replace("{files}", ", ".join(files) or "(none)")
        views[kind] = clean_footnotes(call_llm(WRITER_SYSTEM, head + task, caps[kind]), n_max)
    return views


def developer_view(numbered):
    out = ["## For developers: verified claims"]
    for topic, title in TOPIC_TITLES.items():
        rows = [(n, cl) for n, cl in numbered if cl["t"] == topic]
        if rows:
            out.append(f"\n### {title}")
            out.extend(f"- {cl['c']} [{n}]" for n, cl in rows)
    return "\n".join(out)


def sources_section(numbered):
    out = ["## Sources (each [n] is a claim checked against this code)"]
    for n, cl in numbered:
        if cl["tags"]:
            out.append(f"[{n}] " + " ".join(cl["tags"]))
        else:
            out.append(f"[{n}] README / dependency files")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Cache (one file)
# ---------------------------------------------------------------------------

_cache = {}


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


def save_cache():
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": PROMPT_VERSION, "repos": _cache}, f)
    os.replace(tmp, CACHE_PATH)


def cached_stage(repo_key, stage, fn):
    repo = _cache.setdefault(repo_key, {})
    if stage in repo and repo[stage]:
        return repo[stage]
    out = fn()
    repo[stage] = out
    save_cache()
    return out


def mark_done(repo_key):
    _cache.setdefault(repo_key, {})["done"] = True
    save_cache()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def summarize_repo(repo_key, repo_root, retriever):
    calls_before = LLM_CALLS
    tree, lang = get_file_tree_and_languages(repo_root)
    readme, manifest = read_project_context(repo_root)
    pool = EvidencePool(retriever)

    print("    [1] planner...")
    plan = cached_stage(repo_key, "plan", lambda: run_planner(retriever, tree, lang, readme, manifest))

    print("    [2] investigator (tool loop)...")

    def investigate():
        trace = run_investigator(retriever, pool, plan)
        return {"keys": pool.keys(), "trace": trace}

    inv = cached_stage(repo_key, "investigation", investigate)
    if not len(pool):                       # restored from cache: rebuild the same IDs
        pool.restore(inv["keys"])
    print(f"        {len(inv['trace'])} steps, {len(pool)} evidence chunks")

    print("    [3-4] extractor + verifier...")
    claims_pack = cached_stage(
        repo_key, "claims",
        lambda: dict(zip(("claims", "rounds"), build_verified_claims(plan, pool, readme, manifest))))
    claims, rounds = claims_pack["claims"], claims_pack["rounds"]
    if not claims:
        raise RuntimeError("no verified claims survived")

    order = list(TOPIC_TITLES)
    claims = sorted(claims, key=lambda c: (order.index(c["t"]), -c["w"]))
    for cl in claims:
        cl["tags"] = [pool.tag(i) for i in cl["ids"] if i in pool.cid_to_idx]
        cl["labels"] = [pool.label(i) for i in cl["ids"] if i in pool.cid_to_idx]
    numbered = list(enumerate(claims, 1))
    files = list(OrderedDict.fromkeys(
        m.split(" :: ")[0] for cl in claims for m in cl["labels"]))[:8]

    print("    [5] audience writers...")
    views = cached_stage(repo_key, "views",
                         lambda: write_views(plan, lang, numbered, files, len(claims)))

    title = re.sub(r"^\d+_", "", repo_key)
    report = "\n\n".join([
        f"# {title} - AURA report",
        f"> {plan['purpose'] or plan['repo_type']}  \n> Type: {plan['repo_type']} | Languages: {lang}",
        views["beginner"], views["student"], views["interviewer"],
        developer_view(numbered), sources_section(numbered),
    ])

    first = [r for r in rounds if r.get("extracted")]
    ext = sum(r["extracted"] for r in first)
    sup = sum(r.get("supported", 0) for r in first)
    facts = {
        "repo": repo_key, "plan": plan, "agent_trace": inv["trace"], "verifier_rounds": rounds,
        "first_pass_faithfulness": round(sup / ext, 3) if ext else None,
        "claims": claims, "llm_calls_this_run": LLM_CALLS - calls_before,
    }
    return report, facts


def already_done(out_path, repo_key):
    if not _cache.get(repo_key, {}).get("done"):
        return False
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 100):
        return False
    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        return CITE_TAG_PREFIX in f.read()


def check_model():
    models = {LLM_MODEL, FAST_MODEL}
    print(f"Using API key ending ...{GROQ_API_KEY[-4:]} | writer model: {LLM_MODEL} | "
          f"agent model: {FAST_MODEL}")
    try:
        ids = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        print(f"Could not list models ({e}). Check GROQ_API_KEY.")
        return False
    missing = [m for m in models if m not in ids]
    if not missing:
        return True
    skip = ("whisper", "orpheus", "guard", "safeguard", "tts")
    print(f"\nERROR: model(s) not available to this key: {missing}\nModels this key CAN use:")
    for i in ids:
        if not any(s in i for s in skip):
            print(f"  - {i}")
    print('\nPowerShell:  $env:AURA_MODEL="<id>"   (and optionally $env:AURA_FAST_MODEL="<id>")')
    return False


def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set.")
        return
    if not check_model():
        return
    os.makedirs(AURA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FACTS_DIR, exist_ok=True)
    load_cache()

    if not os.path.exists(ROOT_INDEX_PATH):
        print(f"ERROR: {ROOT_INDEX_PATH} not found.")
        return
    with open(ROOT_INDEX_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    env_repos = os.environ.get("AURA_REPOS", "")
    selected = [k.strip() for k in env_repos.split(",") if k.strip()] or DEFAULT_REPOS
    print(f"\nProcessing {len(selected)} repositories\n")

    for idx, repo_key in enumerate(selected, 1):
        entry = manifest.get(repo_key)
        if not entry or entry.get("skipped"):
            print(f"[{idx}/{len(selected)}] {repo_key}: not in {ROOT_INDEX_PATH} (or skipped)")
            continue
        out_path = os.path.join(AURA_OUTPUT_DIR, f"{repo_key}.txt")
        if already_done(out_path, repo_key):
            print(f"[{idx}/{len(selected)}] {repo_key}: already done, skipping")
            continue
        repo_root = os.path.join(REPOSITORIES_DIR, entry["slug"])
        if not os.path.isdir(repo_root):
            print(f"[{idx}/{len(selected)}] {repo_key}: repo folder missing at {repo_root}")
            continue

        start = time.time()
        print(f"[{idx}/{len(selected)}] {repo_key}: building retriever...")
        retriever = RepoRetriever(repo_key, entry["index_path"], repo_root)
        if not retriever.chunks or retriever.vectorizer is None:
            print(f"[{idx}/{len(selected)}] {repo_key}: no retrievable chunks")
            continue
        print(f"[{idx}/{len(selected)}] {repo_key}: {len(retriever.chunks)} symbols indexed")

        result = None
        for attempt in range(MAX_REPO_ATTEMPTS):
            try:
                result = summarize_repo(repo_key, repo_root, retriever)
                break
            except DailyQuotaExhausted as e:
                print(f"\n[STOP] Groq daily token quota exhausted: {e}")
                print(f"Finished stages are cached in {CACHE_PATH}. Re-run later to resume.")
                return
            except Exception as e:
                print(f"[{idx}/{len(selected)}] {repo_key}: ERROR (attempt {attempt + 1}): {e}")
        if result is None:
            print(f"[{idx}/{len(selected)}] {repo_key}: giving up (finished stages are cached)")
            continue

        report, facts = result
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(FACTS_DIR, f"{repo_key}.json"), "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        mark_done(repo_key)

        print(f"[{idx}/{len(selected)}] {repo_key}: saved -> {out_path} "
              f"({len(report.split())} words, {report.count(CITE_TAG_PREFIX)} citations, "
              f"{len(facts['claims'])} verified claims, first-pass faithfulness "
              f"{facts['first_pass_faithfulness']}, {facts['llm_calls_this_run']} LLM calls, "
              f"{time.time() - start:.0f}s)")
        if idx < len(selected):
            time.sleep(INTER_REPO_PAUSE)

    print("\n" + "=" * 60)
    print("AURA agentic pipeline complete.")
    print(f"Reports: {AURA_OUTPUT_DIR} | agent traces + claims: {FACTS_DIR}")
    print("Next: citation_validator.py, then summary_quality_judge.py --rejudge AURA")
    print("=" * 60)


if __name__ == "__main__":
    main()