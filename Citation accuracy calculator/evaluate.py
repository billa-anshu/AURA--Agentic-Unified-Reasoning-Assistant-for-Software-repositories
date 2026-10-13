"""
evaluate.py - ONE script for every comparison, ONE results page.

It replaces citation_validator.py + faithfulness_checker.py + summary_quality_judge.py
(+ repo_summary.py). It runs the same checks, reuses the verdicts and scores you already
paid for, stores everything in ONE results file and builds ONE page you can open in a browser.

Stages (run in this order; all by default)
  citations      do the [cite:...] tags point at real code?                    (local, free)
  faithfulness   does the cited code actually SUPPORT the sentence?            (independent judge model)
  quality        completeness / accuracy / clarity / depth / structure, 1-5    (judge model)
  report         print the tables and write evaluation/results.html + results.json  (local, free)

Usage (PowerShell)
  python evaluate.py                        # everything; API calls only for items not judged yet
  python evaluate.py --report-only          # NO API calls: rebuild the page from cached results
  python evaluate.py --stages citations,report
  python evaluate.py --systems AURA,claude_idx --max-claims 15
  python evaluate.py --dev-only             # faithfulness: only the "For developers" part
  python evaluate.py --rejudge AURA         # re-score one system's quality (or: all)
  python evaluate.py --exclude-dev          # leave the tuning repo out of every table
  python evaluate.py --open                 # open the page in your browser when done

Settings (.env or $env:...)
  AURA_REPOS=01_...,04_...        repos to score (default: the 8 selected repos)
  AURA_DEV_REPOS=19_AI-...        repos used for tuning; shown separately (default: 19_AI-COACHING-PLATFORM)
  AURA_FAITH_MODEL=openai/gpt-oss-120b      faithfulness judge (must differ from AURA's own models)
  AURA_JUDGE_GROQ_MODEL=openai/gpt-oss-120b quality judge on Groq (default: groq_runner's gpt-oss-20b)
  AURA_JUDGE_REPEATS=1            quality judge repeats per pair (default 2)
  AURA_BASE_URL / AURA_API_KEY    any OpenAI-compatible provider instead of Groq
  GROQ_API_KEY                    in .env
"""

import os
import re
import csv
import json
import math
import time
import shutil
import random
import hashlib
import argparse
import datetime
import html as htmllib
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_OUTPUTS_DIR = "Model_outputs"
REPOSITORIES_DIR = "repositories"
PROMPTS_DIR = "prompts"
ROOT_INDEX_PATH = "metadata_index.json"
FACTS_DIRS = ["AURA_facts", "AURA_crewai_facts"]

OUT_DIR = "evaluation"
STATE_DIR = os.path.join(OUT_DIR, "state")
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")
REPORT_HTML = os.path.join(OUT_DIR, "results.html")
DETAIL_CSV = os.path.join(OUT_DIR, "faithfulness_detail.csv")
SPOTCHECK_TXT = os.path.join(OUT_DIR, "faithfulness_spotcheck.txt")

FAITH_CACHE = os.path.join(STATE_DIR, "faithfulness_cache.json")
QUALITY_RAW = os.path.join(STATE_DIR, "summary_quality_raw.json")
QUALITY_VERSION = os.path.join(STATE_DIR, "summary_quality_judge_version.txt")
LEGACY_STATE = {                       # old root-level files are copied in once, so nothing is re-judged
    "faithfulness_cache.json": FAITH_CACHE,
    "summary_quality_raw.json": QUALITY_RAW,
    "summary_quality_judge_version.txt": QUALITY_VERSION,
}

EXCLUDE_SYSTEMS = {"gemini"}
OURS = "AURA"

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
_dev = os.environ.get("AURA_DEV_REPOS", "19_AI-COACHING-PLATFORM")
DEV_REPOS = {k.strip() for k in _dev.split(",") if k.strip()}

GROQ_URL = "https://api.groq.com/openai/v1"
DIMENSIONS = ["completeness", "accuracy", "clarity", "depth", "structure"]

# citations
TOLERANCE = 3
CITE_RE = re.compile(r"\[cite:file=(?P<file>[^;]+);symbol=(?P<symbol>[^;]+);lines=(?P<lines>\d+-\d+)\]")

# faithfulness judge
FAITH_MODEL = os.environ.get("AURA_FAITH_MODEL", "openai/gpt-oss-120b")
FAITH_SLEEP = float(os.environ.get("FAITH_SLEEP", "2"))
MAX_TAGS_PER_CLAIM = 2
CODE_CHARS = 1200
BATCH = 8
TAG_RE = re.compile(r"\[cite:file=(?P<file>[^;]+);symbol=(?P<symbol>[^;]+);lines=(?P<a>\d+)-(?P<b>\d+)\]")
TAG_CLUSTER_RE = re.compile(r"(?:\[cite:file=[^;]+;symbol=[^;]+;lines=\d+-\d+\][\s,;]*)+")
FOOT_CLUSTER_RE = re.compile(r"(?:\[\d+\][\s,]*)+")

# quality judge
JUDGE_MODEL = "gpt-oss-20b"           # default path: groq_runner (keeps your existing scores valid)
ALT_BASE_URL = os.environ.get("AURA_BASE_URL")
ALT_API_KEY = os.environ.get("AURA_API_KEY")
ALT_JUDGE_MODEL = os.environ.get("AURA_JUDGE_MODEL")
GROQ_JUDGE_MODEL = os.environ.get("AURA_JUDGE_GROQ_MODEL")
ALT_SLEEP = float(os.environ.get("AURA_JUDGE_SLEEP", "8"))
JUDGE_REPEATS = int(os.environ.get("AURA_JUDGE_REPEATS", "2"))
JUDGE_VERSION = "v5-readme-context"
REPO_CONTEXT_CHAR_LIMIT = 11000
PER_FILE_CHARS = 350
SUMMARY_CHAR_LIMIT = 7000

try:
    from groq_runner import DailyQuotaExhausted as _GroqQuota
except Exception:                       # groq_runner.py not present: only needed for the default judge
    class _GroqQuota(Exception):
        pass


class DailyQuotaExhausted(Exception):
    pass


QUOTA_ERRORS = (DailyQuotaExhausted, _GroqQuota)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def migrate_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    for old, new in LEGACY_STATE.items():
        if os.path.exists(old) and not os.path.exists(new):
            shutil.copy2(old, new)
            print(f"Copied existing {old} -> {new} (old file is no longer needed)")


def norm_path(p):
    return p.strip().replace(os.sep, "/").replace("\\", "/")


def systems_present(wanted):
    if not os.path.isdir(MODEL_OUTPUTS_DIR):
        return []
    out = []
    for s in sorted(os.listdir(MODEL_OUTPUTS_DIR)):
        p = os.path.join(MODEL_OUTPUTS_DIR, s)
        if not os.path.isdir(p) or s in EXCLUDE_SYSTEMS:
            continue
        if wanted and s.lower() not in wanted:
            continue
        if any(os.path.exists(os.path.join(p, f"{r}.txt")) for r in SELECTED):
            out.append(s)
    return out


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


_index_cache = {}


def load_repo_index(key, manifest):
    if key in _index_cache:
        return _index_cache[key]
    entry = manifest.get(key)
    idx = None
    if entry and not entry.get("skipped"):
        p = entry.get("index_path")
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                idx = json.load(f)
    _index_cache[key] = idx
    return idx


# ---------------------------------------------------------------------------
# Stage 1: citation validity (local)
# ---------------------------------------------------------------------------

def extract_citations(text):
    out = []
    for m in CITE_RE.finditer(text):
        a, b = m["lines"].split("-")
        out.append({"file": norm_path(m["file"]), "symbol": m["symbol"].strip(),
                    "line_start": int(a), "line_end": int(b)})
    return out


def validate_citation(c, index):
    e = index.get(f"{c['file']}::{c['symbol']}")
    if not e:
        return False
    cited = set(range(c["line_start"], c["line_end"] + 1))
    actual = set(range(e["line_start"] - TOLERANCE, e["line_end"] + TOLERANCE + 1))
    return bool(cited & actual)


def stage_citations(systems, manifest):
    out = defaultdict(dict)
    for system in systems:
        for repo in SELECTED:
            path = os.path.join(MODEL_OUTPUTS_DIR, system, f"{repo}.txt")
            index = load_repo_index(repo, manifest)
            if not os.path.exists(path) or index is None:
                continue
            cits = extract_citations(read_text(path))
            out[system][repo] = {"citations": len(cits),
                                 "passed": sum(1 for c in cits if validate_citation(c, index))}
    return out


# ---------------------------------------------------------------------------
# Stage 2: faithfulness (does the cited code support the sentence?)
# ---------------------------------------------------------------------------

_faith_client = None


def call_faith_judge(system, user):
    global _faith_client
    if _faith_client is None:
        if ALT_BASE_URL and ALT_API_KEY:
            base, key = ALT_BASE_URL, ALT_API_KEY
        else:
            base, key = GROQ_URL, os.environ.get("GROQ_API_KEY", "")
        _faith_client = OpenAI(api_key=key, base_url=base, timeout=120, max_retries=0)
    for attempt in range(6):
        kwargs = dict(model=FAITH_MODEL, temperature=0, max_tokens=500,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if "gpt-oss" in FAITH_MODEL:
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            resp = _faith_client.chat.completions.create(**kwargs)
            time.sleep(FAITH_SLEEP)
            return resp.choices[0].message.content or ""
        except RateLimitError as e:
            low = str(e).lower()
            if "per day" in low or "tpd" in low or "daily" in low:
                raise DailyQuotaExhausted(str(e))
            m = re.search(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", low)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else 20 * (attempt + 1)
            print(f"    ...rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                print(f"Judge model error ({e.status_code}) for '{FAITH_MODEL}': {e}")
                raise
            time.sleep(10 * (attempt + 1))
        except (APITimeoutError, APIConnectionError):
            print("    ...network error, retrying")
            time.sleep(5 * (attempt + 1))
    return ""


FAITH_SYSTEM = (
    "You are a strict code reviewer. For each ITEM you get a STATEMENT about a codebase and the "
    "CODE it cites. Decide using ONLY the code shown. S = the code fully supports the statement; "
    "P = part of the statement is supported but part is unsupported or wrong; U = the code does "
    "not support it or contradicts it. Do not penalize a statement for leaving details out. "
    "Respond with ONLY a JSON array like [{\"n\":1,\"v\":\"S\"},{\"n\":2,\"v\":\"U\"}]."
)


def judge_batch(items):
    blocks = ["ITEM %d\nSTATEMENT: %s\nCODE:\n%s" % (n, it["claim"], it["code"]) for n, it in enumerate(items, 1)]
    out = call_faith_judge(FAITH_SYSTEM, "\n\n".join(blocks))
    verdicts = {int(n): v.upper() for n, v in re.findall(r'"n"\s*:\s*(\d+)\s*,\s*"v"\s*:\s*"([SPUspu])', out)}
    return [verdicts.get(n) for n in range(1, len(items) + 1)]


def parse_sources(text):
    m = re.search(r"^## Sources.*$", text, flags=re.M)
    if not m:
        return text, None
    body, src = text[:m.start()], text[m.end():]
    foot = {}
    for line in src.splitlines():
        lm = re.match(r"\s*\[(\d+)\]\s*(.*)", line)
        if lm:
            foot[int(lm.group(1))] = [t.groupdict() for t in TAG_RE.finditer(lm.group(2))]
    return body, foot


def extract_claims(text, dev_only=False):
    body, foot = parse_sources(text)
    pattern = FOOT_CLUSTER_RE if foot is not None else TAG_CLUSTER_RE
    use_dev = dev_only and re.search(r"^#{1,3}.*developer", body, flags=re.M | re.I)
    in_dev = not use_dev
    in_code = False
    claims = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not s:
            continue
        if s.startswith("#"):
            if use_dev:
                if "developer" in s.lower():
                    in_dev = True
                elif re.match(r"#{1,2}\s", s):
                    in_dev = False
            continue
        if not in_dev:
            continue
        prev = 0
        for m in pattern.finditer(line):
            segment = line[prev:m.start()]
            prev = m.end()
            claim = re.sub(r"^[\s\-\*\u2022>]+(?:\d+[\.\)]\s+)?", "", segment).strip()[-400:]
            if foot is not None:
                tags = []
                for n in re.findall(r"\d+", m.group(0)):
                    tags.extend(foot.get(int(n), []))
            else:
                tags = [t.groupdict() for t in TAG_RE.finditer(m.group(0))]
            if len(claim) >= 20 and tags:
                claims.append({"claim": claim, "tags": tags})
    seen, out = set(), []
    for c in claims:
        k = (c["claim"], tuple((t["file"], t["symbol"], t["a"], t["b"]) for t in c["tags"]))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


class RepoData:
    def __init__(self, entry, index):
        self.root = os.path.join(REPOSITORIES_DIR, entry["slug"])
        self.index = index
        self._lines = {}

    def lines(self, rel):
        if rel not in self._lines:
            try:
                with open(os.path.join(self.root, rel), "r", encoding="utf-8", errors="ignore") as f:
                    self._lines[rel] = f.readlines()
            except OSError:
                self._lines[rel] = []
        return self._lines[rel]

    def resolve(self, tag):
        file_ = tag["file"].strip().replace("\\", "/")
        e = self.index.get(f"{file_}::{tag['symbol'].strip()}")
        if not e:
            return None
        a, b = int(tag["a"]), int(tag["b"])
        if not (set(range(a, b + 1)) & set(range(e["line_start"] - TOLERANCE, e["line_end"] + TOLERANCE + 1))):
            return None
        code = "".join(self.lines(file_)[e["line_start"] - 1:e["line_end"]])[:CODE_CHARS]
        return "# %s :: %s\n%s" % (file_, tag["symbol"].strip(), code) if code.strip() else None


def faith_key(claim, code):
    return hashlib.sha1(("%s|%s|%s" % (FAITH_MODEL, claim, code)).encode("utf-8")).hexdigest()


def stage_faithfulness(systems, manifest, max_claims, dev_only, offline):
    cache = load_json(FAITH_CACHE, {})
    repos, items = {}, []
    total_cited = defaultdict(int)
    for system in systems:
        for repo in SELECTED:
            path = os.path.join(MODEL_OUTPUTS_DIR, system, f"{repo}.txt")
            entry, index = manifest.get(repo), load_repo_index(repo, manifest)
            if not os.path.exists(path) or not entry or index is None:
                continue
            if repo not in repos:
                repos[repo] = RepoData(entry, index)
            claims = extract_claims(read_text(path), dev_only)
            total_cited[(system, repo)] = len(claims)
            if len(claims) > max_claims:               # same cap + seed rule for every system
                claims = random.Random("%s|%s" % (repo, system)).sample(claims, max_claims)
            for c in claims:
                codes = [x for x in (repos[repo].resolve(t) for t in c["tags"][:MAX_TAGS_PER_CLAIM]) if x]
                code = "\n\n".join(codes)
                items.append({"system": system, "repo": repo, "claim": c["claim"],
                              "tags": ["%s::%s:%s-%s" % (t["file"], t["symbol"], t["a"], t["b"]) for t in c["tags"]],
                              "code": code, "valid": bool(code),
                              "key": faith_key(c["claim"], code) if code else None, "verdict": None})

    pending = []
    for it in items:
        if it["valid"]:
            it["verdict"] = cache.get(it["key"])
            if it["verdict"] is None:
                pending.append(it)
    print("faithfulness: %d sampled cited claims, %d with broken tags, %d still to judge"
          % (len(items), sum(1 for i in items if not i["valid"]), len(pending)))

    stopped = False
    if pending and offline:
        print("  (offline: not calling the judge; unjudged claims are shown as such)")
    elif pending:
        print("  judge: %s, %d calls" % (FAITH_MODEL, -(-len(pending) // BATCH)))
        gen_models = {os.environ.get("AURA_MODEL"), os.environ.get("AURA_FAST_MODEL")} - {None}
        if FAITH_MODEL in gen_models:
            print("  WARNING: the faithfulness judge is also an AURA model; set AURA_FAITH_MODEL to a different one.")
        for i in range(0, len(pending), BATCH):
            batch = pending[i:i + BATCH]
            try:
                verdicts = judge_batch(batch)
            except DailyQuotaExhausted as e:
                print("  STOP: daily quota exhausted (%s). Re-run later; finished claims are cached." % str(e)[:80])
                stopped = True
                break
            for it, v in zip(batch, verdicts):
                if v in ("S", "P", "U"):
                    it["verdict"] = v
                    cache[it["key"]] = v
            save_json(FAITH_CACHE, cache)
            print("  judged %d/%d" % (min(i + BATCH, len(pending)), len(pending)))

    per = defaultdict(dict)
    for system in systems:
        for repo in SELECTED:
            if (system, repo) in total_cited:
                per[system][repo] = {"cited": total_cited[(system, repo)], "broken": 0, "S": 0, "P": 0,
                                     "U": 0, "unjudged": 0}
    for it in items:
        d = per[it["system"]][it["repo"]]
        if not it["valid"]:
            d["broken"] += 1
        elif it["verdict"] is None:
            d["unjudged"] += 1
        else:
            d[it["verdict"]] += 1
    return per, items, stopped


def write_faith_detail(items):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "repo", "verdict", "claim", "tags"])
        for it in items:
            w.writerow([it["system"], it["repo"], it["verdict"] or ("BROKEN_TAG" if not it["valid"] else ""),
                        it["claim"], " | ".join(it["tags"])])
    judged = [it for it in items if it["verdict"]]
    sample = random.Random(7).sample(judged, min(20, len(judged)))
    with open(SPOTCHECK_TXT, "w", encoding="utf-8") as f:
        f.write("Check these by hand: do you agree with the judge's verdict?\n\n")
        for it in sample:
            f.write("[%s | %s] VERDICT: %s\nCLAIM: %s\nCODE:\n%s\n%s\n"
                    % (it["system"], it["repo"], it["verdict"], it["claim"], it["code"][:600], "-" * 70))


# ---------------------------------------------------------------------------
# Stage 3: summary quality (completeness, accuracy, clarity, depth, structure)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an impartial evaluator scoring a repository summary
for quality. You are given a repository digest and a summary written about it.
Score the summary on 5 dimensions, each from 1 (poor) to 5 (excellent).

The digest contains the README, the dependency file, the file tree and the largest
function/class from each file. It is NOT the whole repository. Do not penalize the summary for
describing code that is not shown; penalize accuracy only when a claim contradicts the digest
or is clearly invented.

=== REPOSITORY DIGEST ===
{repo_context}
=== END DIGEST ===

=== SUMMARY TO EVALUATE ===
{summary}
=== END SUMMARY ===

Scoring guide:
- completeness: covers the project's purpose, main modules, key functions, data flow
- accuracy: claims are consistent with the digest and not invented
- clarity: easy to read and understand
- depth: explains HOW and WHY, not just names
- structure: well organized with sections, not a wall of text

Respond with ONLY this JSON, nothing else:
{{"completeness": <1-5>, "accuracy": <1-5>, "clarity": <1-5>, "depth": <1-5>, "structure": <1-5>}}
"""

CITE_STRIP_RE = re.compile(r"[ \t]*\[cite:file=.*?;lines=\s*\d+\s*-\s*\d+\s*\]", re.DOTALL)
_alt_client = None


def judge_target():
    if ALT_BASE_URL and ALT_API_KEY and ALT_JUDGE_MODEL:
        return ALT_BASE_URL, ALT_API_KEY, ALT_JUDGE_MODEL
    if GROQ_JUDGE_MODEL and os.environ.get("GROQ_API_KEY"):
        return GROQ_URL, os.environ["GROQ_API_KEY"], GROQ_JUDGE_MODEL
    return None


def judge_model_name():
    t = judge_target()
    return t[2] if t else JUDGE_MODEL


def judge_identity():
    return "%s|%s|x%d" % (JUDGE_VERSION, judge_model_name(), JUDGE_REPEATS)


def call_quality_judge(prompt):
    global _alt_client
    target = judge_target()
    if target is None:
        from groq_runner import call_groq_model
        return call_groq_model(JUDGE_MODEL, prompt)
    base_url, api_key, model = target
    if _alt_client is None:
        _alt_client = OpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=0)
    for attempt in range(5):
        kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=800)
        if "gpt-oss" in model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            resp = _alt_client.chat.completions.create(**kwargs)
            time.sleep(ALT_SLEEP)
            return resp.choices[0].message.content
        except RateLimitError as e:
            low = str(e).lower()
            if "per day" in low or "tpd" in low or "daily" in low:
                raise DailyQuotaExhausted(str(e))
            if "too large" in low:
                print("    ...request too large for this model, skipping pair")
                return None
            m = re.search(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", low)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else 20 * (attempt + 1)
            print("    ...rate limited, waiting %.0fs" % wait)
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                print("    Judge model error (%s) for '%s': %s" % (e.status_code, model, e))
                raise
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print("    ...network error (%s), retrying" % e)
            time.sleep(5 * (attempt + 1))
    return None


def normalize_for_judge(text):
    text = text.split("## Sources")[0]
    text = CITE_STRIP_RE.sub("", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def read_project_background(repo_root):
    parts = []
    try:
        for name in sorted(os.listdir(repo_root)):
            if name.lower().startswith("readme"):
                parts.append("README:\n" + read_text(os.path.join(repo_root, name))[:1200])
                break
    except Exception:
        pass
    for name in ("requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml"):
        p = os.path.join(repo_root, name)
        if os.path.exists(p):
            parts.append("%s:\n%s" % (name, read_text(p)[:600]))
            break
    return "\n\n".join(parts)


def build_neutral_context(index, repo_root):
    best = {}
    for key, e in index.items():
        if "::" not in key:
            continue
        f, sym = key.split("::", 1)
        size = int(e["line_end"]) - int(e["line_start"])
        if f not in best or size > best[f][0]:
            best[f] = (size, sym, e)
    files = sorted(best)
    parts = ["FILES WITH CODE:\n" + "\n".join(files[:80])]
    cache = {}
    for f in files:
        _, sym, e = best[f]
        try:
            full = os.path.join(repo_root, f)
            if full not in cache:
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    cache[full] = fh.readlines()
            code = "".join(cache[full][int(e["line_start"]) - 1:int(e["line_end"])])
            parts.append("# %s :: %s\n%s" % (f, sym, code[:PER_FILE_CHARS]))
        except Exception:
            continue
    return (read_project_background(repo_root) + "\n\n" + "\n\n".join(parts)).strip()[:REPO_CONTEXT_CHAR_LIMIT]


def load_repo_context(repo_key, manifest):
    entry = manifest.get(repo_key) or {}
    index = load_repo_index(repo_key, manifest)
    slug = entry.get("slug")
    if index is not None and slug:
        root = os.path.join(REPOSITORIES_DIR, slug)
        if os.path.isdir(root):
            ctx = build_neutral_context(index, root)
            if ctx.strip():
                return ctx
    pp = os.path.join(PROMPTS_DIR, repo_key + ".txt")
    if os.path.exists(pp):
        t = read_text(pp)
        m = re.search(r"=== REPOSITORY CONTENT ===\n(.*?)\n=== END REPOSITORY CONTENT ===", t, re.DOTALL)
        return (m.group(1) if m else t)[:REPO_CONTEXT_CHAR_LIMIT]
    return None


def extract_scores(text):
    if text is None:
        return None
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in DIMENSIONS:
        try:
            out[k] = min(5.0, max(1.0, float(d[k])))
        except (KeyError, TypeError, ValueError):
            return None
    return out


def judge_one(context, summary_text):
    clean = normalize_for_judge(summary_text)
    if not clean:
        return None, "empty_after_stripping_tags"
    prompt = JUDGE_PROMPT.format(repo_context=context[:REPO_CONTEXT_CHAR_LIMIT], summary=clean[:SUMMARY_CHAR_LIMIT])
    runs = []
    for _ in range(max(1, JUDGE_REPEATS)):
        s = extract_scores(call_quality_judge(prompt))
        if s:
            runs.append(s)
    if not runs:
        return None, "call_failed_or_unparseable"
    return {d: round(sum(r[d] for r in runs) / len(runs), 2) for d in DIMENSIONS}, "ok"


def stage_quality(systems, manifest, rejudge, offline):
    raw = load_json(QUALITY_RAW, {})
    saved_version = read_text(QUALITY_VERSION).strip() if os.path.exists(QUALITY_VERSION) else ""

    def clear(arg):
        wanted = {s.strip().lower() for s in arg.split(",") if s.strip()}
        n = 0
        for system, repos in raw.items():
            if "all" in wanted or system.lower() in wanted:
                for k in repos:
                    if k in SELECTED and repos[k] is not None:
                        repos[k] = None
                        n += 1
        print("quality: cleared %d scores to re-judge" % n)

    if not offline:
        if rejudge:
            clear(rejudge)
        elif saved_version and saved_version != judge_identity():
            print("quality: judge changed ('%s' -> '%s'): re-scoring ALL selected pairs" % (saved_version, judge_identity()))
            clear("all")

    def missing():
        return [(s, r) for s in systems for r in SELECTED
                if os.path.exists(os.path.join(MODEL_OUTPUTS_DIR, s, r + ".txt")) and raw.get(s, {}).get(r) is None]

    todo = [] if offline else missing()
    before = sum(1 for s in raw.values() for k, v in s.items() if v is not None and k in SELECTED)
    stopped = False
    if todo:
        print("quality: %d pairs to judge (judge: %s, repeats: %d)" % (len(todo), judge_model_name(), JUDGE_REPEATS))
    ctx_cache = {}
    for system, repo in todo:
        raw.setdefault(system, {})
        summary = read_text(os.path.join(MODEL_OUTPUTS_DIR, system, repo + ".txt")).strip()
        if not summary:
            continue
        if repo not in ctx_cache:
            ctx_cache[repo] = load_repo_context(repo, manifest)
        if not ctx_cache[repo]:
            print("[%s][%s] no repo context available - skipping" % (system, repo))
            continue
        print("[%s][%s] judging..." % (system, repo))
        try:
            scores, status = judge_one(ctx_cache[repo], summary)
        except QUOTA_ERRORS:
            print("STOP: daily quota exhausted. Scores so far are saved; re-run later to continue.")
            stopped = True
            break
        if scores is None:
            print("[%s][%s] %s - skipping" % (system, repo, status))
            continue
        raw[system][repo] = scores
        print("[%s][%s] -> %s" % (system, repo, scores))

    after = sum(1 for s in raw.values() for k, v in s.items() if v is not None and k in SELECTED)
    if todo and after > before:
        if os.path.exists(QUALITY_RAW):
            shutil.copy2(QUALITY_RAW, QUALITY_RAW + ".bak")
        save_json(QUALITY_RAW, raw)
        with open(QUALITY_VERSION, "w", encoding="utf-8") as f:
            f.write(judge_identity())
    elif todo:
        print("quality: nothing new was scored, files left untouched")
    return {s: {r: raw.get(s, {}).get(r) for r in SELECTED if raw.get(s, {}).get(r)} for s in systems}


# ---------------------------------------------------------------------------
# Aggregation + report: plain tables, numbers only
# ---------------------------------------------------------------------------

esc = htmllib.escape


def display_name(s):
    if s == OURS:
        return "AURA (ours)"
    if s.startswith(OURS):
        return s + " (ours)"
    if s.endswith("_idx"):
        return s[:-4] + " + index"
    return s


def order_systems(systems):
    ours = [s for s in systems if s.startswith(OURS)]
    plain = sorted(s for s in systems if not s.startswith(OURS) and not s.endswith("_idx"))
    idx = sorted(s for s in systems if not s.startswith(OURS) and s.endswith("_idx"))
    return ours + plain + idx


def repo_score(res, system, repo, key):
    q = res["quality"].get(system, {}).get(repo)
    if not q:
        return None
    if key == "overall":
        return sum(q[d] for d in DIMENSIONS) / len(DIMENSIONS)
    return q.get(key)


def agg_citations(res, system, repos):
    rows = [x for x in (res["citations"].get(system, {}).get(r) for r in repos) if x]
    n = sum(x["citations"] for x in rows)
    k = sum(x["passed"] for x in rows)
    return {"repos": len(rows), "citations": n, "passed": k, "acc": 100.0 * k / n if n else None}


def agg_quality(res, system, repos):
    out = {"repos": len([r for r in repos if res["quality"].get(system, {}).get(r)])}
    for key in DIMENSIONS + ["overall"]:
        vals = [v for v in (repo_score(res, system, r, key) for r in repos) if v is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def agg_faith(res, system, repos):
    f = [x for x in (res["faithfulness"].get(system, {}).get(r) for r in repos) if x]
    S, P, U, B, unj = (sum(x[k] for x in f) for k in ("S", "P", "U", "broken", "unjudged"))
    judged = S + P + U
    return {"repos": len(f), "checked": judged + B, "S": S, "P": P, "U": U, "broken": B, "unjudged": unj,
            "valid_only": 100.0 * S / judged if judged else None,
            "strict": 100.0 * S / (judged + B) if judged + B else None}


def C(text, val=None):
    return (str(text), val)


def pct(v):
    return "-" if v is None else "%.1f%%" % v


def num(v):
    return "-" if v is None else "%.2f" % v


def make_tables(res, systems, repos):
    ours = [s.startswith(OURS) for s in systems]
    names = [display_name(s) for s in systems]
    tables = []

    rows = []
    for s in systems:
        a = agg_citations(res, s, repos)
        rows.append([C(display_name(s)), C(a["repos"]), C(a["citations"]), C(a["passed"]), C(pct(a["acc"]), a["acc"])])
    tables.append({"title": "1. Citation accuracy",
                   "sub": "Share of citations that point at real code (file, symbol, line range).",
                   "headers": ["System", "Repos", "Citations", "Valid", "Accuracy"],
                   "rows": rows, "mode": "col", "best_cols": [4], "ours_rows": ours})

    rows = []
    for s in systems:
        a = agg_quality(res, s, repos)
        rows.append([C(display_name(s)), C(a["repos"])] + [C(num(a[k]), a[k]) for k in DIMENSIONS + ["overall"]])
    tables.append({"title": "2. Summary quality (average over repos, 1-5)",
                   "sub": "Judge scores per dimension; Overall = mean of the five.",
                   "headers": ["System", "Repos", "Completeness", "Accuracy", "Clarity", "Depth", "Structure", "Overall"],
                   "rows": rows, "mode": "col", "best_cols": [2, 3, 4, 5, 6, 7], "ours_rows": ours})

    rows, unj = [], 0
    for s in systems:
        a = agg_faith(res, s, repos)
        unj += a["unjudged"]
        rows.append([C(display_name(s)), C(a["repos"]), C(a["checked"]), C(a["S"]), C(a["P"]), C(a["U"]), C(a["broken"]),
                     C(pct(a["valid_only"]), a["valid_only"]), C(pct(a["strict"]), a["strict"])])
    tables.append({"title": "3. Faithfulness (does the cited code support the sentence?)",
                   "sub": "Independent judge reads each cited sentence and only the code it cites. "
                          "'Valid tags' ignores broken citations; 'Broken tags fail' counts them as failures.",
                   "headers": ["System", "Repos", "Claims checked", "Supported", "Partly", "Not supported", "Broken tag",
                               "Faithfulness (valid tags)", "Faithfulness (broken tags fail)"],
                   "rows": rows, "mode": "col", "best_cols": [7, 8], "ours_rows": ours,
                   "note": ("%d claims are not judged yet; run python evaluate.py again to finish them." % unj) if unj else ""})

    def repo_label(r):
        return r + (" *" if r in DEV_REPOS else "")

    rows = []
    for r in repos:
        row = [C(repo_label(r))]
        for s in systems:
            d = res["citations"].get(s, {}).get(r)
            v = 100.0 * d["passed"] / d["citations"] if d and d["citations"] else None
            row.append(C("-" if v is None else "%.1f%% (%d)" % (v, d["citations"]), v))
        rows.append(row)
    tables.append({"title": "4. Per repo: citation accuracy",
                   "sub": "Accuracy, with the number of citations in brackets.",
                   "headers": ["Repo"] + names, "rows": rows, "mode": "row", "ours_cols": [i + 1 for i, o in enumerate(ours) if o]})

    rows = []
    for r in repos:
        row = [C(repo_label(r))]
        for s in systems:
            v = repo_score(res, s, r, "overall")
            row.append(C(num(v), v))
        rows.append(row)
    tables.append({"title": "5. Per repo: summary quality (overall, 1-5)", "sub": "",
                   "headers": ["Repo"] + names, "rows": rows, "mode": "row", "ours_cols": [i + 1 for i, o in enumerate(ours) if o]})
    return tables


def _best_flags(t):
    flags = [[False] * len(r) for r in t["rows"]]
    if t["mode"] == "col":
        for c in t["best_cols"]:
            vals = [r[c][1] for r in t["rows"] if r[c][1] is not None]
            if len(vals) > 1 and max(vals) > min(vals):
                for i, r in enumerate(t["rows"]):
                    flags[i][c] = r[c][1] is not None and abs(r[c][1] - max(vals)) < 1e-9
    else:
        for i, r in enumerate(t["rows"]):
            vals = [x[1] for x in r[1:] if x[1] is not None]
            if len(vals) > 1 and max(vals) > min(vals):
                for c in range(1, len(r)):
                    flags[i][c] = r[c][1] is not None and abs(r[c][1] - max(vals)) < 1e-9
    return flags


def table_text(t):
    cols = len(t["headers"])
    cells = [t["headers"]] + [[c[0] for c in r] for r in t["rows"]]
    widths = [max(len(row[i]) for row in cells) + 2 for i in range(cols)]
    lines = [t["title"]]
    if t.get("sub"):
        lines.append(t["sub"])
    lines.append("".join(h.ljust(widths[i]) for i, h in enumerate(t["headers"])))
    lines.append("-" * sum(widths))
    for row in cells[1:]:
        lines.append("".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    if t.get("note"):
        lines.append("NOTE: " + t["note"])
    return "\n".join(lines)


CSS = """
body{margin:0;background:#fff;color:#1b2430;font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 2px}h2{font-size:16px;margin:28px 0 2px}
.sub{color:#5d6b7a;margin:0 0 8px;font-size:13px}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%}th,td{padding:6px 10px;border-bottom:1px solid #e3e7ec;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#f5f7f9;font-size:12.5px;color:#4a5865}
.ours{background:#e6f4f2}.best{font-weight:700}.note{color:#92600a;font-size:13px;margin-top:6px}
.foot{color:#5d6b7a;font-size:12.5px;margin-top:22px}
"""


def table_html(t):
    flags = _best_flags(t)
    ours_cols = set(t.get("ours_cols", []))
    out = ['<h2>%s</h2>' % esc(t["title"])]
    if t.get("sub"):
        out.append('<p class="sub">%s</p>' % esc(t["sub"]))
    out.append('<div class="scroll"><table><tr>')
    for i, h in enumerate(t["headers"]):
        out.append('<th class="%s">%s</th>' % ("ours" if i in ours_cols else "", esc(h)))
    out.append("</tr>")
    for ri, row in enumerate(t["rows"]):
        row_ours = t["mode"] == "col" and t["ours_rows"][ri]
        out.append("<tr>")
        for ci, cell in enumerate(row):
            cls = []
            if row_ours or ci in ours_cols:
                cls.append("ours")
            if flags[ri][ci]:
                cls.append("best")
            out.append('<td class="%s">%s</td>' % (" ".join(cls), esc(cell[0])))
        out.append("</tr>")
    out.append("</table></div>")
    if t.get("note"):
        out.append('<div class="note">%s</div>' % esc(t["note"]))
    return "".join(out)


def build_page(tables, meta):
    foot = ["Bold = best value in the column (tables 1-3) or row (tables 4-5). Teal = AURA."]
    if meta["dev_in_tables"]:
        foot.append("* = repo used while tuning AURA.")
    foot.append("Faithfulness judge: %s. Quality judge: %s. Generated %s." %
                (meta["faith_model"], meta["quality_judge"], meta["generated"]))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>AURA results</title>'
            "<style>%s</style></head><body><div class=\"wrap\"><h1>AURA results</h1>"
            '<p class="sub">%d systems, %d repositories</p>%s<p class="foot">%s</p></div></body></html>'
            % (CSS, meta["n_systems"], meta["n_repos"], "".join(table_html(t) for t in tables), esc(" ".join(foot))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="citations,faithfulness,quality,report")
    ap.add_argument("--systems", default="", help="comma-separated system folders (default: all)")
    ap.add_argument("--max-claims", type=int, default=25, help="sampled cited claims per system per repo")
    ap.add_argument("--dev-only", action="store_true", help="faithfulness: only the 'For developers' part")
    ap.add_argument("--exclude-dev", action="store_true", help="leave the tuning repo(s) out of every table")
    ap.add_argument("--rejudge", default="", help="'all' or comma-separated system names (quality scores)")
    ap.add_argument("--report-only", action="store_true", help="no API calls; rebuild the tables from cached results")
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    args = ap.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    offline = args.report_only
    os.makedirs(OUT_DIR, exist_ok=True)
    migrate_state()

    if not os.path.exists(ROOT_INDEX_PATH):
        print("ERROR: %s not found." % ROOT_INDEX_PATH)
        return
    manifest = load_json(ROOT_INDEX_PATH, {})
    wanted = {s.strip().lower() for s in args.systems.split(",") if s.strip()}
    systems = systems_present(wanted)
    if not systems:
        print("No system folders with selected repos under %s/." % MODEL_OUTPUTS_DIR)
        return
    print("Systems: %s\nRepos (%d): %s\n" % (systems, len(SELECTED), SELECTED))

    prev = load_json(RESULTS_JSON, {})
    res = {"citations": prev.get("citations", {}), "faithfulness": prev.get("faithfulness", {}),
           "quality": prev.get("quality", {})}

    if "citations" in stages:
        for s, d in stage_citations(systems, manifest).items():
            res["citations"][s] = d
        print("citations: done")
    if "faithfulness" in stages:
        per, items, _ = stage_faithfulness(systems, manifest, args.max_claims, args.dev_only, offline)
        for s, d in per.items():
            res["faithfulness"][s] = d
        write_faith_detail(items)
    if "quality" in stages:
        for s, d in stage_quality(systems, manifest, args.rejudge, offline).items():
            res["quality"][s] = d

    if "report" not in stages:
        save_json(RESULTS_JSON, dict(res))
        return

    ordered = order_systems(systems)
    repos = [r for r in SELECTED if (not args.exclude_dev or r not in DEV_REPOS)
             and any(r in res[k].get(s, {}) for s in ordered for k in ("citations", "quality", "faithfulness"))]
    tables = make_tables(res, ordered, repos)
    meta = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "faith_model": FAITH_MODEL,
            "quality_judge": judge_model_name(), "n_systems": len(ordered), "n_repos": len(repos),
            "dev_in_tables": any(r in DEV_REPOS for r in repos)}

    print("\n" + "=" * 78)
    for t in tables:
        print("\n" + table_text(t))
    print("\n" + "=" * 78)

    res_out = dict(res)
    res_out["meta"] = {"systems": ordered, "repos": repos, "dev_repos": sorted(DEV_REPOS),
                       "faith_model": FAITH_MODEL, "quality_judge": judge_model_name(), "generated": meta["generated"]}
    res_out["tables"] = [{"title": t["title"], "headers": t["headers"], "rows": [[c[0] for c in r] for r in t["rows"]]}
                         for t in tables]
    save_json(RESULTS_JSON, res_out)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(build_page(tables, meta))
    print("\nSaved -> %s (same tables as above), %s" % (REPORT_HTML, RESULTS_JSON))
    if args.open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(REPORT_HTML))


if __name__ == "__main__":
    main()