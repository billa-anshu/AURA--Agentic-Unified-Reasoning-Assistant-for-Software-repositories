"""
faithfulness_checker.py
Does the cited code actually SUPPORT the sentence that cites it?

citation_validator.py only checks that a tag points to a real place. This script checks the
sentence: for every cited claim in every system's output, an independent judge model reads the
claim and ONLY the cited code, and answers S (supported), P (partly) or U (not supported).
Every system is scored the same way.

Understands both output styles:
  * baselines: inline tags  "... claim [cite:file=..;symbol=..;lines=a-b]"
  * AURA:      footnotes    "... claim [3]"  + a "## Sources" section mapping [3] to tags

Metrics per system
  strict    = S / all sampled cited claims   (a tag that points nowhere counts as unsupported)
  valid-only= S / claims whose tag is valid  (judges the sentences, ignoring broken tags)
  soft      = (S + 0.5*P) / claims whose tag is valid

Usage (PowerShell):
    python faithfulness_checker.py
    python faithfulness_checker.py --systems AURA,claude_idx --max-claims 15
    python faithfulness_checker.py --dev-only        # compare only the "For developers" part

Judge model (must differ from the model that wrote/verified AURA's text). In .env or PowerShell:
    AURA_FAITH_MODEL=openai/gpt-oss-120b            # default; uses GROQ_API_KEY
    or  AURA_BASE_URL=... AURA_API_KEY=... AURA_FAITH_MODEL=...   (any OpenAI-compatible API)
    FAITH_SLEEP=2                                   # seconds between calls
Results are cached in faithfulness_cache.json, so a re-run continues where it stopped.
"""

import os
import re
import csv
import json
import time
import random
import hashlib
import argparse
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError

MODEL_OUTPUTS_DIR = "Model_outputs"
REPOSITORIES_DIR = "repositories"
ROOT_INDEX_PATH = "metadata_index.json"
CACHE_PATH = "faithfulness_cache.json"
EXCLUDE_SYSTEMS = {"gemini"}

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

GROQ_URL = "https://api.groq.com/openai/v1"
FAITH_MODEL = os.environ.get("AURA_FAITH_MODEL", "openai/gpt-oss-120b")
ALT_BASE_URL = os.environ.get("AURA_BASE_URL")
ALT_API_KEY = os.environ.get("AURA_API_KEY")
SLEEP = float(os.environ.get("FAITH_SLEEP", "2"))

TOLERANCE = 3            # same line tolerance as citation_validator.py
MAX_TAGS_PER_CLAIM = 2
CODE_CHARS = 1200        # code shown to the judge per tag
BATCH = 8                # claims per judge call

TAG_RE = re.compile(r"\[cite:file=(?P<file>[^;]+);symbol=(?P<symbol>[^;]+);lines=(?P<a>\d+)-(?P<b>\d+)\]")
TAG_CLUSTER_RE = re.compile(r"(?:\[cite:file=[^;]+;symbol=[^;]+;lines=\d+-\d+\][\s,;]*)+")
FOOT_CLUSTER_RE = re.compile(r"(?:\[\d+\][\s,]*)+")


class DailyQuotaExhausted(Exception):
    pass


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

_client = None


def _target():
    if ALT_BASE_URL and ALT_API_KEY:
        return ALT_BASE_URL, ALT_API_KEY
    return GROQ_URL, os.environ.get("GROQ_API_KEY", "")


def call_judge(system, user):
    global _client
    if _client is None:
        base, key = _target()
        _client = OpenAI(api_key=key, base_url=base, timeout=120, max_retries=0)
    for attempt in range(6):
        kwargs = dict(model=FAITH_MODEL, temperature=0, max_tokens=500,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}])
        if "gpt-oss" in FAITH_MODEL:
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            resp = _client.chat.completions.create(**kwargs)
            time.sleep(SLEEP)
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


JUDGE_SYSTEM = (
    "You are a strict code reviewer. For each ITEM you get a STATEMENT about a codebase and the "
    "CODE it cites. Decide using ONLY the code shown. S = the code fully supports the statement; "
    "P = part of the statement is supported but part is unsupported or wrong; U = the code does "
    "not support it or contradicts it. Do not penalize a statement for leaving details out. "
    "Respond with ONLY a JSON array like [{\"n\":1,\"v\":\"S\"},{\"n\":2,\"v\":\"U\"}]."
)


def judge_batch(items):
    blocks = [f"ITEM {n}\nSTATEMENT: {it['claim']}\nCODE:\n{it['code']}" for n, it in enumerate(items, 1)]
    out = call_judge(JUDGE_SYSTEM, "\n\n".join(blocks))
    verdicts = {int(n): v.upper() for n, v in re.findall(r'"n"\s*:\s*(\d+)\s*,\s*"v"\s*:\s*"([SPUspu])', out)}
    return [verdicts.get(n) for n in range(1, len(items) + 1)]


# ---------------------------------------------------------------------------
# Extract (claim, tags) pairs from an output file
# ---------------------------------------------------------------------------

def parse_sources(text):
    """Split off an AURA-style '## Sources' section: returns (body, {n: [tag dicts]} or None)."""
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
    for c in claims:                                   # drop exact duplicates
        k = (c["claim"], tuple((t["file"], t["symbol"], t["a"], t["b"]) for t in c["tags"]))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Resolve tags against the index and fetch the cited code
# ---------------------------------------------------------------------------

class RepoData:
    def __init__(self, entry):
        self.root = os.path.join(REPOSITORIES_DIR, entry["slug"])
        with open(entry["index_path"], "r", encoding="utf-8") as f:
            self.index = json.load(f)
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
        """Return the cited code if the tag is valid (same rule as citation_validator), else None."""
        file_ = tag["file"].strip().replace("\\", "/")
        e = self.index.get(f"{file_}::{tag['symbol'].strip()}")
        if not e:
            return None
        a, b = int(tag["a"]), int(tag["b"])
        if not (set(range(a, b + 1)) & set(range(e["line_start"] - TOLERANCE, e["line_end"] + TOLERANCE + 1))):
            return None
        src = self.lines(file_)[e["line_start"] - 1:e["line_end"]]
        code = "".join(src)[:CODE_CHARS]
        return f"# {file_} :: {tag['symbol'].strip()}\n{code}" if code.strip() else None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def item_key(claim, code):
    return hashlib.sha1(f"{FAITH_MODEL}|{claim}|{code}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="", help="comma-separated system folders (default: all)")
    ap.add_argument("--max-claims", type=int, default=25, help="sampled cited claims per system per repo")
    ap.add_argument("--dev-only", action="store_true", help="only claims under the 'For developers' header")
    args = ap.parse_args()

    if not (ALT_API_KEY or os.environ.get("GROQ_API_KEY")):
        print("ERROR: set GROQ_API_KEY (or AURA_BASE_URL + AURA_API_KEY).")
        return
    gen_models = {os.environ.get("AURA_MODEL"), os.environ.get("AURA_FAST_MODEL")} - {None}
    if FAITH_MODEL in gen_models:
        print(f"WARNING: the faithfulness judge ({FAITH_MODEL}) is also an AURA generator/verifier model. "
              f"Use a different one (AURA_FAITH_MODEL) for an independent check.")

    with open(ROOT_INDEX_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    wanted = {s.strip().lower() for s in args.systems.split(",") if s.strip()}
    systems = [s for s in sorted(os.listdir(MODEL_OUTPUTS_DIR))
               if os.path.isdir(os.path.join(MODEL_OUTPUTS_DIR, s)) and s not in EXCLUDE_SYSTEMS
               and (not wanted or s.lower() in wanted)]
    print(f"Judge: {FAITH_MODEL} | systems: {systems} | repos: {len(SELECTED)} | "
          f"max {args.max_claims} claims per system per repo | dev-only: {args.dev_only}\n")

    repos = {}
    items = []
    total_cited = defaultdict(int)
    for system in systems:
        for repo_key in SELECTED:
            path = os.path.join(MODEL_OUTPUTS_DIR, system, f"{repo_key}.txt")
            entry = manifest.get(repo_key)
            if not os.path.exists(path) or not entry:
                continue
            if repo_key not in repos:
                repos[repo_key] = RepoData(entry)
            rd = repos[repo_key]
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                claims = extract_claims(f.read(), args.dev_only)
            total_cited[system] += len(claims)
            if len(claims) > args.max_claims:               # same cap and seed rule for every system
                claims = random.Random(f"{repo_key}|{system}").sample(claims, args.max_claims)
            for c in claims:
                codes = []
                for t in c["tags"][:MAX_TAGS_PER_CLAIM]:
                    code = rd.resolve(t)
                    if code:
                        codes.append(code)
                code = "\n\n".join(codes)
                items.append({"system": system, "repo": repo_key, "claim": c["claim"],
                              "tags": [f"{t['file']}::{t['symbol']}:{t['a']}-{t['b']}" for t in c["tags"]],
                              "code": code, "valid": bool(code),
                              "key": item_key(c["claim"], code) if code else None, "verdict": None})

    cache = load_cache()
    pending = []
    for it in items:
        if it["valid"]:
            it["verdict"] = cache.get(it["key"])
            if it["verdict"] is None:
                pending.append(it)
    print(f"{len(items)} sampled cited claims; {sum(1 for i in items if not i['valid'])} with broken tags; "
          f"{len(pending)} to judge ({-(-len(pending) // BATCH)} calls)\n")

    stopped = False
    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        try:
            verdicts = judge_batch(batch)
        except DailyQuotaExhausted as e:
            print(f"\nSTOP: daily quota exhausted ({e}). Re-run later; finished claims are cached.")
            stopped = True
            break
        for it, v in zip(batch, verdicts):
            if v in ("S", "P", "U"):
                it["verdict"] = v
                cache[it["key"]] = v
        save_cache(cache)
        print(f"  judged {min(i + BATCH, len(pending))}/{len(pending)}")

    # ---- aggregate ----
    stats = defaultdict(lambda: defaultdict(int))
    for it in items:
        s = stats[it["system"]]
        s["repos"] += 0
        if not it["valid"]:
            s["broken"] += 1
        elif it["verdict"] is None:
            s["unjudged"] += 1
        else:
            s[it["verdict"]] += 1
    for system in stats:
        stats[system]["repos"] = len({it["repo"] for it in items if it["system"] == system})

    print(f"\n{'System':<16}{'Repos':<7}{'Cited':<8}{'Sampled':<9}{'Broken':<8}{'S':<6}{'P':<6}{'U':<6}"
          f"{'Strict %':<10}{'Valid-only %':<14}{'Soft %':<8}")
    print("-" * 104)
    rows = []
    for system in systems:
        s = stats.get(system)
        if not s:
            continue
        judged = s["S"] + s["P"] + s["U"]
        sampled = judged + s["broken"]
        strict = 100 * s["S"] / sampled if sampled else 0.0
        valid_only = 100 * s["S"] / judged if judged else 0.0
        soft = 100 * (s["S"] + 0.5 * s["P"]) / judged if judged else 0.0
        print(f"{system:<16}{s['repos']:<7}{total_cited[system]:<8}{sampled:<9}{s['broken']:<8}"
              f"{s['S']:<6}{s['P']:<6}{s['U']:<6}{strict:<10.1f}{valid_only:<14.1f}{soft:<8.1f}")
        rows.append({"system": system, "repos": s["repos"], "cited_claims_total": total_cited[system],
                     "sampled": sampled, "broken_tags": s["broken"], "supported": s["S"],
                     "partial": s["P"], "unsupported": s["U"], "unjudged": s["unjudged"],
                     "strict_pct": round(strict, 1), "valid_only_pct": round(valid_only, 1),
                     "soft_pct": round(soft, 1)})
    if any(r["unjudged"] for r in rows):
        print("\nNOTE: some claims are unjudged (quota or unparseable replies); re-run to finish them.")

    with open("faithfulness_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["system"])
        w.writeheader()
        w.writerows(rows)
    with open("faithfulness_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "repo", "verdict", "claim", "tags"])
        for it in items:
            w.writerow([it["system"], it["repo"], it["verdict"] or ("BROKEN_TAG" if not it["valid"] else ""),
                        it["claim"], " | ".join(it["tags"])])

    judged_items = [it for it in items if it["verdict"]]
    sample = random.Random(7).sample(judged_items, min(20, len(judged_items)))
    with open("faithfulness_spotcheck.txt", "w", encoding="utf-8") as f:
        f.write("Check these by hand: do you agree with the judge's verdict?\n\n")
        for it in sample:
            f.write(f"[{it['system']} | {it['repo']}] VERDICT: {it['verdict']}\n"
                    f"CLAIM: {it['claim']}\nCODE:\n{it['code'][:600]}\n{'-' * 70}\n")

    print("\nSaved -> faithfulness_summary.csv, faithfulness_detail.csv, faithfulness_spotcheck.txt")
    if stopped:
        print("Run stopped early (quota): the table above covers only the claims judged so far.")


if __name__ == "__main__":
    main()