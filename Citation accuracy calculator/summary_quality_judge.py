"""
summary_quality_judge.py  (v5: neutral context WITH README/dependencies, same for EVERY system)

What changed vs v4
  * The judge digest now starts with the repo's README + dependency file (package.json,
    requirements.txt, pom.xml ...), then the file tree and the largest symbol per file. Before,
    correct statements about the tech stack or how to run the project looked "invented".
  * The "## Sources" section (footnote tag list) is cut off before judging; no other system has it.
  * Only the repos in SELECTED are judged/reported, so all systems are compared on the SAME repos.
  * Changing JUDGE_VERSION or the judge model re-scores all selected pairs automatically.

Usage:
    python summary_quality_judge.py                  # fill missing pairs (auto re-scores on version change)
    python summary_quality_judge.py --rejudge all
    python summary_quality_judge.py --rejudge AURA

Pick a different judge model on Groq (separate daily quota per model), in .env:
    AURA_JUDGE_GROQ_MODEL=openai/gpt-oss-120b
    AURA_JUDGE_REPEATS=1
Or any other OpenAI-compatible provider, in .env:
    AURA_BASE_URL=...  AURA_API_KEY=...  AURA_JUDGE_MODEL=...  AURA_JUDGE_SLEEP=3
Subset of repos (PowerShell):  $env:AURA_REPOS="01_...,04_..."
"""

import os
import re
import csv
import json
import time
import shutil
import argparse
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq_runner import call_groq_model, DailyQuotaExhausted

MODEL_OUTPUTS_DIR = "Model_outputs"
PROMPTS_DIR = "prompts"
REPOSITORIES_DIR = "repositories"
ROOT_INDEX_PATH = "metadata_index.json"
RAW_JSON_PATH = "summary_quality_raw.json"
JUDGE_VERSION_PATH = "summary_quality_judge_version.txt"
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
_env_repos = os.environ.get("AURA_REPOS", "")
SELECTED = {k.strip() for k in _env_repos.split(",") if k.strip()} or set(DEFAULT_SELECTED)

JUDGE_MODEL = "gpt-oss-20b"      # used via groq_runner unless the AURA_* settings are set
ALT_BASE_URL = os.environ.get("AURA_BASE_URL")
ALT_API_KEY = os.environ.get("AURA_API_KEY")
ALT_JUDGE_MODEL = os.environ.get("AURA_JUDGE_MODEL")
ALT_SLEEP = float(os.environ.get("AURA_JUDGE_SLEEP", "8"))
JUDGE_REPEATS = int(os.environ.get("AURA_JUDGE_REPEATS", "2"))
_alt_client = None

DIMENSIONS = ["completeness", "accuracy", "clarity", "depth", "structure"]

# Bump to force ALL systems to be re-scored on the next run.
JUDGE_VERSION = "v5-readme-context"

REPO_CONTEXT_CHAR_LIMIT = 11000
PER_FILE_CHARS = 350
SUMMARY_CHAR_LIMIT = 7000

JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator scoring a repository summary
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


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_JUDGE_MODEL = os.environ.get("AURA_JUDGE_GROQ_MODEL")


def _judge_target():
    """(base_url, api_key, model) of the judge, or None = old groq_runner default."""
    if ALT_BASE_URL and ALT_API_KEY and ALT_JUDGE_MODEL:
        return ALT_BASE_URL, ALT_API_KEY, ALT_JUDGE_MODEL
    if GROQ_JUDGE_MODEL and os.environ.get("GROQ_API_KEY"):
        return GROQ_URL, os.environ["GROQ_API_KEY"], GROQ_JUDGE_MODEL
    return None


def alt_enabled():
    return _judge_target() is not None


def judge_model_name():
    t = _judge_target()
    return t[2] if t else JUDGE_MODEL


def judge_identity():
    return f"{JUDGE_VERSION}|{judge_model_name()}|x{JUDGE_REPEATS}"


def call_judge_model(prompt):
    global _alt_client
    target = _judge_target()
    if target is None:
        return call_groq_model(JUDGE_MODEL, prompt)

    base_url, api_key, model = target
    from openai import OpenAI, RateLimitError, APIStatusError
    if _alt_client is None:
        _alt_client = OpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=0)

    for attempt in range(5):
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
        )
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
            print(f"    ...rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (400, 401, 403, 404):
                print(f"    Judge model error ({e.status_code}) for '{model}': {e}")
                raise
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"    ...network error ({e}), retrying")
            time.sleep(5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Context + parsing helpers
# ---------------------------------------------------------------------------

def normalize_for_judge(text):
    text = text.split("## Sources")[0]          # footnote tag list: not prose, no baseline has it
    text = CITE_STRIP_RE.sub("", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_repo_code_only(full_prompt_text):
    """Fallback context: the code section of the original prompt file."""
    match = re.search(
        r"=== REPOSITORY CONTENT ===\n(.*?)\n=== END REPOSITORY CONTENT ===",
        full_prompt_text, re.DOTALL,
    )
    return match.group(1) if match else full_prompt_text


def read_project_background(repo_root):
    """README + dependency file, identical for every system."""
    parts = []
    try:
        for name in sorted(os.listdir(repo_root)):
            if name.lower().startswith("readme"):
                with open(os.path.join(repo_root, name), "r", encoding="utf-8", errors="ignore") as fh:
                    parts.append("README:\n" + fh.read()[:1200])
                break
    except Exception:
        pass
    for name in ("requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml"):
        p = os.path.join(repo_root, name)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    parts.append(f"{name}:\n" + fh.read()[:600])
            except Exception:
                pass
            break
    return "\n\n".join(parts)


def build_neutral_context(index, repo_root):
    """Same digest for every system: README + dependencies + file tree + the largest symbol of
    each file. Built only from the ground-truth repo, never from any system's output."""
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
            parts.append(f"# {f} :: {sym}\n{code[:PER_FILE_CHARS]}")
        except Exception:
            continue
    background = read_project_background(repo_root)
    return (background + "\n\n" + "\n\n".join(parts)).strip()[:REPO_CONTEXT_CHAR_LIMIT]


def load_repo_context(repo_key, root_manifest):
    """Neutral digest if the index + repo exist, else fall back to the prompt excerpt."""
    entry = root_manifest.get(repo_key) or {}
    index_path = entry.get("index_path")
    slug = entry.get("slug")
    if index_path and slug and os.path.exists(index_path):
        repo_root = os.path.join(REPOSITORIES_DIR, slug)
        if os.path.isdir(repo_root):
            with open(index_path, "r", encoding="utf-8") as f:
                ctx = build_neutral_context(json.load(f), repo_root)
            if ctx.strip():
                return ctx

    prompt_path = os.path.join(PROMPTS_DIR, f"{repo_key}.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return extract_repo_code_only(f.read())[:REPO_CONTEXT_CHAR_LIMIT]
    return None


def extract_json(text):
    if text is None:
        return None
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _valid(scores):
    if not isinstance(scores, dict):
        return None
    out = {}
    for d in DIMENSIONS:
        try:
            v = float(scores[d])
        except (KeyError, TypeError, ValueError):
            return None
        out[d] = min(5.0, max(1.0, v))
    return out


def judge_one(repo_context, summary_text):
    clean = normalize_for_judge(summary_text)
    if not clean:
        return None, "empty_after_stripping_tags"
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        repo_context=repo_context[:REPO_CONTEXT_CHAR_LIMIT],
        summary=clean[:SUMMARY_CHAR_LIMIT],
    )
    runs = []
    for _ in range(max(1, JUDGE_REPEATS)):
        raw = call_judge_model(prompt)
        scores = _valid(extract_json(raw))
        if scores:
            runs.append(scores)
    if not runs:
        return None, "call_failed_or_unparseable"
    avg = {d: round(sum(r[d] for r in runs) / len(runs), 2) for d in DIMENSIONS}
    return avg, "ok"


# ---------------------------------------------------------------------------
# Resume / bookkeeping
# ---------------------------------------------------------------------------

def load_existing_raw():
    if not os.path.exists(RAW_JSON_PATH):
        print(f"No existing {RAW_JSON_PATH} - starting fresh.")
        return {}
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def count_scored(raw):
    return sum(1 for repos in raw.values() for k, s in repos.items()
               if s is not None and k in SELECTED)


def backup_file(path):
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak")


def find_missing_pairs(raw):
    missing = []
    if not os.path.isdir(MODEL_OUTPUTS_DIR):
        print(f"No {MODEL_OUTPUTS_DIR}/ folder found.")
        return missing
    for system in sorted(os.listdir(MODEL_OUTPUTS_DIR)):
        spath = os.path.join(MODEL_OUTPUTS_DIR, system)
        if not os.path.isdir(spath) or system in EXCLUDE_SYSTEMS:
            continue
        for fname in sorted(os.listdir(spath)):
            if fname.endswith(".txt") and fname[:-4] in SELECTED:
                key = fname[:-4]
                if raw.get(system, {}).get(key) is None:
                    missing.append((system, key))
    return missing


def apply_rejudge(raw, arg):
    """Clear scores for the selected repos only (scores of other repos are left untouched)."""
    wanted = {s.strip().lower() for s in arg.split(",") if s.strip()}
    everything = "all" in wanted
    cleared = 0
    for system, repos in raw.items():
        if everything or system.lower() in wanted:
            for k in repos:
                if k in SELECTED and repos[k] is not None:
                    repos[k] = None
                    cleared += 1
    print(f"--rejudge: cleared {cleared} scores in memory (disk unchanged until save).")
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejudge", default="", help="'all' or comma-separated system names")
    args = parser.parse_args()

    if not os.path.exists(ROOT_INDEX_PATH):
        print(f"ERROR: {ROOT_INDEX_PATH} not found.")
        return
    with open(ROOT_INDEX_PATH, "r", encoding="utf-8") as f:
        root_manifest = json.load(f)

    print(f"Judging {len(SELECTED)} selected repos: {sorted(SELECTED)}")

    raw_results = load_existing_raw()
    saved_version = ""
    if os.path.exists(JUDGE_VERSION_PATH):
        with open(JUDGE_VERSION_PATH, "r", encoding="utf-8") as f:
            saved_version = f.read().strip()

    if args.rejudge:
        raw_results = apply_rejudge(raw_results, args.rejudge)
    elif saved_version != judge_identity():
        print(f"Judge rules/model changed ('{saved_version or 'none'}' -> '{judge_identity()}'): "
              f"re-scoring ALL systems so everyone is judged by the same judge.")
        raw_results = apply_rejudge(raw_results, "all")

    scored_before = count_scored(raw_results)
    missing = find_missing_pairs(raw_results)
    if not missing:
        print("Nothing to do - all selected pairs already scored.")
        return

    print(f"{len(missing)} pairs to judge (judge: "
          f"{judge_model_name()}, repeats: {JUDGE_REPEATS})")

    fail_counts = defaultdict(int)
    stopped_early = False
    ctx_cache = {}

    for system, repo_key in missing:
        raw_results.setdefault(system, {})
        spath = os.path.join(MODEL_OUTPUTS_DIR, system, f"{repo_key}.txt")
        with open(spath, "r", encoding="utf-8", errors="ignore") as f:
            summary = f.read().strip()
        if not summary:
            print(f"[{system}][{repo_key}] empty file - skipping")
            raw_results[system][repo_key] = None
            continue

        if repo_key not in ctx_cache:
            ctx_cache[repo_key] = load_repo_context(repo_key, root_manifest)
        context = ctx_cache[repo_key]
        if not context:
            print(f"[{system}][{repo_key}] no repo context available - skipping")
            continue

        print(f"[{system}][{repo_key}] judging...")
        try:
            scores, status = judge_one(context, summary)
        except DailyQuotaExhausted:
            print("\nSTOPPING: daily token quota exhausted. Scores so far will be saved.")
            print("Re-run WITHOUT --rejudge after the quota resets to continue.\n")
            stopped_early = True
            break

        if scores is None:
            print(f"[{system}][{repo_key}] {status} - skipping")
            raw_results[system][repo_key] = None
            fail_counts[status] += 1
            continue
        raw_results[system][repo_key] = scores
        print(f"[{system}][{repo_key}] -> {scores}")

    scored_after = count_scored(raw_results)
    if stopped_early and scored_after == scored_before:
        print("\nNo new scores were produced this run, so nothing was saved or changed.")
        print("Switch the judge model (see AURA_JUDGE_GROQ_MODEL) or wait for the quota reset.")
        return
    if scored_after < scored_before:
        print(f"\nREFUSING TO SAVE: would leave {scored_after} scored entries vs {scored_before} before.")
        print("Fix the underlying issue (rate limits / keys) and re-run. Files untouched.")
        return

    backup_file(RAW_JSON_PATH)
    backup_file("summary_quality_per_repo.csv")
    backup_file("summary_quality_overall.csv")

    per_repo_rows = []
    for system, repos in raw_results.items():
        if system in EXCLUDE_SYSTEMS:
            continue
        for repo_key, s in repos.items():
            if s is None or repo_key not in SELECTED:
                continue
            avg = round(sum(s.get(d, 0) for d in DIMENSIONS) / len(DIMENSIONS), 2)
            per_repo_rows.append({"system": system, "repo_key": repo_key,
                                  **{d: s.get(d, "") for d in DIMENSIONS}, "average": avg})

    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)
    with open(JUDGE_VERSION_PATH, "w", encoding="utf-8") as f:
        f.write(judge_identity())

    with open("summary_quality_per_repo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system", "repo_key"] + DIMENSIONS + ["average"])
        w.writeheader()
        w.writerows(sorted(per_repo_rows, key=lambda r: (r["system"], r["repo_key"])))

    systems_seen = sorted({r["system"] for r in per_repo_rows})
    pivot = defaultdict(dict)
    for r in per_repo_rows:
        pivot[r["repo_key"]][r["system"]] = r["average"]

    print(f"\n{'=' * 70}\nPER-REPO SUMMARY QUALITY (1-5)\n{'=' * 70}")
    print(f"{'Repo':<32}" + "".join(f"{s:<14}" for s in systems_seen))
    print("-" * (32 + 14 * len(systems_seen)))
    for repo_key in sorted(pivot):
        print(f"{repo_key:<32}" + "".join(f"{pivot[repo_key].get(s, '-'):<14}" for s in systems_seen))

    totals = defaultdict(lambda: defaultdict(list))
    for r in per_repo_rows:
        for d in DIMENSIONS:
            if r[d] != "":
                totals[r["system"]][d].append(float(r[d]))

    overall_rows = []
    print(f"\n{'=' * 70}\nOVERALL SUMMARY QUALITY (average across selected repos)\n{'=' * 70}")
    print(f"{'System':<15}" + "".join(f"{d:<14}" for d in DIMENSIONS) + "Overall Avg")
    print("-" * (15 + 14 * len(DIMENSIONS) + 12))
    for system, dims in sorted(totals.items()):
        row, allv, line = {"system": system}, [], f"{system:<15}"
        for d in DIMENSIONS:
            vals = dims.get(d, [])
            avg = round(sum(vals) / len(vals), 2) if vals else 0.0
            row[d] = avg
            allv.extend(vals)
            line += f"{avg:<14}"
        row["overall_average"] = round(sum(allv) / len(allv), 2) if allv else 0.0
        print(line + f"{row['overall_average']}")
        overall_rows.append(row)

    with open("summary_quality_overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system"] + DIMENSIONS + ["overall_average"])
        w.writeheader()
        w.writerows(overall_rows)

    print("\nSaved -> summary_quality_raw.json / _per_repo.csv / _overall.csv")
    if fail_counts:
        print(f"Failures this run: {dict(fail_counts)} (re-run WITHOUT --rejudge to retry them)")
    if stopped_early:
        print("Run stopped early (quota). Re-run WITHOUT --rejudge later to continue.")


if __name__ == "__main__":
    main()