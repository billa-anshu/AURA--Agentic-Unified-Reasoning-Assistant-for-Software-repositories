"""
summary_quality_judge_resume.py
Resume version of summary_quality_judge.py.

Only judges (system, repo_key) pairs that are currently null/missing in
summary_quality_raw.json - e.g. AURA (all skipped last run) and
deepseek's later repos (which stopped after hitting failures). Every
pair that already has real scores is left untouched and just carried
forward into the merged output.

Usage:
    python summary_quality_judge_resume.py

Requires (same as original script):
    - Model_outputs/<system>/<repo_key>.txt   (the summaries)
    - prompts/<repo_key>.txt                  (the source prompt w/ repo code)
    - groq_runner.py                          (with GROQ_API_KEY set in env)
    - summary_quality_raw.json                (existing results to resume from)
"""

import os
import json
import re
import csv
import shutil
import datetime
from collections import defaultdict

from groq_runner import call_groq_model, DailyQuotaExhausted

MODEL_OUTPUTS_DIR = "Model_outputs"
PROMPTS_DIR = "prompts"
RAW_JSON_PATH = "summary_quality_raw.json"
JUDGE_MODEL = "gpt-oss-20b"
DIMENSIONS = ["completeness", "accuracy", "clarity", "depth", "structure"]

REPO_CONTEXT_CHAR_LIMIT = 1500
SUMMARY_CHAR_LIMIT = 1200

JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator scoring a repository summary
for quality. You are given repository code excerpts and a summary written about it.
Score the summary on 5 dimensions, each from 1 (poor) to 5 (excellent).

=== REPOSITORY CODE EXCERPT ===
{repo_context}
=== END EXCERPT ===

=== SUMMARY TO EVALUATE ===
{summary}
=== END SUMMARY ===

Score: completeness, accuracy (claims match the code shown), clarity, depth, structure.

Respond with ONLY this JSON, nothing else:
{{"completeness": <1-5>, "accuracy": <1-5>, "clarity": <1-5>, "depth": <1-5>, "structure": <1-5>}}
"""


def extract_repo_code_only(full_prompt_text):
    match = re.search(
        r"=== REPOSITORY CONTENT ===\n(.*?)\n=== END REPOSITORY CONTENT ===",
        full_prompt_text, re.DOTALL,
    )
    return match.group(1) if match else full_prompt_text


def extract_json(text):
    if text is None:
        return None
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_one(repo_context, summary_text):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        repo_context=repo_context[:REPO_CONTEXT_CHAR_LIMIT],
        summary=summary_text[:SUMMARY_CHAR_LIMIT],
    )
    raw = call_groq_model(JUDGE_MODEL, prompt)
    if raw is None:
        return None, "call_failed_or_too_large"
    scores = extract_json(raw)
    if scores is None:
        return None, "could_not_parse_json"
    return scores, "ok"


def load_existing_raw():
    if not os.path.exists(RAW_JSON_PATH):
        print(f"No existing {RAW_JSON_PATH} found - nothing to resume from, "
              f"starting fresh (everything will be treated as missing).")
        return {}
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def count_scored(raw_results):
    """How many (system, repo_key) pairs currently hold a real (non-null) score."""
    return sum(
        1
        for repos in raw_results.values()
        for scores in repos.values()
        if scores is not None
    )


def backup_file(path):
    """Copy path -> path.YYYYmmdd_HHMMSS.bak before we touch it, so a bad run
    (e.g. every call failing due to a transient rate limit) can never destroy
    previously-good data without a way back."""
    if not os.path.exists(path):
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{stamp}.bak"
    shutil.copy2(path, backup_path)
    print(f"Backed up {path} -> {backup_path}")
    return backup_path


def find_missing_pairs(raw_results):
    """Return list of (system, repo_key) where the score is currently None/null,
    for every system folder present in Model_outputs/."""
    missing = []
    if not os.path.isdir(MODEL_OUTPUTS_DIR):
        print(f"No {MODEL_OUTPUTS_DIR}/ folder found.")
        return missing

    for system_name in sorted(os.listdir(MODEL_OUTPUTS_DIR)):
        system_path = os.path.join(MODEL_OUTPUTS_DIR, system_name)
        if not os.path.isdir(system_path):
            continue

        existing_for_system = raw_results.get(system_name, {})

        for fname in sorted(os.listdir(system_path)):
            if not fname.endswith(".txt"):
                continue
            repo_key = fname.replace(".txt", "")

            current_score = existing_for_system.get(repo_key, "MISSING_KEY")
            if current_score is None or current_score == "MISSING_KEY":
                missing.append((system_name, repo_key))

    return missing


def main():
    raw_results = load_existing_raw()
    scored_before = count_scored(raw_results)
    missing_pairs = find_missing_pairs(raw_results)

    if not missing_pairs:
        print("Nothing to do - no null/missing entries found. All pairs already scored.")
        return

    print(f"Found {len(missing_pairs)} unscored (system, repo_key) pairs to judge:")
    for system_name, repo_key in missing_pairs:
        print(f"  - [{system_name}] {repo_key}")

    fail_counts = defaultdict(int)
    stopped_early = False

    for system_name, repo_key in missing_pairs:
        raw_results.setdefault(system_name, {})

        summary_path = os.path.join(MODEL_OUTPUTS_DIR, system_name, f"{repo_key}.txt")
        if not os.path.exists(summary_path):
            print(f"[{system_name}][{repo_key}] summary file missing - skipping")
            raw_results[system_name][repo_key] = None
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read().strip()

        if not summary_text:
            print(f"[{system_name}][{repo_key}] empty file - skipping")
            raw_results[system_name][repo_key] = None
            continue

        prompt_path = os.path.join(PROMPTS_DIR, f"{repo_key}.txt")
        if not os.path.exists(prompt_path):
            print(f"[{system_name}][{repo_key}] no matching prompt file - skipping")
            continue
        with open(prompt_path, "r", encoding="utf-8") as f:
            full_prompt = f.read()
        repo_context = extract_repo_code_only(full_prompt)

        print(f"[{system_name}][{repo_key}] judging...")
        try:
            scores, status = judge_one(repo_context, summary_text)
        except DailyQuotaExhausted:
            print(f"\n{'='*70}")
            print(f"STOPPING RUN: Groq's daily token quota is exhausted.")
            print(f"Everything judged so far this run has been kept and will be saved.")
            print(f"Re-run this same script again once your quota resets (Groq's TPD")
            print(f"window is rolling ~24h, so check the 'try again in ...' time Groq")
            print(f"gave, or just try again tomorrow) - it will pick up exactly where")
            print(f"this run left off, since already-scored pairs are never redone.")
            print(f"{'='*70}\n")
            stopped_early = True
            break

        if scores is None:
            print(f"[{system_name}][{repo_key}] {status} - skipping")
            raw_results[system_name][repo_key] = None
            fail_counts[status] += 1
            continue

        raw_results[system_name][repo_key] = scores
        print(f"[{system_name}][{repo_key}] -> {scores}")

    # ---------- Safety check BEFORE writing anything ----------
    scored_after = count_scored(raw_results)
    if scored_after < scored_before:
        print(f"\n{'!'*70}")
        print(f"REFUSING TO SAVE: this run would leave you with FEWER scored")
        print(f"entries ({scored_after}) than you had before ({scored_before}).")
        print(f"This usually means calls failed (e.g. a rate limit) rather than")
        print(f"the data genuinely needing to be blanked out.")
        print(f"Nothing has been overwritten. Your existing files are untouched.")
        print(f"Fix the underlying issue (check groq_runner.py / GROQ_API_KEY /")
        print(f"rate limits) and re-run.")
        print(f"{'!'*70}")
        return

    # ---------- Backups before we touch anything on disk ----------
    backup_file(RAW_JSON_PATH)
    backup_file("summary_quality_per_repo.csv")
    backup_file("summary_quality_overall.csv")

    # ---------- Rebuild per_repo_rows from the FULL merged raw_results ----------
    per_repo_rows = []
    for system_name, repos in raw_results.items():
        for repo_key, scores in repos.items():
            if scores is None:
                continue
            avg = round(sum(scores.get(d, 0) for d in DIMENSIONS) / len(DIMENSIONS), 2)
            per_repo_rows.append({
                "system": system_name, "repo_key": repo_key,
                **{d: scores.get(d, "") for d in DIMENSIONS},
                "average": avg,
            })

    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    with open("summary_quality_per_repo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "repo_key"] + DIMENSIONS + ["average"])
        writer.writeheader()
        writer.writerows(sorted(per_repo_rows, key=lambda r: (r["system"], r["repo_key"])))

    # ---------- PRINT 1: PER-REPO TABLE (pivoted - repo x system) ----------
    systems_seen = sorted(set(r["system"] for r in per_repo_rows))
    repo_pivot = defaultdict(dict)
    for row in per_repo_rows:
        repo_pivot[row["repo_key"]][row["system"]] = row["average"]

    print(f"\n{'='*70}\nPER-REPO SUMMARY QUALITY (average score per system, 1-5)\n{'='*70}")
    print(f"{'Repo':<32}" + "".join(f"{s:<14}" for s in systems_seen))
    print("-" * (32 + 14 * len(systems_seen)))
    for repo_key in sorted(repo_pivot.keys()):
        line = f"{repo_key:<32}"
        for system in systems_seen:
            val = repo_pivot[repo_key].get(system, "-")
            line += f"{val:<14}"
        print(line)

    system_dim_totals = defaultdict(lambda: defaultdict(list))
    for row in per_repo_rows:
        for d in DIMENSIONS:
            if row[d] != "":
                system_dim_totals[row["system"]][d].append(row[d])

    overall_rows = []
    print(f"\n{'='*70}\nOVERALL SUMMARY QUALITY (average across all repos, per dimension)\n{'='*70}")
    print(f"{'System':<15}" + "".join(f"{d:<14}" for d in DIMENSIONS) + f"{'Overall Avg':<12}")
    print("-" * (15 + 14 * len(DIMENSIONS) + 12))
    for system, dims in system_dim_totals.items():
        row = {"system": system}
        all_scores = []
        line = f"{system:<15}"
        for d in DIMENSIONS:
            vals = dims.get(d, [])
            avg = round(sum(vals) / len(vals), 2) if vals else 0.0
            row[d] = avg
            all_scores.extend(vals)
            line += f"{avg:<14}"
        overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
        row["overall_average"] = overall
        line += f"{overall:<12}"
        print(line)
        overall_rows.append(row)

    with open("summary_quality_overall.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system"] + DIMENSIONS + ["overall_average"])
        writer.writeheader()
        writer.writerows(overall_rows)

    print("\nSaved -> summary_quality_raw.json (merged)")
    print("Saved -> summary_quality_per_repo.csv (merged)")
    print("Saved -> summary_quality_overall.csv (merged)")
    if fail_counts:
        print(f"\nFailures this run: {dict(fail_counts)}")
    if stopped_early:
        print("\nNote: this run stopped early due to the daily quota limit - "
              "just re-run the script later to continue with the rest.")


if __name__ == "__main__":
    main()