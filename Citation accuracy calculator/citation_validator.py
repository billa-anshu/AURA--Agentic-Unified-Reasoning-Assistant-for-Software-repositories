"""
citation_validator.py
Scans every system folder under Model_outputs/, matches each output file (NN_slug.txt) to its
repo's metadata index, validates every citation, and prints/saves the comparison table.

Only the repos in SELECTED are scored, so every system is compared on the SAME repositories.
Override from PowerShell:  $env:AURA_REPOS="01_...,04_..."

Usage:
  python citation_validator.py
"""

import re
import json
import os
import csv

CITE_RE = re.compile(
    r"\[cite:file=(?P<file>[^;]+);symbol=(?P<symbol>[^;]+);lines=(?P<lines>\d+-\d+)\]"
)

BASELINE_DIR = "Model_outputs"
ROOT_INDEX_PATH = "metadata_index.json"
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
SELECTED = {k.strip() for k in _env.split(",") if k.strip()} or set(DEFAULT_SELECTED)


def normalize_path(path):
    return path.strip().replace(os.sep, "/").replace("\\", "/")


def extract_citations(text):
    citations = []
    for m in CITE_RE.finditer(text):
        start, end = m["lines"].split("-")
        citations.append({
            "file": normalize_path(m["file"]),
            "symbol": m["symbol"].strip(),
            "line_start": int(start),
            "line_end": int(end),
        })
    return citations


def validate_citation(citation, index, tolerance=3):
    key = f"{citation['file']}::{citation['symbol']}"
    entry = index.get(key)
    if not entry:
        return "FAIL_NOT_FOUND"
    cited_range = set(range(citation["line_start"], citation["line_end"] + 1))
    actual_range = set(range(entry["line_start"] - tolerance, entry["line_end"] + tolerance + 1))
    if cited_range & actual_range:
        return "PASS"
    return "FAIL_LINE_MISMATCH"


def load_repo_index(key, root_manifest):
    entry = root_manifest.get(key)
    if not entry or entry.get("skipped"):
        return None
    index_path = entry.get("index_path")
    if not index_path or not os.path.exists(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    with open(ROOT_INDEX_PATH, "r", encoding="utf-8") as f:
        root_manifest = json.load(f)

    if not os.path.isdir(BASELINE_DIR):
        print(f"No {BASELINE_DIR}/ folder found.")
        return

    print(f"Scoring {len(SELECTED)} selected repos: {sorted(SELECTED)}\n")

    system_totals = {}
    per_repo_rows = []

    for system_name in sorted(os.listdir(BASELINE_DIR)):
        system_path = os.path.join(BASELINE_DIR, system_name)
        if not os.path.isdir(system_path):
            continue
        if system_name in EXCLUDE_SYSTEMS:
            print(f"[{system_name}] excluded from totals (see EXCLUDE_SYSTEMS)")
            continue

        system_totals[system_name] = {"citations": 0, "passed": 0, "repos_covered": 0}

        for fname in sorted(os.listdir(system_path)):
            if not fname.endswith(".txt"):
                continue
            key = fname[:-4]
            if key not in SELECTED:
                continue

            index = load_repo_index(key, root_manifest)
            if index is None:
                print(f"[{system_name}][{fname}] no matching metadata index found - skipping")
                continue

            with open(os.path.join(system_path, fname), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            citations = extract_citations(text)
            passed = sum(1 for c in citations if validate_citation(c, index) == "PASS")

            system_totals[system_name]["citations"] += len(citations)
            system_totals[system_name]["passed"] += passed
            system_totals[system_name]["repos_covered"] += 1

            per_repo_rows.append({
                "system": system_name, "repo_key": key,
                "citations": len(citations), "passed": passed,
            })

    print(f"\n{'System':<15}{'Repos':<10}{'Citations':<12}{'Passed':<10}{'Accuracy %':<12}")
    print("-" * 59)
    summary_rows = []
    for system, t in system_totals.items():
        acc = round(100 * t["passed"] / t["citations"], 1) if t["citations"] else 0.0
        print(f"{system:<15}{t['repos_covered']:<10}{t['citations']:<12}{t['passed']:<10}{acc:<12}")
        summary_rows.append({
            "system": system, "repos_covered": t["repos_covered"],
            "total_citations": t["citations"], "total_passed": t["passed"],
            "overall_accuracy_pct": acc,
        })

    with open("final_accuracy_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "repos_covered", "total_citations",
                                               "total_passed", "overall_accuracy_pct"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open("per_repo_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "repo_key", "citations", "passed"])
        writer.writeheader()
        writer.writerows(per_repo_rows)

    print("\nSaved -> final_accuracy_summary.csv")
    print("Saved -> per_repo_breakdown.csv")


if __name__ == "__main__":
    main()