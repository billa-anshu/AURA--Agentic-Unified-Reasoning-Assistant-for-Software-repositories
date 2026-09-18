"""
citation_validator.py
STAGE 4 of 4.
Scans EVERY folder under baseline_outputs/ (whether filled by the automated
Groq script or by manual copy-paste from ChatGPT/Claude/Gemini/DeepSeek),
matches each output file to its repo's metadata index via filename
(NN_slug.txt), validates every citation, and prints/saves the final
comparison table - per system, summed across all repos.

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
METADATA_DIR = "metadata"
ROOT_INDEX_PATH = "metadata_index.json"

# Folders under baseline_outputs/ to skip entirely when computing totals.
# Gemini consistently produced 0 valid citations across repeated prompting
# attempts (see project notes) - excluded from the ranked comparison here.
# Its raw output files are left untouched on disk if you want to reference
# the 0% finding separately in your report.
EXCLUDE_SYSTEMS = {"gemini"}


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
    """key = 'NN_slug' matching a prompt/output filename (without .txt)"""
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
        print(f"No {BASELINE_DIR}/ folder found yet - run batch_pipeline.py first,")
        print("or manually create system folders and paste outputs into them.")
        return

    system_totals = {}   # system_name -> {"citations": 0, "passed": 0, "repos_covered": set}
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
            key = fname.replace(".txt", "")  # e.g. "01_MajorProjectFirstDraft"

            index = load_repo_index(key, root_manifest)
            if index is None:
                print(f"[{system_name}][{fname}] no matching metadata index found - skipping")
                continue

            with open(os.path.join(system_path, fname), "r", encoding="utf-8") as f:
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
        writer = csv.DictWriter(f, fieldnames=["system", "repos_covered", "total_citations", "total_passed", "overall_accuracy_pct"])
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