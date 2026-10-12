"""
repo_summary.py
Run AFTER citation_validator.py.
Reads per_repo_breakdown.csv and pivots it into REPO-WISE summary stats -
one row per repo, showing every system's accuracy side by side, plus
the average accuracy across all systems for that repo.

Usage:
  python repo_summary.py
"""

import csv
from collections import defaultdict

INPUT_CSV = "per_repo_breakdown.csv"
OUTPUT_CSV = "repo_wise_summary.csv"


def main():
    # repo_key -> {system: {"citations": n, "passed": n}}
    data = defaultdict(dict)
    systems_seen = set()

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row["repo_key"]
            system = row["system"]
            systems_seen.add(system)
            data[repo][system] = {
                "citations": int(row["citations"]),
                "passed": int(row["passed"]),
            }

    systems = sorted(systems_seen)

    # Build header: repo_key, then per-system accuracy%, then citation counts
    header = ["repo_key"] + [f"{s}_accuracy_pct" for s in systems] + [f"{s}_citations" for s in systems]

    rows = []
    print(f"{'Repo':<32}" + "".join(f"{s:<14}" for s in systems))
    print("-" * (32 + 14 * len(systems)))

    for repo_key in sorted(data.keys()):
        row = {"repo_key": repo_key}
        line = f"{repo_key:<32}"

        for system in systems:
            entry = data[repo_key].get(system)
            if entry and entry["citations"] > 0:
                acc = round(100 * entry["passed"] / entry["citations"], 1)
            elif entry:
                acc = 0.0  # had 0 citations
            else:
                acc = None  # no data at all for this system on this repo

            row[f"{system}_accuracy_pct"] = acc if acc is not None else ""
            row[f"{system}_citations"] = entry["citations"] if entry else ""
            line += f"{(str(acc) + '%') if acc is not None else '-':<14}"

        print(line)

        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()