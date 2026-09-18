"""
clone_repositories.py
STAGE 1 of 4.
Reads repos.txt, clones each into repositories/<slug>/, and writes
repositories_manifest.json (NN -> slug -> url) so every later stage
uses the same consistent numbering.

Usage:
  python clone_repositories.py
"""

import os
import re
import json
import shutil
import subprocess
import urllib.request

REPOS_FILE = "repos.txt"
REPO_DIR = "repositories"
MANIFEST_PATH = "repositories_manifest.json"
CLONE_SIZE_LIMIT_MB = 150


def repo_slug(url):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", url.rstrip("/").split("/")[-1])


def get_repo_size_mb(owner_repo):
    try:
        api_url = f"https://api.github.com/repos/{owner_repo}"
        with urllib.request.urlopen(api_url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("size", 0) / 1024
    except Exception:
        return 0


def main():
    os.makedirs(REPO_DIR, exist_ok=True)

    with open(REPOS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    for i, url in enumerate(urls, start=1):
        num = f"{i:02d}"
        slug = repo_slug(url)
        key = f"{num}_{slug}"
        dest = os.path.join(REPO_DIR, slug)

        if key in manifest and os.path.isdir(dest):
            print(f"[{key}] already cloned, skipping")
            continue

        owner_repo = "/".join(url.rstrip("/").split("/")[-2:])
        size_mb = get_repo_size_mb(owner_repo)
        if size_mb > CLONE_SIZE_LIMIT_MB:
            print(f"[{key}] SKIPPED - {size_mb:.0f}MB, over {CLONE_SIZE_LIMIT_MB}MB limit")
            manifest[key] = {"url": url, "slug": slug, "skipped": True, "reason": f"too large ({size_mb:.0f}MB)"}
            continue

        print(f"[{key}] cloning {url}...")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, dest],
                check=True, capture_output=True, timeout=120,
            )
            manifest[key] = {"url": url, "slug": slug, "path": dest}
            print(f"[{key}] done")
        except Exception as e:
            print(f"[{key}] CLONE FAILED: {e}")
            manifest[key] = {"url": url, "slug": slug, "error": str(e)}

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} repos in manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()