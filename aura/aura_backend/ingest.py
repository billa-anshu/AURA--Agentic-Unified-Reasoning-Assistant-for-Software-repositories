"""ingest.py - turn a zip / uploaded folder / GitHub link into a local directory.

Safety: zip-slip protection, symlinks dropped, size + file-count caps, git token passed through a
temporary HTTP header (never written to .git/config, never put in the URL, never logged).
"""
import base64
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

MAX_UPLOAD_BYTES = int(os.environ.get("AURA_MAX_UPLOAD_MB", "100")) * 1024 * 1024
MAX_EXTRACTED_BYTES = int(os.environ.get("AURA_MAX_EXTRACTED_MB", "300")) * 1024 * 1024
MAX_FILES = int(os.environ.get("AURA_MAX_FILES", "8000"))
CLONE_TIMEOUT = 180

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
             ".idea", ".vscode", ".next", "target", ".gradle"}


class IngestError(Exception):
    """User-facing problem with the submitted source (maps to HTTP 400)."""


def _safe_dest(base: Path, rel: str) -> Path:
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts):
        raise IngestError(f"unsafe path in upload: {rel!r}")
    dest = base.joinpath(*parts).resolve()
    if base.resolve() not in dest.parents:
        raise IngestError(f"unsafe path in upload: {rel!r}")
    return dest


def _skipped(rel: str) -> bool:
    return any(p in SKIP_DIRS for p in rel.replace("\\", "/").split("/")[:-1])


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise IngestError("the uploaded file is not a valid zip archive")
    total = count = 0
    with zf:
        for info in zf.infolist():
            if info.is_dir() or _skipped(info.filename):
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:   # symlink
                continue
            count += 1
            if count > MAX_FILES:
                raise IngestError(f"too many files (limit {MAX_FILES})")
            target = _safe_dest(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_EXTRACTED_BYTES:   # also stops zip bombs
                        raise IngestError("archive is too large when extracted")
                    out.write(chunk)


def safe_path(base: Path, rel: str) -> Path:
    if _skipped(rel):
        return None
    return _safe_dest(base, rel)


def normalize_root(dest: Path) -> Path:
    """GitHub-style zips wrap everything in one folder; descend into it."""
    root = dest
    while True:
        entries = [e for e in root.iterdir() if e.name != "__MACOSX"]
        if len(entries) == 1 and entries[0].is_dir():
            root = entries[0]
        else:
            return root


def enforce_limits(root: Path) -> None:
    total = count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath, fn)
            if p.is_symlink():
                p.unlink()
                continue
            count += 1
            total += p.stat().st_size
            if count > MAX_FILES:
                raise IngestError(f"too many files (limit {MAX_FILES})")
            if total > MAX_EXTRACTED_BYTES:
                raise IngestError("repository is too large")


_GH = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/tree/([^/?#]+))?/?(?:[?#].*)?$")
_SHORT = re.compile(r"^([\w.-]+)/([\w.-]+)$")


def parse_github_url(url: str):
    url = (url or "").strip()
    m = _GH.match(url)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = _SHORT.match(url)
    if m:
        return m.group(1), m.group(2), None
    raise IngestError("not a valid GitHub repository link (expected https://github.com/owner/repo)")


def clone_github(url: str, token: str, dest: Path) -> str:
    owner, repo, branch = parse_github_url(url)
    clean_url = f"https://github.com/{owner}/{repo}.git"
    cmd = ["git", "-c", "core.symlinks=false", "-c", "protocol.file.allow=never"]
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.extraheader=Authorization: Basic {basic}"]
    cmd += ["clone", "--depth", "1", "--single-branch", "--no-tags"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clean_url, str(dest)]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CLONE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        raise IngestError("cloning timed out")
    except FileNotFoundError:
        raise RuntimeError("git is not installed on the server")
    if proc.returncode != 0:
        err = (proc.stderr or "").replace(token or "\0", "***").lower()
        if any(s in err for s in ("not found", "authentication", "could not read", "403", "401")):
            raise IngestError("repository not found, or it is private and the token is missing "
                              "or lacks read access (needs 'repo' / Contents: read)")
        raise IngestError("git clone failed")
    shutil.rmtree(dest / ".git", ignore_errors=True)
    enforce_limits(dest)
    return f"{owner}/{repo}"
