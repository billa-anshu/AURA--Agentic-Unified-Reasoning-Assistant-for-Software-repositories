"""main.py - FastAPI backend for AURA.

POST /api/summaries     multipart form, send exactly ONE of:
                          file=<repo.zip>
                          files=<many files>   (folder upload; filename = relative path)
                          github_url=<https://github.com/owner/repo>  [+ github_token for private]
                        -> 202 {"job_id": ...}
GET  /api/summaries/{id}          -> {"status": queued|running|done|failed, "stage": 0-4, ...}
GET  /api/summaries/{id}/result   -> full JSON (report_markdown, claims, stats, agent_trace)
GET  /api/summaries/{id}/report   -> the report as text/markdown

Auth:
  SUPABASE_URL + SUPABASE_ANON_KEY set -> every call needs "Authorization: Bearer <supabase access token>"
                                          and jobs are visible only to the user who created them.
  otherwise, if AURA_API_KEY is set     -> send header X-API-Key.
  otherwise                              -> open (local testing only).

Run:  uvicorn main:app --port 8000
Env:  GROQ_API_KEY (required), SUPABASE_URL, SUPABASE_ANON_KEY, AURA_API_KEY, AURA_CORS_ORIGINS
"""
import json
import os
import shutil
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

import aura
import ingest
from service import NoCodeError, run_aura

DATA_DIR = Path(os.environ.get("AURA_DATA_DIR", "aura_jobs"))
DATA_DIR.mkdir(exist_ok=True)
API_KEY = os.environ.get("AURA_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
RESULT_TTL = 24 * 3600

app = FastAPI(title="AURA backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get(
        "AURA_CORS_ORIGINS", "*").split(",") if o],
    allow_methods=["*"], allow_headers=["*"],
)
pool = ThreadPoolExecutor(max_workers=3)
jobs, jobs_lock = {}, threading.Lock()
_tok_cache = {}


def current_user(authorization: str = Header(default=""), x_api_key: str = Header(default="")) -> str:
    if SUPABASE_URL:
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if not token:
            raise HTTPException(401, "login required")
        now = time.time()
        hit = _tok_cache.get(token)
        if hit and hit[1] > now:
            return hit[0]
        req = urllib.request.Request(f"{SUPABASE_URL}/auth/v1/user",
                                     headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                uid = json.load(r).get("id")
        except urllib.error.HTTPError:
            raise HTTPException(401, "invalid or expired session, please sign in again")
        except Exception:
            raise HTTPException(503, "auth service unreachable")
        if not uid:
            raise HTTPException(401, "invalid session")
        if len(_tok_cache) > 1000:
            _tok_cache.clear()
        _tok_cache[token] = (uid, now + 60)
        return uid
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid API key")
    return "anonymous"


def _set(job_id, **kw):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(kw)


def _check_owner(job_id: str, user: str):
    if not job_id.isalnum():
        raise HTTPException(404, "unknown job")
    owner_file = DATA_DIR / job_id / "owner.txt"
    if not owner_file.exists() or owner_file.read_text().strip() != user:
        raise HTTPException(404, "unknown job")      # same answer for "not yours" and "not found"


def _cleanup_old():
    now = time.time()
    for d in DATA_DIR.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > RESULT_TTL:
            shutil.rmtree(d, ignore_errors=True)


@app.on_event("startup")
def _startup():
    if not aura.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    if SUPABASE_URL and not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL is set but SUPABASE_ANON_KEY is missing")
    _cleanup_old()


async def _save_stream(upload: UploadFile, path: Path, budget: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as out:
        while chunk := await upload.read(1 << 20):
            budget["left"] -= len(chunk)
            if budget["left"] < 0:
                raise ingest.IngestError("upload is too large")
            out.write(chunk)


def _worker(job_id, kind, workdir: Path, github_url=None, token=None):
    src = workdir / "src"
    try:
        _set(job_id, status="running", stage=0)
        name = "repository"
        if kind == "github":
            name = ingest.clone_github(github_url, token, src)
            root = src
        elif kind == "zip":
            ingest.extract_zip(workdir / "upload.zip", src)
            root = ingest.normalize_root(src)
            name = root.name if root != src else "uploaded-project"
        else:
            ingest.enforce_limits(src)
            root = ingest.normalize_root(src)
            name = root.name if root != src else "uploaded-folder"
        result = run_aura(f"{job_id}_{name}", root, workdir, name,
                          on_stage=lambda n: _set(job_id, stage=n))
        (workdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        _set(job_id, status="done", name=name, stage=4, stats=result["stats"])
    except ingest.IngestError as e:
        _set(job_id, status="failed", error=str(e), code=400)
    except NoCodeError as e:
        _set(job_id, status="failed", error=str(e), code=422)
    except aura.DailyQuotaExhausted:
        _set(job_id, status="failed", error="LLM daily quota exhausted, try again later", code=429)
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="failed", error=f"internal error: {type(e).__name__}: {str(e)[:300]}", code=500)
    finally:
        shutil.rmtree(src, ignore_errors=True)       # never keep user code on disk
        (workdir / "upload.zip").unlink(missing_ok=True)


@app.post("/api/summaries", status_code=202)
async def create_summary(
    user: str = Depends(current_user),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    github_url: Optional[str] = Form(None),
    github_token: Optional[str] = Form(None),
    x_github_token: Optional[str] = Header(default=None),
):
    token = github_token or x_github_token
    given = [bool(file), bool(files), bool(github_url)]
    if sum(given) != 1:
        raise HTTPException(400, "send exactly one of: file (zip), files (folder), github_url")

    job_id = uuid.uuid4().hex[:12]
    workdir = DATA_DIR / job_id
    workdir.mkdir(parents=True)
    (workdir / "owner.txt").write_text(user)
    budget = {"left": ingest.MAX_UPLOAD_BYTES}
    try:
        if github_url:
            ingest.parse_github_url(github_url)          # fail fast on bad links
            kind = "github"
        elif file:
            await _save_stream(file, workdir / "upload.zip", budget)
            kind = "zip"
        else:
            kind = "folder"
            for f in files:
                dest = ingest.safe_path(workdir / "src", f.filename or "")
                if dest is not None:
                    await _save_stream(f, dest, budget)
            if not (workdir / "src").exists():
                raise ingest.IngestError("no usable files in upload")
    except ingest.IngestError as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, str(e))

    _set(job_id, status="queued", stage=0, kind=kind, created=time.time())
    pool.submit(_worker, job_id, kind, workdir, github_url, token)
    return {"job_id": job_id, "status_url": f"/api/summaries/{job_id}"}


def _load_result(job_id):
    p = DATA_DIR / job_id / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/summaries/{job_id}")
def get_status(job_id: str, user: str = Depends(current_user)):
    _check_owner(job_id, user)
    with jobs_lock:
        job = dict(jobs.get(job_id, {}))
    if not job:
        if _load_result(job_id):
            return {"status": "done", "stage": 4}
        return {"status": "failed", "error": "job lost (server restarted), please re-run"}
    return job


@app.get("/api/summaries/{job_id}/result")
def get_result(job_id: str, user: str = Depends(current_user)):
    _check_owner(job_id, user)
    res = _load_result(job_id)
    if res is None:
        raise HTTPException(404, "result not ready")
    return JSONResponse(res)


@app.get("/api/summaries/{job_id}/report", response_class=PlainTextResponse)
def get_report(job_id: str, user: str = Depends(current_user)):
    _check_owner(job_id, user)
    res = _load_result(job_id)
    if res is None:
        raise HTTPException(404, "result not ready")
    return PlainTextResponse(res["report_markdown"], media_type="text/markdown")
