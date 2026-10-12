# AURA backend

1. Copy your `aura.py` (v6) into this folder.
2. `pip install -r requirements.txt`
3. Set env vars (PowerShell shown; in cmd use `set NAME=value`, no quotes):
     $env:GROQ_API_KEY="..."
     $env:SUPABASE_URL="https://YOUR-PROJECT.supabase.co"     # enables login-protected API
     $env:SUPABASE_ANON_KEY="..."
4. `uvicorn main:app --port 8000`

POST /api/summaries with ONE of: file (zip), files (folder), github_url [+ github_token for private].
Then GET /api/summaries/{id}, /result or /report. With Supabase on, send `Authorization: Bearer <token>`.
Optional env: AURA_API_KEY (used only when SUPABASE_URL is unset), AURA_CORS_ORIGINS,
AURA_MAX_UPLOAD_MB, AURA_DATA_DIR.
