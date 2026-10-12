# AURA frontend

1. Supabase: create a project. SQL Editor -> run `supabase_schema.sql`.
   Authentication -> Providers -> Email is on by default. For quick testing you can turn off
   "Confirm email" (Authentication -> Providers -> Email).
2. Edit `config.js`: API_URL, SUPABASE_URL, SUPABASE_ANON_KEY (Project Settings -> API).
3. Serve it (do NOT open the file directly, Supabase login needs http):
       python -m http.server 3000
   then open http://localhost:3000
4. Backend must allow this origin (default http://localhost:3000) and know your Supabase project:
       set SUPABASE_URL=https://YOUR-PROJECT.supabase.co
       set SUPABASE_ANON_KEY=YOUR-ANON-PUBLIC-KEY
       set GROQ_API_KEY=...
       uvicorn main:app --port 8000
