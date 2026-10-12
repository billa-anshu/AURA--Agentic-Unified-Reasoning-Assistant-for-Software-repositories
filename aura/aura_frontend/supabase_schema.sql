-- Run this once in Supabase: SQL Editor -> New query -> paste -> Run.
create table if not exists public.analyses (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null default auth.uid() references auth.users(id) on delete cascade,
  name          text not null,
  source        text not null check (source in ('folder','zip','github')),
  source_detail text,              -- github url or file name (never the token)
  stats         jsonb,
  result        jsonb not null,    -- full AURA result (report, claims, trace)
  created_at    timestamptz not null default now()
);

create index if not exists analyses_user_created on public.analyses (user_id, created_at desc);

alter table public.analyses enable row level security;

drop policy if exists "own select" on public.analyses;
drop policy if exists "own insert" on public.analyses;
drop policy if exists "own delete" on public.analyses;
create policy "own select" on public.analyses for select using (auth.uid() = user_id);
create policy "own insert" on public.analyses for insert with check (auth.uid() = user_id);
create policy "own delete" on public.analyses for delete using (auth.uid() = user_id);
