"""
groq_runner.py (fixed for free-tier rate limits + Windows encoding)
Free tier: 8000 tokens/minute PER MODEL, shared across your whole org -
running things in parallel makes this worse, not better. This version
throttles calls and keeps completions small so requests actually fit.

BUGFIX (see call_groq_model): the previous version treated ANY
rate_limit_exceeded error as a permanent "prompt too large" failure and
gave up instantly with no retry. That's wrong - Groq returns
rate_limit_exceeded for the ordinary per-minute TPM cap too, which is
temporary and just needs a longer wait. Only a message that explicitly
says the request itself exceeds the model's limit (no amount of
waiting will fix it) should be treated as permanently too large.
"""

import os
import re
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
)

GROQ_MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen3.6-27b": "qwen/qwen3.6-27b",
}

# Free tier cap is 8000 TPM per model. Keep completion small so
# prompt_tokens + completion_tokens has room to fit under that ceiling.
MAX_COMPLETION_TOKENS = 900

# Pause between EVERY call (success or fail) - spreads token usage out
# over time instead of bursting, which is what triggers 429s. Bumped up
# from 8s: 8s was too tight once you're mid-run and close to the TPM
# ceiling, which is what caused the false "too large" cascade.
SECONDS_BETWEEN_CALLS = 12


def _is_permanently_too_large(msg):
    """True only when THIS SINGLE request exceeds what the model can ever
    accept in one go - no retry or wait will fix that, so the caller should
    shrink the prompt instead. False for an ordinary temporary rate-limit
    (TPM/RPM) hit, which DOES deserve a retry with backoff."""
    if "request too large" in msg.lower():
        return True
    return False


class DailyQuotaExhausted(Exception):
    """Raised when Groq reports the per-DAY token budget is used up.
    This is fundamentally different from a per-minute rate limit: retrying
    the same call a few seconds later won't help (the daily bucket won't
    have refilled), and every other repo in the run will hit the exact same
    wall seconds after this one does. The caller should stop the whole run,
    not just skip this pair and move to the next."""
    pass


def _is_daily_quota_exhausted(msg):
    # Groq's message explicitly names the limit type, e.g.:
    # "...on tokens per day (TPD): Limit 200000, Used 199947, Requested 1064..."
    return "tokens per day" in msg.lower() or "(tpd)" in msg.lower()


def call_groq_model(model_key, prompt, max_retries=5):
    model_name = GROQ_MODELS[model_key]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.3,
            )
            time.sleep(SECONDS_BETWEEN_CALLS)
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)

            if _is_daily_quota_exhausted(msg):
                # Don't retry, don't waste more calls on later repos - the
                # whole day's budget for this model is gone. Let the caller
                # decide what to do (stop the run, save progress).
                print(f"  [{model_key}] DAILY token quota exhausted - aborting run for today")
                raise DailyQuotaExhausted(msg)

            if _is_permanently_too_large(msg):
                print(f"  [{model_key}] prompt too large for this model's TPM cap - skipping retries")
                return None

            # Try to honor Groq's suggested wait time if it gives one
            # (e.g. "Please try again in 42.5s"), otherwise fall back to
            # a longer exponential-ish backoff than before.
            suggested = re.search(r"try again in ([\d.]+)s", msg, re.IGNORECASE)
            if suggested:
                wait = float(suggested.group(1)) + 2  # small buffer
            else:
                wait = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s, 100s

            print(f"  [{model_key}] attempt {attempt+1} failed: {e} - retrying in {wait:.0f}s")
            time.sleep(wait)
    print(f"  [{model_key}] gave up after {max_retries} attempts (temporary rate limit never cleared)")
    return None


def call_all_models(prompt):
    results = {}
    for key in GROQ_MODELS:
        print(f"  Calling {key}...")
        results[key] = call_groq_model(key, prompt)
    return results


def list_live_models():
    import requests
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}"},
    )
    for m in resp.json().get("data", []):
        print(m["id"])


if __name__ == "__main__":
    list_live_models()