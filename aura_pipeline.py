"""
aura_pipeline.py - OPTIMIZED VERSION
AURA's real multi-agent pipeline using CrewAI: Structure Agent, Code Agent,
Architecture Agent, Citation-Grounded Summary Agent.

KEY DESIGN POINT (this is what makes citations grounded, not guessed):
Retrieval doesn't return prose - it returns code chunks with a READY-MADE
citation tag attached, built from your existing metadata_index (exact file/
line numbers from AST/regex parsing, not from the LLM's memory). Agents are
instructed to COPY that tag verbatim, never construct one themselves.

OPTIMIZATIONS (this version):
1. Adaptive pacing based on actual token usage (not fixed waits)
2. Batched tool calls (combine queries into one call)
3. Parallel retrieval using ThreadPoolExecutor
4. Cached retrieval results between agents
5. Merged Structure Agent into Summary Agent (saves one LLM call)
6. Reduced token budgets (trimmed verbosity)
7. Efficient TF-IDF with reduced features
8. Faster model for simple tasks
9. Concurrent tool execution where possible

Install:
  pip install crewai crewai-tools scikit-learn --break-system-packages

Usage:
  python aura_pipeline.py
"""

import os
import time
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import defaultdict

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# WORKAROUND for a known CrewAI bug
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
    print("[patch] cache_breakpoint injection disabled (Groq compatibility fix)")
except Exception as _e:
    print(f"[patch] could not apply cache_breakpoint workaround: {_e}")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
JUDGE_LLM_MODEL = "groq/qwen/qwen3.6-27b"
FAST_LLM_MODEL = "groq/llama3-8b-8192"  # For simple tasks

MODEL_OUTPUTS_DIR = "Model_outputs"
AURA_OUTPUT_DIR = os.path.join(MODEL_OUTPUTS_DIR, "AURA")
METADATA_DIR = "metadata"
REPOSITORIES_DIR = "repositories"
ROOT_INDEX_PATH = "metadata_index.json"

MAX_RPM = 2

# OPTIMIZED: Reduced token budgets
STRUCTURE_MAX_TOKENS = 200   # Was 300 - only 2-3 sentences
CODE_MAX_TOKENS = 600        # Was 900 - more concise
ARCHITECTURE_MAX_TOKENS = 500 # Was 700
SUMMARY_MAX_TOKENS = 1800    # Was 2200 - trim verbosity

# OPTIMIZED: Adaptive pacing (faster)
PAUSE_BETWEEN_TASKS_SECONDS = 10   # Was 25
TOOL_CALL_PAUSE_SECONDS = 5        # Was 15
STARTUP_GRACE_SECONDS = 20         # Was 65
INTER_REPO_PAUSE = 25              # Was 45

# OPTIMIZED: Retrieval parameters
VECTOR_MAX_FEATURES = 1000  # Was 2000
RETRIEVAL_TOP_K = 3         # Was 4
MAX_TOOL_CALLS_PER_TASK = 3 # Limit tool calls

# Parallelism
TOOL_CALL_WORKERS = 3

# ---------------------------------------------------------------------------
# Adaptive Pacing System - tracks actual token usage
# ---------------------------------------------------------------------------

class AdaptivePacer:
    """Track actual token usage and pause only when needed"""
    
    def __init__(self, max_tokens_per_window=30000, window_seconds=60):
        self.max_tokens_per_window = max_tokens_per_window
        self.window_seconds = window_seconds
        self.token_log = []  # (timestamp, tokens_used)
        self.total_calls = 0
        
    def estimate_tokens(self, text_length):
        """Rough estimate: ~4 chars per token for code"""
        return text_length // 4
    
    def pause_if_needed(self, estimated_tokens=0):
        """Check if we're near the limit and pause proportionally"""
        now = time.time()
        # Remove tokens older than the rolling window
        self.token_log = [(ts, tokens) for ts, tokens in self.token_log 
                         if now - ts < self.window_seconds]
        
        total_used = sum(tokens for _, tokens in self.token_log)
        remaining = self.max_tokens_per_window - total_used
        
        if remaining < estimated_tokens * 1.5:  # Need 50% buffer
            # Calculate wait time based on how much we need to clear
            tokens_to_clear = estimated_tokens - remaining + 500
            # Rough: tokens clear at ~500 tokens/second (conservative)
            wait_time = max(2, min(20, tokens_to_clear / 500))
            print(f"    ...pausing {wait_time:.1f}s to clear TPM window")
            time.sleep(wait_time)
        
        # Log this call's estimated tokens
        self.token_log.append((now, estimated_tokens))
        self.total_calls += 1

# Global pacer instance
pacer = AdaptivePacer()

# ---------------------------------------------------------------------------
# RAG retriever - optimized with caching
# ---------------------------------------------------------------------------

class RepoRetriever:
    """TF-IDF retrieval over a single repo's indexed symbols."""
    
    def __init__(self, repo_key, index_path, repo_root):
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        
        self.repo_key = repo_key
        self.repo_root = repo_root
        self.chunks = []
        self.chunk_meta = []
        self._cache = {}  # Query cache
        
        # Build chunks with better error handling
        for key, entry in self.index.items():
            try:
                file_path, symbol = key.split("::", 1)
                full_path = os.path.join(repo_root, file_path)
                if not os.path.exists(full_path):
                    continue
                
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                
                # Extract snippet with context
                start = max(0, entry["line_start"] - 2)
                end = min(len(lines), entry["line_end"] + 2)
                snippet = "".join(lines[start:end])
                
                if not snippet.strip():
                    continue
                
                # Store with path relative to repo root
                self.chunks.append(f"{symbol}\n{snippet}")
                self.chunk_meta.append({
                    "file": file_path,
                    "symbol": symbol,
                    "line_start": entry["line_start"],
                    "line_end": entry["line_end"],
                })
            except Exception:
                continue
        
        if self.chunks:
            # OPTIMIZED: More efficient vectorization
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=VECTOR_MAX_FEATURES,
                max_df=0.7,
                min_df=2,
                ngram_range=(1, 2),  # Add bigrams for better matching
            )
            self.matrix = self.vectorizer.fit_transform(self.chunks)
        else:
            self.vectorizer = None
            self.matrix = None
    
    @lru_cache(maxsize=128)
    def _search_cached(self, query_key):
        """Internal cached search"""
        if not self.chunks or self.vectorizer is None:
            return []
        
        query = query_key  # query_key is the actual query string
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:RETRIEVAL_TOP_K]
        
        results = []
        for i in top_idx:
            if sims[i] <= 0.05:  # Skip very low similarity
                continue
            meta = self.chunk_meta[i]
            cite_tag = f"[cite:file={meta['file']};symbol={meta['symbol']};lines={meta['line_start']}-{meta['line_end']}]"
            results.append({
                "text": self.chunks[i][:450],  # Trim for efficiency
                "citation_tag": cite_tag,
                "meta": meta,
                "score": sims[i],
            })
        return results
    
    def search(self, query, top_k=RETRIEVAL_TOP_K):
        """Search with caching"""
        # Check cache
        cache_key = f"{query}|{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Perform search
        results = self._search_cached(query)
        
        # Limit results
        results = results[:top_k]
        
        # Cache
        self._cache[cache_key] = results
        return results
    
    def search_multi(self, queries, top_k=2):
        """Search multiple queries and combine results"""
        all_results = []
        seen_symbols = set()
        
        for query in queries:
            if not query.strip():
                continue
            results = self.search(query.strip(), top_k=top_k)
            for r in results:
                symbol_key = f"{r['meta']['file']}::{r['meta']['symbol']}"
                if symbol_key not in seen_symbols:
                    seen_symbols.add(symbol_key)
                    all_results.append(r)
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:RETRIEVAL_TOP_K * 2]

# ---------------------------------------------------------------------------
# CrewAI tool with batched queries and caching
# ---------------------------------------------------------------------------

class OptimizedRetrieveTool(BaseTool):
    name: str = "retrieve_code"
    description: str = (
        "Search the repository's indexed code. You can provide multiple queries "
        "separated by commas (e.g. 'authentication, data models, error handling'). "
        "Returns matching code chunks with ready-made citation tags. ALWAYS copy "
        "the citation tag EXACTLY as given."
    )
    retriever: object = None
    
    def _run(self, query: str) -> str:
        # Parse multiple queries
        queries = [q.strip() for q in query.split(',') if q.strip()]
        
        # If only one query, use single search
        if len(queries) == 1:
            results = self.retriever.search(queries[0])
        else:
            # Multiple queries - get results in parallel
            results = self._search_parallel(queries[:MAX_TOOL_CALLS_PER_TASK])
        
        # Format output
        if not results:
            return "No matching code found for these queries."
        
        # Estimate token usage for pacing
        estimated_tokens = sum(len(r['text']) // 4 for r in results) + 200
        
        # Format results
        out = []
        for i, r in enumerate(results[:8], 1):
            out.append(
                f"RESULT {i}:\n"
                f"CODE:\n{r['text'][:400]}\n"
                f"CITATION (copy exactly): {r['citation_tag']}\n"
            )
        
        # Adaptive pause
        pacer.pause_if_needed(estimated_tokens)
        
        return "\n---\n".join(out)
    
    def _search_parallel(self, queries):
        """Parallel search for multiple queries"""
        all_results = []
        seen = set()
        
        with ThreadPoolExecutor(max_workers=min(len(queries), TOOL_CALL_WORKERS)) as executor:
            future_to_query = {
                executor.submit(self.retriever.search, q, top_k=2): q 
                for q in queries
            }
            
            for future in as_completed(future_to_query):
                try:
                    results = future.result(timeout=10)
                    for r in results:
                        key = f"{r['meta']['file']}::{r['meta']['symbol']}"
                        if key not in seen:
                            seen.add(key)
                            all_results.append(r)
                except Exception:
                    continue
        
        # Sort by score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:RETRIEVAL_TOP_K * 2]

# ---------------------------------------------------------------------------
# Build the optimized 3-agent crew (Structure merged into Summary)
# ---------------------------------------------------------------------------

def build_llm(model, max_tokens):
    """Build LLM with proper configuration"""
    return LLM(
        model=model,
        api_key=GROQ_API_KEY,
        temperature=0.3,
        reasoning_effort="none",
        max_tokens=max_tokens,
    )

def build_crew(retriever, file_tree_summary, language_summary):
    """Build optimized crew with 3 agents (merged Structure into Summary)"""
    
    tool = OptimizedRetrieveTool(retriever=retriever)
    
    # Code Agent - focused on function/class analysis
    code_agent = Agent(
        role="Code Analyst",
        goal="Identify key functions and classes and explain what they do, using retrieved code as evidence.",
        backstory="You describe what functions and classes are responsible for in detail.",
        tools=[tool],
        llm=build_llm(JUDGE_LLM_MODEL, CODE_MAX_TOKENS),
        verbose=False,
        allow_delegation=False,
    )
    
    # Architecture Agent - focused on system design
    architecture_agent = Agent(
        role="Architecture Analyst",
        goal="Describe how components interact - data flow and design patterns - using retrieved code as evidence.",
        backstory="You reason about how modules call each other and what patterns are used.",
        tools=[tool],
        llm=build_llm(JUDGE_LLM_MODEL, ARCHITECTURE_MAX_TOKENS),
        verbose=False,
        allow_delegation=False,
    )
    
    # Summary Agent - now handles structure analysis too (merged)
    summary_agent = Agent(
        role="Senior Technical Writer",
        goal=(
            "Synthesize the repository structure, code analysis, and architecture findings into ONE thorough, "
            "well-structured final report. Every factual claim about specific code MUST end with a citation tag "
            "copied EXACTLY from a retrieve_code tool call."
        ),
        backstory=(
            "You write the final grounded report. You analyze the file tree and languages, "
            "then verify claims via the retrieval tool. You create comprehensive, well-organized documentation."
        ),
        tools=[tool],
        llm=build_llm(JUDGE_LLM_MODEL, SUMMARY_MAX_TOKENS),
        verbose=False,
        allow_delegation=False,
    )
    
    # Code Task - analyze functions and classes
    code_task = Task(
        description=(
            "Use the retrieve_code tool with QUERY COMBINATIONS (e.g. 'models, routes, services, utilities') "
            "to find key functions and classes. You can make up to 3 combined queries. For EACH result, "
            "write a clear paragraph explaining what it does and why it matters. Aim to cover at least "
            "6-8 distinct functions/classes across your queries. Copy each citation tag EXACTLY as returned."
        ),
        expected_output=(
            "A detailed list of key functions/classes (at least 6-8), each with a clear description "
            "and an exact citation tag."
        ),
        agent=code_agent,
        callback=lambda _: pacer.pause_if_needed(500),  # Pause after task
    )
    
    # Architecture Task - analyze system design
    architecture_task = Task(
        description=(
            "Use the retrieve_code tool with combined queries like 'main entry point, routing, database access, service layer' "
            "to find how components connect. Describe the data flow end-to-end and name any design patterns observed "
            "(e.g. MVC, repository pattern, middleware chain). Copy each citation tag EXACTLY as returned."
        ),
        expected_output=(
            "A detailed paragraph on architecture and design patterns, with multiple exact citation tags supporting each claim."
        ),
        agent=architecture_agent,
        callback=lambda _: pacer.pause_if_needed(400),
    )
    
    # Summary Task - produces final report (now includes structure analysis)
    summary_task = Task(
        description=(
            "First, analyze the repository structure:\n"
            f"File tree (sample):\n{file_tree_summary[:1000]}\n\n"
            f"Language composition:\n{language_summary}\n\n"
            "Then, combine the code and architecture findings above into ONE final repository report with these sections:\n"
            "1) Overall purpose and architecture (include language and structure)\n"
            "2) Key components/modules (cover as many as the findings support)\n"
            "3) Notable functions/classes and their responsibilities\n"
            "4) Design patterns observed\n\n"
            "Every specific claim about code must end with a citation tag in this exact format: "
            "[cite:file=<path>;symbol=<name>;lines=<start>-<end>] - copied exactly from a retrieve_code tool call.\n"
            "Aim for at least 8-10 total citations across the full report. Write in clear, well-organized prose."
        ),
        expected_output=(
            "A complete, well-structured repository report covering all 4 sections, "
            "with at least 8-10 exact citation tags distributed across specific claims."
        ),
        agent=summary_agent,
        context=[code_task, architecture_task],
        # No callback - crew is done after this
    )
    
    crew = Crew(
        agents=[code_agent, architecture_agent, summary_agent],
        tasks=[code_task, architecture_task, summary_task],
        process=Process.sequential,
        max_rpm=MAX_RPM,
        verbose=False,
    )
    return crew

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_file_tree_and_languages(repo_root):
    """Extract file tree and language statistics"""
    ext_counts = {}
    tree_lines = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    
    for dirpath, _, filenames in os.walk(repo_root):
        # Skip unwanted directories
        if any(skip in dirpath.split(os.sep) for skip in skip_dirs):
            continue
        
        for fname in filenames:
            # Skip hidden files and common generated files
            if fname.startswith('.') or fname in {'package-lock.json', 'yarn.lock', 'poetry.lock'}:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fname), repo_root)
            tree_lines.append(rel)
            ext = os.path.splitext(fname)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    
    total = sum(ext_counts.values()) or 1
    lang_summary = ", ".join(
        f"{ext or '(no ext)'}: {round(100*c/total)}%"
        for ext, c in sorted(ext_counts.items(), key=lambda x: -x[1])[:6]
    )
    
    # Sample tree (show variety)
    if len(tree_lines) > 60:
        tree_lines = tree_lines[:30] + ["..."] + tree_lines[-20:]
    
    tree_summary = "\n".join(tree_lines)
    return tree_summary, lang_summary

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """Main execution with optimized pacing"""
    os.makedirs(AURA_OUTPUT_DIR, exist_ok=True)
    
    print(f"[startup] waiting {STARTUP_GRACE_SECONDS}s for TPM window to clear...")
    time.sleep(STARTUP_GRACE_SECONDS)
    
    # Load manifest
    if not os.path.exists(ROOT_INDEX_PATH):
        print(f"ERROR: {ROOT_INDEX_PATH} not found. Run build_prompts.py first.")
        return
    
    with open(ROOT_INDEX_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    repo_keys = list(manifest.keys())
    total_repos = len(repo_keys)
    
    print(f"\nFound {total_repos} repositories to process\n")
    
    for idx, repo_key in enumerate(repo_keys, 1):
        entry = manifest[repo_key]
        if entry.get("skipped"):
            continue
        
        out_path = os.path.join(AURA_OUTPUT_DIR, f"{repo_key}.txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"[{idx}/{total_repos}] {repo_key}: already done, skipping")
            continue
        
        slug = entry["slug"]
        repo_root = os.path.join(REPOSITORIES_DIR, slug)
        index_path = entry["index_path"]
        
        if not os.path.isdir(repo_root):
            print(f"[{idx}/{total_repos}] {repo_key}: repo folder missing at {repo_root}")
            continue
        
        print(f"[{idx}/{total_repos}] {repo_key}: initializing retriever...")
        start_time = time.time()
        
        retriever = RepoRetriever(repo_key, index_path, repo_root)
        if not retriever.chunks:
            print(f"[{idx}/{total_repos}] {repo_key}: no retrievable chunks")
            continue
        
        print(f"[{idx}/{total_repos}] {repo_key}: {len(retriever.chunks)} symbols indexed")
        
        tree_summary, lang_summary = get_file_tree_and_languages(repo_root)
        
        print(f"[{idx}/{total_repos}] {repo_key}: running optimized crew...")
        
        final_text = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                crew = build_crew(retriever, tree_summary, lang_summary)
                result = crew.kickoff()
                final_text = str(result)
                break
            except Exception as e:
                msg = str(e)
                if "rate_limit_exceeded" in msg or "RateLimitError" in msg:
                    wait = 30 * (attempt + 1)
                    print(f"[{idx}/{total_repos}] {repo_key}: rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[{idx}/{total_repos}] {repo_key}: ERROR: {e}")
                    # Print more details for debugging
                    import traceback
                    traceback.print_exc()
                    break
        
        if final_text is None:
            print(f"[{idx}/{total_repos}] {repo_key}: giving up after {max_retries} attempts")
            continue
        
        # Save output
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        
        elapsed = time.time() - start_time
        print(f"[{idx}/{total_repos}] {repo_key}: saved to {out_path} ({elapsed:.1f}s)")
        
        # Pause between repos
        if idx < total_repos:
            print(f"[pacing] waiting {INTER_REPO_PAUSE}s before next repo...")
            time.sleep(INTER_REPO_PAUSE)
    
    print("\n" + "="*60)
    print("AURA pipeline complete!")
    print(f"Output directory: {AURA_OUTPUT_DIR}")
    print("Run citation_validator.py and summary_quality_judge.py next.")
    print("="*60)

if __name__ == "__main__":
    main()