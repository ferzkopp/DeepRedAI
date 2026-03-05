#!/usr/bin/env python3
"""
Test LLM-based temporal classification for Wikipedia articles.

Evaluates whether a local LLM can classify Wikipedia article content as
"old" (pre-1969, suitable for the Deep Red project), "new" (post-1969),
or "unsure" (ambiguous — content falls in the 1960-1980 transition zone).
The script selects articles with known temporal annotations, sends article
excerpts to the LLM with a structured prompt, and compares the LLM's
categorical assessment against the ground-truth dates.

Results include a 3x3 confusion matrix, classification accuracy, and the
critical misclassification rate (new articles wrongly tagged as old).

Prerequisites:
    - Source deepred-env.sh (sets INFERENCE_HOST, INFERENCE_PORT, PG_*, etc.)
    - PostgreSQL database with augmented temporal data (has_temporal_info)
    - LLM server running on INFERENCE_HOST:INFERENCE_PORT (default localhost:1234)
    - Optionally: REMOTE_HOST set for remote GPU inference

Usage:
    source /mnt/data/DeepRedAI/deepred-env.sh
    python3 scripts/test_llm_temporal.py                    # 100 articles
    python3 scripts/test_llm_temporal.py -n 50              # 50 articles
    python3 scripts/test_llm_temporal.py --verbose          # show per-article details
    python3 scripts/test_llm_temporal.py --evaluate          # evaluate articles without ground truth
    python3 scripts/test_llm_temporal.py --evaluate -n 20    # evaluate 20 unannotated articles
    python3 scripts/test_llm_temporal.py --max-chars 4000   # longer excerpts
    python3 scripts/test_llm_temporal.py --category people  # only people (birth+death)
    python3 scripts/test_llm_temporal.py --category events  # only events
    python3 scripts/test_llm_temporal.py --category mixed   # mixture (default)

Environment Variables:
    INFERENCE_HOST     LLM server host (default: localhost)
    INFERENCE_PORT     LLM server port (default: 1234)
    REMOTE_HOST        Remote GPU server hostname (auto-used if reachable)
    REMOTE_LLM_PORT    Remote LLM port (default: 1234)
    PG_HOST            PostgreSQL host (default: localhost)
    PG_PORT            PostgreSQL port (default: 5432)
    PG_DATABASE        Database name (default: wikidb)
    PG_USER            Database user (default: wiki)
    PG_PASSWORD        Database password (default: wiki)
"""

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import psycopg2
import requests

# =============================================================================
# Configuration
# =============================================================================

# LLM endpoints — local
LOCAL_LLM_HOST = os.environ.get('INFERENCE_HOST', 'localhost')
LOCAL_LLM_PORT = int(os.environ.get('INFERENCE_PORT', 1234))

# LLM endpoints — remote (optional)
REMOTE_HOST = os.environ.get('REMOTE_HOST', '')
REMOTE_LLM_PORT = int(os.environ.get('REMOTE_LLM_PORT', 1234))

# PostgreSQL
DB_CONFIG = {
    'host': os.environ.get('PG_HOST', 'localhost'),
    'port': int(os.environ.get('PG_PORT', 5432)),
    'database': os.environ.get('PG_DATABASE', 'wikidb'),
    'user': os.environ.get('PG_USER', 'wiki'),
    'password': os.environ.get('PG_PASSWORD', 'wiki'),
}

# Defaults
DEFAULT_N = 100
DEFAULT_MAX_CHARS = 3000
DEFAULT_CONCURRENCY = 4  # concurrent requests per LLM endpoint
REQUEST_TIMEOUT = 120  # seconds per LLM call
CONNECT_TIMEOUT = 10   # seconds to probe server availability

# ANSI codes
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Temporal cutoff for the Deep Red project
TEMPORAL_CUTOFF_YEAR = 1969

# Boundaries for the "unsure" zone around the cutoff
UNSURE_ZONE_START = 1960
UNSURE_ZONE_END = 1980

# Valid classification labels
VALID_CLASSIFICATIONS = {'old', 'new', 'unsure'}

# =============================================================================
# Prompt configuration
# =============================================================================

_JSON_SCHEMA = (
    'Respond ONLY with JSON: '
    '{"classification":"old|new|unsure",'
    '"confidence":"high|medium|low","reasoning":"<brief>"}'
)

# Classification prompt — categorical old/new/unsure around 1969 cutoff
CLASSIFY_STRATEGY = {
    'name': 'classify',
    'system': (
        "You are a temporal classifier for Wikipedia articles. "
        "Determine whether the subject of the article primarily references "
        "information that was available BEFORE July 1969 or AFTER. "
        "Focus on the most specific origin: birth, founding, start, "
        "discovery, publication, or creation date of the main subject.\n\n"
        "Classify as:\n"
        '  "old"   — the subject clearly predates 1969 '
        "(e.g. born before 1960, founded before 1960, historical events)\n"
        '  "new"   — the subject clearly postdates 1969 '
        "(e.g. born after 1980, founded after 1980, modern topics)\n"
        '  "unsure" — the subject falls in the 1960-1980 transition zone '
        "or there is not enough temporal signal to decide\n\n"
        + _JSON_SCHEMA
    ),
    'user': '{title}\n\n{content}\n\nIs this article old (pre-1969), new (post-1969), or unsure?',
    'temperature': 0.1,
}


# =============================================================================
# Helpers
# =============================================================================

def check_llm_server(host: str, port: int) -> Optional[str]:
    """Check if an LLM server is reachable. Returns model ID or None."""
    url = f"http://{host}:{port}/v1/models"
    try:
        resp = requests.get(url, timeout=CONNECT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get('id', '?') for m in data.get('data', [])]
        return models[0] if models else '(unknown)'
    except Exception:
        return None


def discover_llm_endpoints() -> List[Dict]:
    """Discover all reachable LLM endpoints (remote + local).

    Returns a list of dicts with keys: host, port, model, label.
    """
    endpoints = []

    if REMOTE_HOST:
        model = check_llm_server(REMOTE_HOST, REMOTE_LLM_PORT)
        if model:
            endpoints.append({
                'host': REMOTE_HOST,
                'port': REMOTE_LLM_PORT,
                'model': model,
                'label': f"remote ({REMOTE_HOST}:{REMOTE_LLM_PORT}, model: {model})",
            })

    model = check_llm_server(LOCAL_LLM_HOST, LOCAL_LLM_PORT)
    if model:
        # Avoid duplicate if local == remote
        is_dup = any(
            e['host'] == LOCAL_LLM_HOST and e['port'] == LOCAL_LLM_PORT
            for e in endpoints
        )
        if not is_dup:
            endpoints.append({
                'host': LOCAL_LLM_HOST,
                'port': LOCAL_LLM_PORT,
                'model': model,
                'label': f"local ({LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}, model: {model})",
            })

    return endpoints


def _validate_classification(value) -> Optional[str]:
    """Validate a classification label. Returns normalized label or None."""
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in VALID_CLASSIFICATIONS:
        return label
    # Try to recover common LLM variants
    if label in ('pre-1969', 'pre_1969', 'before', 'before 1969', 'historical'):
        return 'old'
    if label in ('post-1969', 'post_1969', 'after', 'after 1969', 'modern', 'recent'):
        return 'new'
    if label in ('uncertain', 'unknown', 'ambiguous', 'maybe', 'borderline'):
        return 'unsure'
    return None


def ground_truth_category(earliest_date: date) -> str:
    """Map a ground-truth earliest_date to old/new/unsure.

    - year < UNSURE_ZONE_START (1960) -> 'old'
    - year > UNSURE_ZONE_END   (1980) -> 'new'
    - 1960 <= year <= 1980             -> 'unsure'
    """
    year = earliest_date.year
    if year < UNSURE_ZONE_START:
        return 'old'
    if year > UNSURE_ZONE_END:
        return 'new'
    return 'unsure'


def call_llm_single(host: str, port: int, title: str, content: str,
                    max_chars: int, strategy: Dict) -> Dict:
    """Send one prompt strategy to the LLM. Returns parsed result dict."""
    url = f"http://{host}:{port}/v1/chat/completions"
    truncated = content[:max_chars]

    system_msg = strategy['system']
    user_msg = strategy['user'].format(
        title=title,
        chars=len(truncated),
        content=truncated,
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": strategy.get('temperature', 0.1),
        "max_tokens": 256,
    }

    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        elapsed = time.monotonic() - t0

        data = resp.json()
        raw = data['choices'][0]['message']['content']

        # Extract JSON from the response (LLMs sometimes wrap in markdown)
        json_match = re.search(r'\{[\s\S]*?\}', raw)
        if not json_match:
            return {'error': 'No JSON in response', 'raw': raw[:300],
                    'elapsed': elapsed}

        json_text = json_match.group()
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            # LLMs sometimes return single-quoted JSON — fix common issues
            fixed = json_text.replace("'", '"')
            # Also strip trailing commas before closing brace
            fixed = re.sub(r',\s*}', '}', fixed)
            try:
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                return {'error': f'Invalid JSON: {json_text[:200]}',
                        'raw': raw[:300], 'elapsed': elapsed}
        # Validate classification
        parsed['classification'] = _validate_classification(
            parsed.get('classification'))
        parsed['elapsed'] = elapsed
        parsed['raw'] = raw[:300]
        return parsed

    except requests.RequestException as e:
        return {'error': f'Request failed: {e}',
                'elapsed': time.monotonic() - t0}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {'error': f'Parse error: {e}', 'raw': raw[:300] if 'raw' in dir() else '',
                'elapsed': time.monotonic() - t0}


def fetch_test_articles(n: int, category: str) -> List[Dict]:
    """Fetch articles with known temporal dates from the database.

    Selects a balanced sample with equal numbers of pre-1969 and post-1969
    articles (based on TEMPORAL_CUTOFF_YEAR).  Within each half, articles
    are stratified by century for variety.

    Uses a two-phase approach: first pick IDs using only indexed columns
    (no content scan), then fetch content for the selected rows.

    Each returned dict has: id, title, content (truncated),
    earliest_date, latest_date.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Build category filter (operates only on date columns — indexed)
    if category == 'people':
        extra_where = """
            AND earliest_date < latest_date
            AND (latest_date - earliest_date) > 3650
            AND (latest_date - earliest_date) < 45000
        """
    elif category == 'events':
        extra_where = """
            AND (latest_date - earliest_date) < 3650
        """
    else:
        extra_where = ""

    # Split evenly: half pre-cutoff, half post-cutoff
    n_pre = n // 2
    n_post = n - n_pre
    cutoff_date = f'{TEMPORAL_CUTOFF_YEAR}-07-01'

    all_ids = []
    for label, date_filter, limit in [
        ('pre',  f"AND earliest_date < '{cutoff_date}'", n_pre),
        ('post', f"AND earliest_date >= '{cutoff_date}'", n_post),
    ]:
        per_bucket = max(2, limit // 5)
        # Over-request by 2× to compensate for Phase 2 content filtering,
        # and use a larger BERNOULLI sample to ensure enough candidates.
        over_limit = limit * 2
        id_query = f"""
            WITH sample AS (
                SELECT id, earliest_date,
                       FLOOR(EXTRACT(YEAR FROM earliest_date) / 100) AS century
                FROM articles TABLESAMPLE BERNOULLI (5)
                WHERE has_temporal_info = TRUE
                  AND earliest_date IS NOT NULL
                  AND latest_date IS NOT NULL
                  AND earliest_date >= '0001-01-01'
                  AND latest_date >= earliest_date
                  AND content IS NOT NULL
                  AND LENGTH(content) > 500
                  {extra_where}
                  {date_filter}
            ),
            ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY century ORDER BY RANDOM()
                       ) AS rn
                FROM sample
            )
            SELECT id FROM ranked
            WHERE rn <= {per_bucket}
            ORDER BY RANDOM()
            LIMIT {over_limit};
        """
        cur.execute(id_query)
        all_ids.extend(row[0] for row in cur.fetchall())

    if not all_ids:
        cur.close()
        conn.close()
        return []

    # Phase 2: fetch title, content, dates for the selected IDs.
    # Content filter already applied in Phase 1; trim to exact n here.
    cur.execute("""
        SELECT id, title, LEFT(content, 6000), earliest_date, latest_date
        FROM articles
        WHERE id = ANY(%s)
        ORDER BY RANDOM()
        LIMIT %s
    """, (all_ids, n))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            'id': r[0],
            'title': r[1],
            'content': r[2],
            'earliest_date': r[3],
            'latest_date': r[4],
        }
        for r in rows
    ]


def fetch_unannotated_articles(n: int) -> List[Dict]:
    """Fetch articles WITHOUT temporal data for LLM evaluation.

    Returns articles that have no YAGO/Wikidata temporal annotations,
    suitable for manual review of LLM estimates.

    Each returned dict has: id, title, content (truncated), url.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        WITH sample AS (
            SELECT id
            FROM articles TABLESAMPLE BERNOULLI (5)
            WHERE (has_temporal_info = FALSE OR has_temporal_info IS NULL)
              AND content IS NOT NULL
              AND LENGTH(content) > 500
        )
        SELECT a.id, a.title, LEFT(a.content, 6000), a.url
        FROM articles a
        JOIN sample s ON a.id = s.id
        ORDER BY RANDOM()
        LIMIT %s
    """, (n,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            'id': r[0],
            'title': r[1],
            'content': r[2],
            'url': r[3],
        }
        for r in rows
    ]


def _category_color(category: str) -> str:
    """Return ANSI color for a classification category."""
    if category == 'old':
        return GREEN
    if category == 'new':
        return CYAN
    if category == 'unsure':
        return YELLOW
    return RED


# =============================================================================
# Evaluate mode — no ground truth
# =============================================================================

def _run_evaluate_mode(args, endpoints, parallel) -> int:
    """Run LLM classification on articles without temporal data.
    Outputs categorical assessments with Wikipedia URLs for manual review."""

    print(f"\n{BOLD}Fetching unannotated articles from database...{RESET}")
    try:
        articles = fetch_unannotated_articles(args.n)
    except Exception as e:
        print(f"{RED}✗ Database error: {e}{RESET}")
        return 1

    if not articles:
        print(f"{RED}✗ No unannotated articles found.{RESET}")
        return 1
    print(f"  Retrieved {len(articles)} articles without temporal annotations")

    print(f"\n{BOLD}Running LLM temporal classification"
          f"{' (' + str(len(endpoints) * args.concurrency) + ' workers)' if parallel else ''}..."
          f"{RESET}\n")

    results_lock = threading.Lock()
    completed = 0
    errors = 0
    start_time = time.monotonic()

    def _eval_article(idx, article, ep):
        title = article['title']
        url = article.get('url', '')
        result = call_llm_single(ep['host'], ep['port'], title,
                                 article['content'], args.max_chars,
                                 CLASSIFY_STRATEGY)
        if 'error' in result:
            return idx, {
                'title': title, 'url': url,
                'error': result.get('error'),
                'elapsed': result.get('elapsed', 0),
                '_endpoint': ep['label'],
            }
        return idx, {
            'title': title,
            'url': url,
            'classification': result.get('classification'),
            'confidence': result.get('confidence', '?'),
            'reasoning': result.get('reasoning', ''),
            'elapsed': result.get('elapsed', 0),
            '_endpoint': ep['label'],
        }

    results = [None] * len(articles)
    total = len(articles)

    total_workers = len(endpoints) * args.concurrency
    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = {}
        for i, article in enumerate(articles):
            ep = endpoints[i % len(endpoints)]
            fut = pool.submit(_eval_article, i, article, ep)
            futures[fut] = i

        for fut in as_completed(futures):
            idx, entry = fut.result()
            results[idx] = entry
            with results_lock:
                completed += 1
                if 'error' in entry:
                    errors += 1
                if not args.verbose:
                    pct = completed * 100 // total
                    elapsed_total = time.monotonic() - start_time
                    eta = (elapsed_total / completed) * (total - completed) if completed > 0 else 0
                    print(f"\r  Progress: {completed}/{total} ({pct}%) "
                          f"| Elapsed: {elapsed_total:.0f}s | ETA: {eta:.0f}s "
                          f"| Errors: {errors}", end='', flush=True)

    total_elapsed = time.monotonic() - start_time
    if not args.verbose:
        print()

    # ── Output results ───────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 90}{RESET}")
    print(f"{BOLD}  Evaluation Results — No Ground Truth (manual review){RESET}")
    print(f"{BOLD}{'=' * 90}{RESET}\n")

    valid = [r for r in results if r and 'classification' in r]
    error_count = len([r for r in results if r and 'error' in r])

    print(f"  Total: {len(results)}  |  Successful: {len(valid)}  |  Errors: {error_count}")
    if total_elapsed > 0:
        print(f"  Time : {total_elapsed:.1f}s  |  {len(valid) / total_elapsed * 3600:.0f} articles/hour\n")

    # Table header
    print(f"  {'#':>3s}  {'Class':>6s}  {'Conf':>6s}  "
          f"{'Title':<40s}  {'Reasoning':<30s}  URL")
    print(f"  {'─' * 3}  {'─' * 6}  {'─' * 6}  {'─' * 40}  {'─' * 30}  {'─' * 40}")

    counts = {'old': 0, 'new': 0, 'unsure': 0, None: 0}
    for i, r in enumerate(valid, 1):
        cls = r.get('classification')
        conf = r.get('confidence', '?')
        title = r.get('title', '?')[:40]
        reasoning = r.get('reasoning', '')[:30]
        url = r.get('url', '')

        # Build Wikipedia URL from title if url column is empty
        if not url:
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        color = _category_color(cls) if cls else RED
        cls_label = f"{color}{(cls or '?'):>6s}{RESET}"
        counts[cls] = counts.get(cls, 0) + 1

        print(f"  {i:3d}  {cls_label}  {conf:>6s}  {title:<40s}  {reasoning:<30s}  {url}")

    # Summary
    print(f"\n  {BOLD}Classification summary:{RESET}")
    print(f"    {GREEN}old{RESET}    (pre-{TEMPORAL_CUTOFF_YEAR}) : {counts.get('old', 0)}")
    print(f"    {CYAN}new{RESET}    (post-{TEMPORAL_CUTOFF_YEAR}): {counts.get('new', 0)}")
    print(f"    {YELLOW}unsure{RESET} (1960-1980) : {counts.get('unsure', 0)}")
    print(f"    {RED}failed{RESET}             : {counts.get(None, 0)}")
    print()

    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Test LLM-based temporal classification (old/new/unsure) on Wikipedia articles')
    parser.add_argument('-n', type=int, default=DEFAULT_N,
                        help=f'Number of articles to test (default: {DEFAULT_N})')
    parser.add_argument('--max-chars', type=int, default=DEFAULT_MAX_CHARS,
                        help=f'Max content chars sent to LLM (default: {DEFAULT_MAX_CHARS})')
    parser.add_argument('--category', choices=['people', 'events', 'mixed'],
                        default='mixed',
                        help='Article category filter (default: mixed)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate articles WITHOUT ground truth (outputs estimates + URLs for manual review)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-article results')
    parser.add_argument('--host', default=None,
                        help='Override LLM host (skip auto-detection)')
    parser.add_argument('--port', type=int, default=None,
                        help='Override LLM port')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Concurrent requests per LLM endpoint (default: {DEFAULT_CONCURRENCY})')
    args = parser.parse_args()

    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  LLM Temporal Classification — old / new / unsure{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    # ── Discover LLM endpoints ────────────────────────────────────────────
    if args.host and args.port:
        model = check_llm_server(args.host, args.port)
        if not model:
            print(f"\n{RED}✗ LLM server not reachable at {args.host}:{args.port}{RESET}")
            return 1
        endpoints = [{
            'host': args.host, 'port': args.port, 'model': model,
            'label': f"override ({args.host}:{args.port}, model: {model})",
        }]
    else:
        endpoints = discover_llm_endpoints()
        if not endpoints:
            print(f"\n{RED}✗ No LLM server reachable.{RESET}")
            print("  Ensure a server is running on INFERENCE_HOST:INFERENCE_PORT")
            if REMOTE_HOST:
                print(f"  or on REMOTE_HOST ({REMOTE_HOST}:{REMOTE_LLM_PORT})")
            return 1

    parallel = len(endpoints) > 1 or args.concurrency > 1
    total_workers = len(endpoints) * args.concurrency
    print(f"\n  LLM endpoints: {len(endpoints)}"
          f"  ×  {args.concurrency} concurrent = {total_workers} workers"
          + (" (parallel)" if parallel else ""))
    for ep in endpoints:
        print(f"    - {ep['label']}")
    print(f"  Articles     : {args.n}")
    print(f"  Max chars    : {args.max_chars}")
    print(f"  Concurrency  : {args.concurrency} per endpoint")
    print(f"  Mode         : {'evaluate (no ground truth)' if args.evaluate else 'test (with ground truth)'}")
    if not args.evaluate:
        print(f"  Category     : {args.category}")

    # ── Evaluate mode (no ground truth) ──────────────────────────────────
    if args.evaluate:
        return _run_evaluate_mode(args, endpoints, parallel)

    # ── Fetch test articles ──────────────────────────────────────────────
    print(f"\n{BOLD}Fetching articles from database...{RESET}")
    try:
        articles = fetch_test_articles(args.n, args.category)
    except Exception as e:
        print(f"{RED}✗ Database error: {e}{RESET}")
        return 1

    if not articles:
        print(f"{RED}✗ No articles found matching criteria.{RESET}")
        return 1
    print(f"  Retrieved {len(articles)} articles with known temporal dates")

    # ── Run LLM classification ────────────────────────────────────────────
    print(f"\n{BOLD}Running LLM temporal classification"
          f"{' (' + str(total_workers) + ' workers)' if parallel else ''}..."
          f"{RESET}\n")

    # -- Shared state for progress tracking --
    results_lock = threading.Lock()
    results = []           # filled in original article order
    errors = 0
    completed = 0
    ep_counts = {ep['label']: 0 for ep in endpoints}  # per-endpoint counters
    start_time = time.monotonic()

    def _process_article(idx: int, article: Dict, ep: Dict) -> Tuple[int, Dict]:
        """Worker function: call LLM and return (idx, entry)."""
        title = article['title']
        gt_earliest = article['earliest_date']
        gt_latest = article['latest_date']
        gt_category = ground_truth_category(gt_earliest)

        result = call_llm_single(ep['host'], ep['port'], title,
                                 article['content'], args.max_chars,
                                 CLASSIFY_STRATEGY)

        if 'error' in result:
            return idx, {
                'title': title,
                'gt_earliest': gt_earliest,
                'gt_latest': gt_latest,
                'gt_category': gt_category,
                'error': result.get('error'),
                'elapsed': result.get('elapsed', 0),
                '_endpoint': ep['label'],
            }

        classification = result.get('classification')
        confidence = result.get('confidence', '?')
        reasoning = result.get('reasoning', '')
        elapsed = result.get('elapsed', 0)

        return idx, {
            'title': title,
            'gt_earliest': gt_earliest,
            'gt_latest': gt_latest,
            'gt_category': gt_category,
            'classification': classification,
            'confidence': confidence,
            'reasoning': reasoning,
            'elapsed': elapsed,
            '_endpoint': ep['label'],
        }

    # Pre-allocate result slots so we can fill them out-of-order
    results = [None] * len(articles)
    total = len(articles)

    # Submit all tasks — round-robin across endpoints
    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = {}
        for i, article in enumerate(articles):
            ep = endpoints[i % len(endpoints)]
            fut = pool.submit(_process_article, i, article, ep)
            futures[fut] = i

        for fut in as_completed(futures):
            idx, entry = fut.result()
            results[idx] = entry

            with results_lock:
                completed += 1
                ep_label = entry.get('_endpoint', '?')
                ep_counts[ep_label] = ep_counts.get(ep_label, 0) + 1
                if 'error' in entry:
                    errors += 1

                if args.verbose:
                    title = entry['title']
                    gt_cat = entry.get('gt_category', '?')
                    if 'error' in entry:
                        print(f"  [{completed:3d}/{total}] {RED}✗{RESET} {title}")
                        print(f"            Error: {entry['error']}")
                    else:
                        cls = entry.get('classification', '?')
                        confidence = entry.get('confidence', '?')
                        elapsed = entry.get('elapsed', 0)
                        match = cls == gt_cat
                        color = GREEN if match else RED
                        ep_short = ep_label.split('(')[0].strip()
                        gt_year = entry['gt_earliest'].year
                        print(f"  [{completed:3d}/{total}] {title}")
                        print(f"            GT: {gt_cat} ({gt_year})  "
                              f"LLM: {color}{cls}{RESET}  "
                              f"({confidence})  "
                              f"{elapsed:.1f}s  [{ep_short}]")
                else:
                    pct = completed * 100 // total
                    elapsed_total = time.monotonic() - start_time
                    eta = (elapsed_total / completed) * (total - completed) if completed > 0 else 0
                    print(f"\r  Progress: {completed}/{total} ({pct}%) "
                          f"| Elapsed: {elapsed_total:.0f}s | ETA: {eta:.0f}s "
                          f"| Errors: {errors}", end='', flush=True)

    total_elapsed = time.monotonic() - start_time
    if not args.verbose:
        print()  # newline after progress bar

    # ── Compute statistics ───────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  Results Summary{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    valid = [r for r in results if r and 'classification' in r and r.get('classification') is not None]
    error_count = len([r for r in results if r and 'error' in r])
    unparsed = len([r for r in results if r and 'classification' in r and r.get('classification') is None])

    print(f"\n  Total articles tested : {len(results)}")
    print(f"  Successful responses  : {len(valid)}")
    print(f"  Errors / unparseable  : {error_count + unparsed}")

    if not valid:
        print(f"\n{RED}  No valid results to analyze.{RESET}")
        return 1

    # ── Classification Distribution ──────────────────────────────────────
    print(f"\n{BOLD}  LLM Classification Distribution:{RESET}")
    cls_counts = {'old': 0, 'new': 0, 'unsure': 0}
    for r in valid:
        cls_counts[r['classification']] = cls_counts.get(r['classification'], 0) + 1
    for label in ['old', 'new', 'unsure']:
        count = cls_counts.get(label, 0)
        pct = 100 * count / len(valid)
        color = _category_color(label)
        bar = '█' * int(pct / 2)
        print(f"    {color}{label:>6s}{RESET}: {count:4d} ({pct:5.1f}%)  {bar}")

    # ── Ground Truth Distribution ────────────────────────────────────────
    print(f"\n{BOLD}  Ground Truth Distribution (year -> category):{RESET}")
    print(f"    old    = earliest_date.year < {UNSURE_ZONE_START}")
    print(f"    unsure = {UNSURE_ZONE_START} <= year <= {UNSURE_ZONE_END}")
    print(f"    new    = year > {UNSURE_ZONE_END}")
    gt_counts = {'old': 0, 'new': 0, 'unsure': 0}
    for r in valid:
        gt_counts[r['gt_category']] = gt_counts.get(r['gt_category'], 0) + 1
    for label in ['old', 'new', 'unsure']:
        count = gt_counts.get(label, 0)
        pct = 100 * count / len(valid)
        color = _category_color(label)
        print(f"    {color}{label:>6s}{RESET}: {count:4d} ({pct:5.1f}%)")

    # ── Confidence breakdown ─────────────────────────────────────────────
    print(f"\n{BOLD}  Confidence Distribution:{RESET}")
    conf_buckets = {}
    conf_correct = {}  # count of correct classifications per confidence
    for r in valid:
        c = r.get('confidence', '?').lower()
        conf_buckets[c] = conf_buckets.get(c, 0) + 1
        if r['classification'] == r['gt_category']:
            conf_correct[c] = conf_correct.get(c, 0) + 1
    for c in sorted(conf_buckets.keys()):
        count = conf_buckets[c]
        pct = 100 * count / len(valid)
        correct = conf_correct.get(c, 0)
        acc = 100 * correct / count if count > 0 else 0
        print(f"    {c:>8s}: {count:4d} ({pct:5.1f}%)  accuracy: {acc:.1f}%")

    # ── 3×3 Confusion Matrix ─────────────────────────────────────────────
    labels = ['old', 'new', 'unsure']
    matrix = {gt: {est: 0 for est in labels} for gt in labels}
    for r in valid:
        gt = r['gt_category']
        est = r['classification']
        if gt in labels and est in labels:
            matrix[gt][est] += 1

    print(f"\n{BOLD}  Confusion Matrix (rows = Ground Truth, cols = LLM):{RESET}")
    print(f"{'':>20s}  {'LLM old':>8s}  {'LLM new':>8s}  {'LLM unsure':>10s}")
    for gt in labels:
        cells = []
        for est in labels:
            val = matrix[gt][est]
            # Green for correct diagonal, red for critical error (new->old),
            # yellow for other misclassifications
            if gt == est:
                cells.append(f"{GREEN}{val:>8d}{RESET}")
            elif gt == 'new' and est == 'old':
                cells.append(f"{RED}{val:>8d}{RESET}" if val > 0 else f"{GREEN}{val:>8d}{RESET}")
            else:
                cells.append(f"{YELLOW}{val:>8d}{RESET}" if val > 0 else f"{val:>8d}")
        # Pad 'LLM unsure' column to 10 chars
        print(f"    GT {gt:>6s}:  {cells[0]}  {cells[1]}  {'':>2s}{cells[2]}")

    # ── Overall accuracy ─────────────────────────────────────────────────
    exact_match = sum(1 for r in valid if r['classification'] == r['gt_category'])
    accuracy = 100 * exact_match / len(valid)
    print(f"\n{BOLD}  Overall Accuracy:{RESET}")
    print(f"    Exact match          : {accuracy:5.1f}%  ({exact_match}/{len(valid)})")

    # "Acceptable" accuracy: also count unsure as acceptable when GT is old or new
    # (unsure is a cautious answer, not a wrong one)
    acceptable = sum(1 for r in valid
                     if r['classification'] == r['gt_category']
                     or r['classification'] == 'unsure')
    acceptable_pct = 100 * acceptable / len(valid)
    print(f"    Acceptable (+unsure) : {acceptable_pct:5.1f}%  ({acceptable}/{len(valid)})")

    # ── Critical error: new articles classified as old ────────────────────
    gt_new_total = gt_counts.get('new', 0)
    new_as_old = matrix.get('new', {}).get('old', 0)
    print(f"\n{BOLD}  Critical Error — new articles wrongly tagged as old:{RESET}")
    if gt_new_total > 0:
        misclass_rate = 100 * new_as_old / gt_new_total
        correct_rate = 100 * matrix.get('new', {}).get('new', 0) / gt_new_total
        rate_color = GREEN if new_as_old == 0 else (YELLOW if misclass_rate < 10 else RED)
        print(f"    {BOLD}> new->old misclass rate : "
              f"{rate_color}{misclass_rate:5.1f}%{RESET}  "
              f"({new_as_old}/{gt_new_total} new articles wrongly tagged as old)")
        print(f"    {BOLD}> new->new correct rate  : "
              f"{GREEN}{correct_rate:5.1f}%{RESET}  "
              f"({matrix['new']['new']}/{gt_new_total})")
    else:
        print(f"    (no ground-truth 'new' articles in sample)")

    # ── Per-category precision / recall ──────────────────────────────────
    print(f"\n{BOLD}  Per-Category Metrics:{RESET}")
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[gt][label] for gt in labels if gt != label)
        fn = sum(matrix[label][est] for est in labels if est != label)
        prec = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        color = _category_color(label)
        print(f"    {color}{label:>6s}{RESET}  "
              f"Precision: {prec:5.1f}%  Recall: {rec:5.1f}%  F1: {f1:5.1f}%")

    # Show miscategorized articles
    miscat = [r for r in valid if r['classification'] != r['gt_category']]
    if miscat and args.verbose:
        print(f"\n    Miscategorized articles ({len(miscat)}):")
        for r in miscat[:20]:
            gt_color = _category_color(r['gt_category'])
            est_color = _category_color(r['classification'])
            print(f"      {r['title'][:38]:<38s}  "
                  f"GT: {gt_color}{r['gt_category']:>6s}{RESET} ({r['gt_earliest'].year})  "
                  f"LLM: {est_color}{r['classification']:>6s}{RESET}")

    # Per-endpoint breakdown
    if parallel:
        print(f"\n{BOLD}  Per-Endpoint Breakdown:{RESET}")
        for ep_label in sorted(ep_counts.keys()):
            count = ep_counts[ep_label]
            ep_times = [r['elapsed'] for r in results
                        if r and r.get('_endpoint') == ep_label
                        and 'elapsed' in r and r['elapsed'] > 0]
            if ep_times:
                avg_t = statistics.mean(ep_times)
                print(f"    {ep_label}")
                print(f"      Articles: {count}  |  Mean: {avg_t:.2f}s  |  "
                      f"Throughput: {3600/avg_t:.0f}/hr")

    # Timing
    print(f"\n{BOLD}  Timing:{RESET}")
    times = [r['elapsed'] for r in results if r and 'elapsed' in r and r['elapsed'] > 0]
    if times:
        print(f"    Total wall time      : {total_elapsed:.1f}s")
        print(f"    Mean per article     : {statistics.mean(times):.2f}s")
        print(f"    Median per article   : {statistics.median(times):.2f}s")
        print(f"    Min / Max            : {min(times):.2f}s / {max(times):.2f}s")
        # Wall-clock throughput accounts for parallelism
        wall_throughput = len(times) / total_elapsed * 3600 if total_elapsed > 0 else 0
        serial_throughput = 3600 / statistics.mean(times)
        print(f"    Serial throughput    : {serial_throughput:.0f} articles/hour")
        if parallel:
            print(f"    Parallel throughput  : {wall_throughput:.0f} articles/hour "
                  f"({wall_throughput/serial_throughput:.1f}× speedup)")
            articles_per_hour = wall_throughput
        else:
            articles_per_hour = serial_throughput

        # Extrapolate to full database
        total_articles_no_temporal = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM articles
                WHERE (has_temporal_info = FALSE OR has_temporal_info IS NULL)
                  AND content IS NOT NULL AND LENGTH(content) > 500
            """)
            total_articles_no_temporal = cur.fetchone()[0]
            cur.close()
            conn.close()
        except Exception:
            pass

        if total_articles_no_temporal:
            est_hours = total_articles_no_temporal / articles_per_hour
            print(f"\n    Articles without temporal data: {total_articles_no_temporal:,}")
            mode = "parallel" if parallel else "serial"
            print(f"    Estimated time ({mode:>8s})   : {est_hours:,.0f} hours "
                  f"({est_hours / 24:,.1f} days)")

    # Show critical errors (new->old) in verbose mode
    critical = [r for r in valid
                if r['gt_category'] == 'new' and r['classification'] == 'old']
    if critical and args.verbose:
        print(f"\n{BOLD}  Critical Errors — new articles tagged as old ({len(critical)}):{RESET}")
        for r in critical[:15]:
            ep_short = r.get('_endpoint', '').split('(')[0].strip()
            print(f"    {r['title'][:38]:<38s}  "
                  f"GT: {r['gt_earliest'].year} (new)  "
                  f"LLM: {RED}old{RESET}  "
                  f"Conf: {r.get('confidence', '?')}  "
                  f"[{ep_short}]")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
