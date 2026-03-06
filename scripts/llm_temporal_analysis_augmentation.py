#!/usr/bin/env python3
"""
LLM-based Temporal Classification Augmentation for Wikipedia Articles.

Enriches the Wikipedia PostgreSQL database with LLM-derived temporal
classifications for articles that lack structured temporal metadata from
YAGO or Wikidata.  Each article is classified as:

    O = old    — subject clearly predates 1969
    N = new    — subject clearly postdates 1969
    S = unsure — subject falls in the 1960-1980 transition zone or ambiguous
    U = unset  — not yet classified (default)

The script:
1. Adds a ``temporal_classification`` CHAR(1) column (idempotent, defaults to 'U')
2. Back-fills classification from existing YAGO/Wikidata ``earliest_date`` data
3. Discovers all reachable LLM endpoints (local + remote)
4. Fetches random batches of unclassified articles and sends them to the LLM
5. Updates the database with the LLM's classification
6. Tracks and reports progress, throughput, and ETA

The script is fully resumable — just re-run the same command and it picks up
where it left off (only articles with classification = 'U' are processed).

Prerequisites:
    - Source deepred-env.sh (sets INFERENCE_HOST, INFERENCE_PORT, PG_*, etc.)
    - PostgreSQL database with Wikipedia articles
    - LLM server(s) running Qwen 2.5 7B Q4_K_M on local and/or remote hosts

Usage:
    source /mnt/data/DeepRedAI/deepred-env.sh
    python3 scripts/llm_temporal_analysis_augmentation.py                       # default run
    python3 scripts/llm_temporal_analysis_augmentation.py --verbose             # show per-article details
    python3 scripts/llm_temporal_analysis_augmentation.py --batch-size 500      # larger DB fetch batches
    python3 scripts/llm_temporal_analysis_augmentation.py --concurrency 8       # 8 workers per endpoint
    python3 scripts/llm_temporal_analysis_augmentation.py --max-articles 10000  # stop after 10k
    python3 scripts/llm_temporal_analysis_augmentation.py --dry-run             # classify but don't write
    python3 scripts/llm_temporal_analysis_augmentation.py --backfill-only       # only backfill from dates
    python3 scripts/llm_temporal_analysis_augmentation.py --max-chars 4000      # longer excerpts to LLM

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
import logging
import os
import re
import signal
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
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
DEFAULT_BATCH_SIZE = 200       # articles fetched from DB per round
DEFAULT_MAX_CHARS = 3000       # content chars sent to LLM
DEFAULT_CONCURRENCY = 8        # concurrent requests per LLM endpoint
REQUEST_TIMEOUT = 120          # seconds per LLM call
CONNECT_TIMEOUT = 10           # seconds to probe server availability
PROGRESS_INTERVAL = 30         # seconds between progress log lines

# Temporal cutoff for the Deep Red project
TEMPORAL_CUTOFF_YEAR = 1969
UNSURE_ZONE_START = 1960
UNSURE_ZONE_END = 1980

# Classification column values (CHAR(1))
CLS_UNSET = 'U'
CLS_OLD = 'O'
CLS_NEW = 'N'
CLS_UNSURE = 'S'

# Map from LLM text labels to DB char values
LABEL_TO_CHAR = {
    'old': CLS_OLD,
    'new': CLS_NEW,
    'unsure': CLS_UNSURE,
}

# Valid LLM classification labels
VALID_CLASSIFICATIONS = {'old', 'new', 'unsure'}

# ANSI codes
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# =============================================================================
# Prompt configuration (same as test_llm_temporal.py)
# =============================================================================

_JSON_SCHEMA = (
    'Respond ONLY with JSON: '
    '{"classification":"old|new|unsure",'
    '"confidence":"high|medium|low","reasoning":"<brief>"}'
)

CLASSIFY_PROMPT = {
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
# Graceful shutdown
# =============================================================================

_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM — request graceful shutdown."""
    if not _shutdown_event.is_set():
        log.warning("Shutdown requested — finishing current batch...")
        _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# =============================================================================
# Database helpers
# =============================================================================

def get_db_connection(db_config: Dict) -> psycopg2.extensions.connection:
    """Create a new database connection."""
    return psycopg2.connect(**db_config)


def ensure_classification_column(db_config: Dict) -> None:
    """Add temporal_classification CHAR(1) column if it doesn't exist.

    Idempotent — safe to call on every run.  Defaults to 'U' (unset).
    Also creates an index for efficient queries on unclassified articles.
    """
    conn = get_db_connection(db_config)
    cur = conn.cursor()
    try:
        # Add column (idempotent)
        cur.execute("""
            ALTER TABLE articles
            ADD COLUMN IF NOT EXISTS temporal_classification CHAR(1)
                DEFAULT 'U'
                CONSTRAINT chk_temporal_classification
                    CHECK (temporal_classification IN ('U', 'O', 'N', 'S'))
        """)

        # The ADD COLUMN IF NOT EXISTS won't add the constraint if the column
        # already exists but lacks it.  Add constraint separately (idempotent).
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'chk_temporal_classification'
                ) THEN
                    ALTER TABLE articles
                    ADD CONSTRAINT chk_temporal_classification
                        CHECK (temporal_classification IN ('U', 'O', 'N', 'S'));
                END IF;
            END $$;
        """)

        # Set NULL values to default 'U'
        cur.execute("""
            UPDATE articles
            SET temporal_classification = 'U'
            WHERE temporal_classification IS NULL
        """)
        if cur.rowcount > 0:
            log.info("Set %s NULL classifications to 'U' (unset)", f"{cur.rowcount:,}")

        # Index for efficient lookup of unclassified articles
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_temporal_classification
            ON articles (temporal_classification)
        """)

        conn.commit()
        log.info("Column 'temporal_classification' ensured (CHAR(1), default 'U')")
    except psycopg2.Error as e:
        conn.rollback()
        log.error("Failed to ensure classification column: %s", e)
        raise
    finally:
        cur.close()
        conn.close()


def backfill_from_dates(db_config: Dict, dry_run: bool = False) -> Dict[str, int]:
    """Back-fill temporal_classification from existing earliest_date values.

    Only updates articles that currently have classification = 'U' (unset)
    AND have a valid earliest_date from YAGO/Wikidata augmentation.

    Mapping:
        year < 1960         → 'O' (old)
        1960 ≤ year ≤ 1980  → 'S' (unsure)
        year > 1980          → 'N' (new)
    """
    conn = get_db_connection(db_config)
    cur = conn.cursor()
    stats = {'old': 0, 'new': 0, 'unsure': 0, 'total': 0}

    try:
        # Count how many will be updated
        cur.execute("""
            SELECT COUNT(*) FROM articles
            WHERE temporal_classification = 'U'
              AND has_temporal_info = TRUE
              AND earliest_date IS NOT NULL
        """)
        total = cur.fetchone()[0]
        stats['total'] = total

        if total == 0:
            log.info("No articles to back-fill (all already classified or no temporal dates)")
            return stats

        log.info("Back-filling %s articles from existing temporal dates...", f"{total:,}")

        if dry_run:
            # Count per category without updating
            cur.execute("""
                SELECT
                    SUM(CASE WHEN EXTRACT(YEAR FROM earliest_date) < %s THEN 1 ELSE 0 END),
                    SUM(CASE WHEN EXTRACT(YEAR FROM earliest_date) > %s THEN 1 ELSE 0 END),
                    SUM(CASE WHEN EXTRACT(YEAR FROM earliest_date) >= %s
                              AND EXTRACT(YEAR FROM earliest_date) <= %s THEN 1 ELSE 0 END)
                FROM articles
                WHERE temporal_classification = 'U'
                  AND has_temporal_info = TRUE
                  AND earliest_date IS NOT NULL
            """, (UNSURE_ZONE_START, UNSURE_ZONE_END, UNSURE_ZONE_START, UNSURE_ZONE_END))
            row = cur.fetchone()
            stats['old'] = row[0] or 0
            stats['new'] = row[1] or 0
            stats['unsure'] = row[2] or 0
            log.info("DRY RUN — would back-fill: old=%s, new=%s, unsure=%s",
                     f"{stats['old']:,}", f"{stats['new']:,}", f"{stats['unsure']:,}")
            return stats

        # Bulk update: old (year < 1960)
        cur.execute("""
            UPDATE articles
            SET temporal_classification = 'O'
            WHERE temporal_classification = 'U'
              AND has_temporal_info = TRUE
              AND earliest_date IS NOT NULL
              AND EXTRACT(YEAR FROM earliest_date) < %s
        """, (UNSURE_ZONE_START,))
        stats['old'] = cur.rowcount

        # Bulk update: new (year > 1980)
        cur.execute("""
            UPDATE articles
            SET temporal_classification = 'N'
            WHERE temporal_classification = 'U'
              AND has_temporal_info = TRUE
              AND earliest_date IS NOT NULL
              AND EXTRACT(YEAR FROM earliest_date) > %s
        """, (UNSURE_ZONE_END,))
        stats['new'] = cur.rowcount

        # Bulk update: unsure (1960 ≤ year ≤ 1980)
        cur.execute("""
            UPDATE articles
            SET temporal_classification = 'S'
            WHERE temporal_classification = 'U'
              AND has_temporal_info = TRUE
              AND earliest_date IS NOT NULL
              AND EXTRACT(YEAR FROM earliest_date) >= %s
              AND EXTRACT(YEAR FROM earliest_date) <= %s
        """, (UNSURE_ZONE_START, UNSURE_ZONE_END))
        stats['unsure'] = cur.rowcount

        conn.commit()
        log.info("Back-fill complete: old=%s, new=%s, unsure=%s (total=%s)",
                 f"{stats['old']:,}", f"{stats['new']:,}", f"{stats['unsure']:,}",
                 f"{stats['old'] + stats['new'] + stats['unsure']:,}")
    except psycopg2.Error as e:
        conn.rollback()
        log.error("Back-fill failed: %s", e)
        raise
    finally:
        cur.close()
        conn.close()

    return stats


def get_classification_stats(db_config: Dict) -> Dict[str, int]:
    """Return current classification distribution."""
    conn = get_db_connection(db_config)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT temporal_classification, COUNT(*)
            FROM articles
            GROUP BY temporal_classification
            ORDER BY temporal_classification
        """)
        stats = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute("SELECT COUNT(*) FROM articles")
        stats['_total'] = cur.fetchone()[0]
        return stats
    finally:
        cur.close()
        conn.close()


def fetch_unclassified_batch(db_config: Dict, batch_size: int,
                             min_content_length: int = 500) -> List[Dict]:
    """Fetch a random batch of articles with classification = 'U'.

    Uses TABLESAMPLE BERNOULLI for efficient random sampling without a full
    sequential scan.  Returns articles with sufficient content for LLM
    classification.
    """
    conn = get_db_connection(db_config)
    cur = conn.cursor()

    try:
        # Estimate what BERNOULLI percentage to use.  We want at least
        # batch_size rows from the unclassified pool.  Over-sample by 3×
        # to account for the content length filter and sampling variance.
        cur.execute("""
            SELECT COUNT(*) FROM articles
            WHERE temporal_classification = 'U'
        """)
        unclassified_total = cur.fetchone()[0]

        if unclassified_total == 0:
            return []

        # Calculate sample percentage — at least 0.01%, at most 100%
        desired = batch_size * 3
        pct = max(0.01, min(100.0, 100.0 * desired / max(1, unclassified_total)))

        cur.execute(f"""
            WITH sample AS (
                SELECT id
                FROM articles TABLESAMPLE BERNOULLI ({pct})
                WHERE temporal_classification = 'U'
                  AND content IS NOT NULL
                  AND LENGTH(content) > %s
            )
            SELECT a.id, a.title, LEFT(a.content, 6000)
            FROM articles a
            JOIN sample s ON a.id = s.id
            ORDER BY RANDOM()
            LIMIT %s
        """, (min_content_length, batch_size))

        rows = cur.fetchall()
        return [
            {'id': r[0], 'title': r[1], 'content': r[2]}
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()


def update_classifications(db_config: Dict, updates: List[Tuple[str, int]]) -> int:
    """Bulk-update temporal_classification for a list of (classification_char, article_id) pairs.

    Returns the number of rows updated.
    """
    if not updates:
        return 0
    conn = get_db_connection(db_config)
    cur = conn.cursor()
    try:
        psycopg2.extras.execute_batch(
            cur,
            "UPDATE articles SET temporal_classification = %s WHERE id = %s",
            updates,
            page_size=500,
        )
        count = cur.rowcount
        conn.commit()
        return count
    except psycopg2.Error as e:
        conn.rollback()
        log.error("Failed to update classifications: %s", e)
        return 0
    finally:
        cur.close()
        conn.close()


# =============================================================================
# LLM helpers
# =============================================================================

def check_llm_server(host: str, port: int) -> Optional[str]:
    """Check if an LLM server is reachable.  Returns model ID or None."""
    url = f"http://{host}:{port}/v1/models"
    try:
        resp = requests.get(url, timeout=CONNECT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get('id', '?') for m in data.get('data', [])]
        return models[0] if models else '(unknown)'
    except Exception:
        return None


def get_server_slots(host: str, port: int) -> Optional[int]:
    """Query the llama.cpp /slots endpoint to discover available parallel slots.

    Requires --slots to be enabled on the server (disabled by default
    in llama.cpp builds >= b4000).  Returns None if the endpoint is disabled
    or unreachable.
    """
    url = f"http://{host}:{port}/slots"
    try:
        resp = requests.get(url, timeout=CONNECT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return len(data)
        # Server returned a JSON object (error) — endpoint is disabled
        log.warning("  /slots endpoint disabled on %s:%d — add --slots "
                    "to the server config for auto-detection", host, port)
    except Exception:
        pass
    return None


def discover_llm_endpoints() -> List[Dict]:
    """Discover all reachable LLM endpoints (remote + local).

    Returns a list of dicts with keys: host, port, model, label, slots.
    """
    endpoints = []

    if REMOTE_HOST:
        model = check_llm_server(REMOTE_HOST, REMOTE_LLM_PORT)
        if model:
            slots = get_server_slots(REMOTE_HOST, REMOTE_LLM_PORT)
            endpoints.append({
                'host': REMOTE_HOST,
                'port': REMOTE_LLM_PORT,
                'model': model,
                'slots': slots,
                'label': f"remote ({REMOTE_HOST}:{REMOTE_LLM_PORT}, model: {model})",
            })

    model = check_llm_server(LOCAL_LLM_HOST, LOCAL_LLM_PORT)
    if model:
        is_dup = any(
            e['host'] == LOCAL_LLM_HOST and e['port'] == LOCAL_LLM_PORT
            for e in endpoints
        )
        if not is_dup:
            slots = get_server_slots(LOCAL_LLM_HOST, LOCAL_LLM_PORT)
            endpoints.append({
                'host': LOCAL_LLM_HOST,
                'port': LOCAL_LLM_PORT,
                'model': model,
                'slots': slots,
                'label': f"local ({LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}, model: {model})",
            })

    return endpoints


def _validate_classification(value) -> Optional[str]:
    """Validate a classification label.  Returns normalized label or None."""
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in VALID_CLASSIFICATIONS:
        return label
    # Recover common LLM variants
    if label in ('pre-1969', 'pre_1969', 'before', 'before 1969', 'historical'):
        return 'old'
    if label in ('post-1969', 'post_1969', 'after', 'after 1969', 'modern', 'recent'):
        return 'new'
    if label in ('uncertain', 'unknown', 'ambiguous', 'maybe', 'borderline'):
        return 'unsure'
    return None


def call_llm(host: str, port: int, title: str, content: str,
             max_chars: int) -> Dict:
    """Send a classification prompt to the LLM.  Returns parsed result dict."""
    url = f"http://{host}:{port}/v1/chat/completions"
    truncated = content[:max_chars]

    payload = {
        "messages": [
            {"role": "system", "content": CLASSIFY_PROMPT['system']},
            {"role": "user", "content": CLASSIFY_PROMPT['user'].format(
                title=title, content=truncated)},
        ],
        "temperature": CLASSIFY_PROMPT.get('temperature', 0.1),
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
            fixed = json_text.replace("'", '"')
            fixed = re.sub(r',\s*}', '}', fixed)
            try:
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                return {'error': f'Invalid JSON: {json_text[:200]}',
                        'raw': raw[:300], 'elapsed': elapsed}

        classification = _validate_classification(parsed.get('classification'))
        return {
            'classification': classification,
            'confidence': parsed.get('confidence', '?'),
            'reasoning': parsed.get('reasoning', ''),
            'elapsed': elapsed,
        }

    except requests.RequestException as e:
        return {'error': f'Request failed: {e}',
                'elapsed': time.monotonic() - t0}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {'error': f'Parse error: {e}',
                'elapsed': time.monotonic() - t0}


# =============================================================================
# Classification colour helpers
# =============================================================================

def _cls_color(cls_char: str) -> str:
    """ANSI color for a classification char."""
    return {CLS_OLD: GREEN, CLS_NEW: CYAN, CLS_UNSURE: YELLOW}.get(cls_char, RED)


def _cls_name(cls_char: str) -> str:
    """Human-readable name for a classification char."""
    return {'O': 'old', 'N': 'new', 'S': 'unsure', 'U': 'unset'}.get(cls_char, '?')


# =============================================================================
# Main processing loop
# =============================================================================

def run_llm_classification(db_config: Dict, endpoints: List[Dict], args) -> Dict:
    """Main loop: fetch, classify, update in batches until done or limit reached."""

    total_workers = len(endpoints) * args.concurrency
    stats = {
        'classified': 0,
        'errors': 0,
        'old': 0,
        'new': 0,
        'unsure': 0,
        'batches': 0,
        'elapsed': 0,
    }

    # Collect all latencies for throughput reporting
    latencies = []
    start_time = time.monotonic()
    last_progress = start_time

    log.info("Starting LLM classification with %d workers across %d endpoint(s)",
             total_workers, len(endpoints))

    articles_remaining = args.max_articles  # None = unlimited

    while not _shutdown_event.is_set():
        # Determine how many to fetch this batch
        fetch_size = args.batch_size
        if articles_remaining is not None:
            if articles_remaining <= 0:
                break
            fetch_size = min(fetch_size, articles_remaining)

        # Fetch a batch of unclassified articles
        batch = fetch_unclassified_batch(db_config, fetch_size,
                                         min_content_length=500)
        if not batch:
            log.info("No more unclassified articles to process")
            break

        stats['batches'] += 1
        batch_updates = []   # (cls_char, article_id)
        batch_errors = 0

        # Process the batch with thread pool
        def _classify(idx, article, ep):
            if _shutdown_event.is_set():
                return idx, None
            result = call_llm(ep['host'], ep['port'],
                              article['title'], article['content'],
                              args.max_chars)
            return idx, result

        with ThreadPoolExecutor(max_workers=total_workers) as pool:
            futures = {}
            for i, article in enumerate(batch):
                if _shutdown_event.is_set():
                    break
                ep = endpoints[i % len(endpoints)]
                fut = pool.submit(_classify, i, article, ep)
                futures[fut] = (i, article)

            for fut in as_completed(futures):
                if _shutdown_event.is_set():
                    break
                idx, result = fut.result()
                i, article = futures[fut]

                if result is None:
                    continue  # shutdown

                if 'error' in result:
                    batch_errors += 1
                    stats['errors'] += 1
                    if args.verbose:
                        log.warning("  Error classifying '%s': %s",
                                    article['title'][:50], result['error'])
                    continue

                classification = result.get('classification')
                if classification is None:
                    batch_errors += 1
                    stats['errors'] += 1
                    if args.verbose:
                        log.warning("  Unparseable response for '%s'",
                                    article['title'][:50])
                    continue

                cls_char = LABEL_TO_CHAR[classification]
                batch_updates.append((cls_char, article['id']))
                stats[classification] += 1
                stats['classified'] += 1

                if result.get('elapsed'):
                    latencies.append(result['elapsed'])

                if args.verbose:
                    color = _cls_color(cls_char)
                    conf = result.get('confidence', '?')
                    elapsed = result.get('elapsed', 0)
                    log.info("  [%d] %s%s%s (%s) %.1fs — %s",
                             stats['classified'], color,
                             _cls_name(cls_char), RESET,
                             conf, elapsed, article['title'][:60])

        # Write batch to database
        if batch_updates and not args.dry_run:
            written = update_classifications(db_config, batch_updates)
            if args.verbose:
                log.info("  Batch %d: wrote %d classifications (%d errors)",
                         stats['batches'], written, batch_errors)

        if articles_remaining is not None:
            articles_remaining -= len(batch)

        # Periodic progress reporting (non-verbose mode)
        now = time.monotonic()
        if not args.verbose and (now - last_progress) >= PROGRESS_INTERVAL:
            elapsed_total = now - start_time
            rate = stats['classified'] / elapsed_total * 3600 if elapsed_total > 0 else 0
            log.info("Progress: %s classified | %s errors | %.0f articles/hr | %.0fs elapsed",
                     f"{stats['classified']:,}", f"{stats['errors']:,}",
                     rate, elapsed_total)
            last_progress = now

    stats['elapsed'] = time.monotonic() - start_time
    stats['latencies'] = latencies
    return stats


# =============================================================================
# Reporting
# =============================================================================

def print_summary(stats: Dict, db_config: Dict, dry_run: bool = False) -> None:
    """Print final summary statistics."""
    elapsed = stats['elapsed']
    classified = stats['classified']
    errors = stats['errors']

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  LLM Temporal Classification — Summary{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    if dry_run:
        print(f"\n  {YELLOW}DRY RUN — no changes were written to the database{RESET}")

    print(f"\n  Articles classified : {classified:,}")
    print(f"  Errors / unparseable: {errors:,}")
    print(f"  Batches completed   : {stats['batches']}")

    # Classification distribution
    print(f"\n  {BOLD}Classification Distribution:{RESET}")
    for label, char in [('old', CLS_OLD), ('new', CLS_NEW), ('unsure', CLS_UNSURE)]:
        count = stats.get(label, 0)
        pct = 100 * count / classified if classified > 0 else 0
        color = _cls_color(char)
        bar = '█' * int(pct / 2)
        print(f"    {color}{label:>6s}{RESET}: {count:5,} ({pct:5.1f}%)  {bar}")

    # Throughput
    latencies = stats.get('latencies', [])
    if latencies:
        print(f"\n  {BOLD}Throughput:{RESET}")
        print(f"    Wall time           : {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"    Mean latency        : {statistics.mean(latencies):.2f}s")
        print(f"    Median latency      : {statistics.median(latencies):.2f}s")
        print(f"    Min / Max latency   : {min(latencies):.2f}s / {max(latencies):.2f}s")
        wall_rate = classified / elapsed * 3600 if elapsed > 0 else 0
        print(f"    Throughput (actual)  : {wall_rate:,.0f} articles/hour")

        # ETA for remaining unclassified articles
        try:
            db_stats = get_classification_stats(db_config)
            remaining = db_stats.get(CLS_UNSET, 0)
            if remaining > 0 and wall_rate > 0:
                eta_hours = remaining / wall_rate
                print(f"\n    Remaining unclassified: {remaining:,}")
                print(f"    Estimated time left   : {eta_hours:,.0f} hours ({eta_hours/24:,.1f} days)")
        except Exception:
            pass

    # Current database coverage
    print(f"\n  {BOLD}Database Coverage:{RESET}")
    try:
        db_stats = get_classification_stats(db_config)
        total = db_stats.get('_total', 0)
        for char, label in [('O', 'old'), ('N', 'new'), ('S', 'unsure'), ('U', 'unset')]:
            count = db_stats.get(char, 0)
            pct = 100 * count / total if total > 0 else 0
            color = _cls_color(char)
            print(f"    {color}{label:>6s}{RESET} ({char}): {count:>10,}  ({pct:5.1f}%)")
        print(f"    {'total':>9s}  : {total:>10,}")
    except Exception as e:
        print(f"    (could not retrieve: {e})")

    print()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='LLM-based temporal classification augmentation for Wikipedia articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Server Setup for Maximum Throughput:
  Before running, configure both LLM servers for Qwen 2.5 7B Q4_K_M with
  maximum parallel slots using the --slots flag:

  # StrixHalo (local — ROCm, 128 GB unified memory):
  sudo llm-swap $DEEPRED_MODELS/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \\
      "qwen2.5-7b-instruct" 8192 --slots 8

  # A4000 (remote — CUDA, 16 GB VRAM):
  # SSH in and run:
  sudo llm-swap $DEEPRED_MODELS/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \\
      "qwen2.5-7b-instruct" 8192 --slots 4

Examples:
  # Full augmentation (resume-safe)
  python3 scripts/llm_temporal_analysis_augmentation.py

  # Only back-fill from existing YAGO/Wikidata dates
  python3 scripts/llm_temporal_analysis_augmentation.py --backfill-only

  # Dry run to see what would happen
  python3 scripts/llm_temporal_analysis_augmentation.py --dry-run --max-articles 100

  # Verbose output with 500-article batches
  python3 scripts/llm_temporal_analysis_augmentation.py --verbose --batch-size 500

  # Process at most 10,000 articles then stop
  python3 scripts/llm_temporal_analysis_augmentation.py --max-articles 10000
        """,
    )

    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Articles fetched from DB per round (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-chars', type=int, default=DEFAULT_MAX_CHARS,
                        help=f'Max content chars sent to LLM (default: {DEFAULT_MAX_CHARS})')
    parser.add_argument('--max-articles', type=int, default=None,
                        help='Stop after classifying this many articles (default: unlimited)')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Concurrent requests per LLM endpoint (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Classify articles but do not write to database')
    parser.add_argument('--backfill-only', action='store_true',
                        help='Only back-fill classification from existing earliest_date; skip LLM')
    parser.add_argument('--skip-backfill', action='store_true',
                        help='Skip the back-fill step and go straight to LLM classification')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-article classification details')
    parser.add_argument('--host', default=None,
                        help='Override LLM host (skip auto-detection)')
    parser.add_argument('--port', type=int, default=None,
                        help='Override LLM port')
    parser.add_argument('--db-host', default=None, help='PostgreSQL host')
    parser.add_argument('--db-name', default=None, help='Database name')
    parser.add_argument('--db-user', default=None, help='Database user')
    parser.add_argument('--db-password', default=None, help='Database password')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build DB config with CLI overrides
    db_config = dict(DB_CONFIG)
    if args.db_host:
        db_config['host'] = args.db_host
    if args.db_name:
        db_config['database'] = args.db_name
    if args.db_user:
        db_config['user'] = args.db_user
    if args.db_password:
        db_config['password'] = args.db_password

    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  LLM Temporal Classification Augmentation{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    # ── Step 1: Ensure schema ─────────────────────────────────────────────
    log.info("Ensuring temporal_classification column exists...")
    try:
        ensure_classification_column(db_config)
    except Exception as e:
        log.error("Schema setup failed: %s", e)
        return 1

    # Show current stats
    try:
        db_stats = get_classification_stats(db_config)
        total = db_stats.get('_total', 0)
        unset = db_stats.get(CLS_UNSET, 0)
        log.info("Current database: %s total articles, %s unclassified",
                 f"{total:,}", f"{unset:,}")
        for char, label in [('O', 'old'), ('N', 'new'), ('S', 'unsure'), ('U', 'unset')]:
            count = db_stats.get(char, 0)
            pct = 100 * count / total if total > 0 else 0
            log.info("  %s (%s): %s (%.1f%%)", label, char, f"{count:,}", pct)
    except Exception as e:
        log.warning("Could not retrieve stats: %s", e)

    # ── Step 2: Back-fill from existing dates ─────────────────────────────
    if not args.skip_backfill:
        log.info("Back-filling classifications from existing temporal dates...")
        try:
            bf_stats = backfill_from_dates(db_config, dry_run=args.dry_run)
        except Exception as e:
            log.error("Back-fill failed: %s", e)
            return 1

    if args.backfill_only:
        log.info("--backfill-only specified; skipping LLM classification")
        # Show updated stats
        try:
            db_stats = get_classification_stats(db_config)
            total = db_stats.get('_total', 0)
            for char, label in [('O', 'old'), ('N', 'new'), ('S', 'unsure'), ('U', 'unset')]:
                count = db_stats.get(char, 0)
                pct = 100 * count / total if total > 0 else 0
                log.info("  %s (%s): %s (%.1f%%)", label, char, f"{count:,}", pct)
        except Exception:
            pass
        return 0

    # ── Step 3: Discover LLM endpoints ────────────────────────────────────
    if args.host and args.port:
        model = check_llm_server(args.host, args.port)
        if not model:
            log.error("LLM server not reachable at %s:%d", args.host, args.port)
            return 1
        slots = get_server_slots(args.host, args.port)
        endpoints = [{
            'host': args.host, 'port': args.port, 'model': model,
            'slots': slots,
            'label': f"override ({args.host}:{args.port}, model: {model})",
        }]
    else:
        endpoints = discover_llm_endpoints()
        if not endpoints:
            log.error("No LLM server reachable. Ensure a server is running on "
                      "INFERENCE_HOST:INFERENCE_PORT")
            if REMOTE_HOST:
                log.error("  or on REMOTE_HOST (%s:%d)", REMOTE_HOST, REMOTE_LLM_PORT)
            return 1

    total_workers = len(endpoints) * args.concurrency
    log.info("LLM endpoints: %d  ×  %d concurrent = %d workers",
             len(endpoints), args.concurrency, total_workers)
    for ep in endpoints:
        slots_info = f", slots: {ep['slots']}" if ep.get('slots') else ""
        log.info("  - %s%s", ep['label'], slots_info)

    # Warn if concurrency exceeds server slots
    for ep in endpoints:
        if ep.get('slots') and args.concurrency > ep['slots']:
            log.warning("  ⚠  Concurrency (%d) exceeds server slots (%d) for %s — "
                        "requests will queue on the server",
                        args.concurrency, ep['slots'], ep['label'])

    log.info("Settings: batch_size=%d, max_chars=%d, max_articles=%s, dry_run=%s",
             args.batch_size, args.max_chars,
             f"{args.max_articles:,}" if args.max_articles else "unlimited",
             args.dry_run)

    # ── Step 4: Run LLM classification ────────────────────────────────────
    llm_stats = run_llm_classification(db_config, endpoints, args)

    # ── Step 5: Report ────────────────────────────────────────────────────
    print_summary(llm_stats, db_config, dry_run=args.dry_run)

    if _shutdown_event.is_set():
        log.info("Shutdown completed gracefully. Re-run to continue.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
