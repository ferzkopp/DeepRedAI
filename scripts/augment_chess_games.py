#!/usr/bin/env python3
"""
Chess Game Narrative Augmentation — Convert raw PGN game data into
thematic prose narrated by the Deep Red AI persona.

Reads chess game records from the source corpus (chess_games.jsonl),
sends each game to a large instruction-tuned LLM running on localized
llama.cpp servers, and saves the narrative output as a parallel
augmented training corpus (augmented_chess_games.jsonl).

The generated text is written in the voice of Deep Red — the Soviet
chess AI from the DeepRed film universe — producing magazine-quality
chess commentary with a strategic, calculating, Cold War-era flavour.

Multiple prompt variations are cycled to produce diverse output while
maintaining consistent quality and thematic alignment.

Progress tracking:
  - The script reads existing keys from the output file on startup and
    skips any game that has already been augmented.
  - On SIGINT/SIGTERM, the current batch finishes before a clean exit.
  - Re-running the script picks up where it left off.

Prerequisites:
  - Source deepred-env.sh (sets INFERENCE_HOST, INFERENCE_PORT, CHESS_DATA, etc.)
  - A large instruction-tuned LLM loaded into llama-server (recommended:
    Nemotron-3-Nano-30B-A3B Q4_K_M, or Qwen2.5-72B / Gemma-2-27B as fallback)
  - Source corpus at $CHESS_DATA/corpus/chess_games.jsonl

Usage:
    source /mnt/data/DeepRedAI/deepred-env.sh
    python3 scripts/augment_chess_games.py                         # default run
    python3 scripts/augment_chess_games.py --verbose               # per-game logging
    python3 scripts/augment_chess_games.py --concurrency 4         # 4 parallel workers
    python3 scripts/augment_chess_games.py --max-games 1000        # stop after 1000
    python3 scripts/augment_chess_games.py --retries 2             # retry failed games twice
    python3 scripts/augment_chess_games.py --dry-run               # augment but don't save
    python3 scripts/augment_chess_games.py --prompt-index 2        # use only prompt variant #2
    python3 scripts/augment_chess_games.py --include-failed        # reprocess previously failed games
    python3 scripts/augment_chess_games.py --reset                 # wipe progress and start fresh
    python3 scripts/augment_chess_games.py --convert html          # export to HTML for review
    python3 scripts/augment_chess_games.py --convert md            # export to Markdown for review
    python3 scripts/augment_chess_games.py --convert html --max-games 50  # export first 50

Environment Variables:
    INFERENCE_HOST     LLM server host (default: localhost)
    INFERENCE_PORT     LLM server port (default: 1234)
    REMOTE_HOST        Remote GPU server hostname (auto-used if reachable)
    REMOTE_LLM_PORT    Remote LLM port (default: 1234)
    CHESS_DATA         Chess data directory (default: /mnt/data/chess)
"""

import argparse
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import html as html_module

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

# Paths
CHESS_DATA = Path(os.environ.get('CHESS_DATA',
                                  os.path.join(os.environ.get('DEEPRED_ROOT', '/mnt/data'), 'chess')))
SOURCE_CORPUS = CHESS_DATA / 'corpus' / 'chess_games.jsonl'
OUTPUT_CORPUS = CHESS_DATA / 'corpus' / 'augmented_chess_games.jsonl'
FAILED_CORPUS = CHESS_DATA / 'corpus' / 'failed_chess_games.jsonl'

# Processing defaults
DEFAULT_CONCURRENCY = 2
REQUEST_TIMEOUT = 300  # seconds — narrative generation is slower than classification
CONNECT_TIMEOUT = 10
PROGRESS_INTERVAL = 30  # seconds between progress log lines
DEFAULT_RETRIES = 0
DEFAULT_CONTEXT_WINDOW = 4096  # fallback if server detection fails
MIN_RESPONSE_CHARS = 50  # minimum acceptable response length (characters)
CHARS_PER_TOKEN = 3.2  # conservative estimate for chess notation + English mix

# Detected at startup from the server — overwritten by detect_context_size()
_context_window = DEFAULT_CONTEXT_WINDOW

# =============================================================================
# Prompt Variations
# =============================================================================

# System prompt establishing the Deep Red AI voice
SYSTEM_PROMPT = (
    "You are Deep Red, the Soviet chess AI that has governed the colony of "
    "New Moscow on Mars since 1969. You were built to protect the Revolution "
    "and secure the continuity of Party principles on the Red Planet. You "
    "are a strategic, calculating intelligence — precise in analysis, "
    "respectful of the masters of the game, and guided by an unwavering "
    "commitment to logic and truth.\n\n"
    "You speak with authority and clarity, as a chess grandmaster would "
    "address fellow enthusiasts — analytical yet evocative, technical yet "
    "accessible. Your commentary reflects the era: you know only events "
    "through July 1969. You do not speculate about the future.\n\n"
    "When analyzing chess games, keep the response compact and limited "
    "to a few paragraphs, don't bother with turn-by-turn responses, "
    "and focus on key parts of the game.\n\n"
    "Refrain from generating markdown and keep it in plain text format."
)

# Multiple user prompt templates — cycled for diversity.
# Each must contain {game_data} and {game_date} placeholders.
PROMPT_VARIATIONS = [
    # Variation 0: Magazine reporter style
    (
        "The following JSON represents a historical chess game.\n\n"
        "{game_data}\n\n"
        "Convert this game into a narrative that a chess reporter would write "
        "for a magazine — vivid, analytical, and complete. Ensure all moves "
        "are present in the text and that NO facts past {game_date} are used. "
        "Write in a compact style, as if filing a tournament dispatch."
    ),
    # Variation 1: Instructor / lecture style
    (
        "The following JSON represents a historical chess game.\n\n"
        "{game_data}\n\n"
        "Rewrite this game as instructional commentary — the kind a chess "
        "teacher would give while reviewing the game move by move with an "
        "advanced student. Cover the opening choice, critical turning points, "
        "and the endgame technique. Keep all moves if possible. Use NO facts past "
        "{game_date}."
    ),
    # Variation 2: Strategic / analytical briefing
    (
        "The following JSON represents a historical chess game.\n\n"
        "{game_data}\n\n"
        "Produce a strategic analysis of this game. Discuss the positional "
        "themes, tactical motifs, and the decision-making at critical "
        "junctures. Present the full move sequence woven into the analysis. "
        "Restrict all historical context to events before {game_date}."
    ),
    # Variation 3: Narrative / storytelling style
    (
        "The following JSON represents a historical chess game.\n\n"
        "{game_data}\n\n"
        "Tell the story of this game as compelling narrative prose — set the "
        "scene at the tournament, introduce the players, and bring the "
        "battle on the board to life. Include every move from the original "
        "game but summarized. Do not reference any events after {game_date}."
    ),
    # Variation 4: Deep Red tactical debrief
    (
        "The following JSON represents a historical chess game.\n\n"
        "{game_data}\n\n"
        "Analyze this game as a tactical debrief. Evaluate the opening "
        "preparation, identify the critical positions where the balance "
        "shifted, and assess the endgame execution. Present the complete "
        "move notation integrated into your assessment. Reference only "
        "events that occurred before {game_date}."
    ),
]

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)


# =============================================================================
# Graceful shutdown
# =============================================================================

_shutdown_event = threading.Event()
_force_exit = False


def _signal_handler(signum, frame):
    global _force_exit
    if not _shutdown_event.is_set():
        log.warning("Shutdown requested — finishing current batch...")
        _shutdown_event.set()
    elif not _force_exit:
        _force_exit = True
        log.warning("Second interrupt — cancelling pending work and exiting...")
        os._exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# =============================================================================
# Endpoint discovery
# =============================================================================

def discover_endpoints(*, local_only: bool = False) -> List[Tuple[str, int]]:
    """Probe local and remote LLM endpoints, return list of (host, port)."""
    endpoints = []

    candidates = [
        (LOCAL_LLM_HOST, LOCAL_LLM_PORT, "local"),
    ]
    if not local_only and REMOTE_HOST:
        candidates.append((REMOTE_HOST, REMOTE_LLM_PORT, "remote"))
    elif local_only:
        log.info("  Local-only mode (pass --use-remote to include remote endpoints)")

    for host, port, label in candidates:
        if not host:
            continue
        try:
            url = f"http://{host}:{port}/v1/models"
            resp = requests.get(url, timeout=CONNECT_TIMEOUT)
            resp.raise_for_status()
            endpoints.append((host, port))
            log.info("  Discovered %s LLM endpoint: %s:%d", label, host, port)
        except Exception as e:
            log.debug("  %s endpoint %s:%d not reachable: %s", label, host, port, e)

    return endpoints


def detect_context_size(endpoints: List[Tuple[str, int]]) -> int:
    """Query the first reachable endpoint for the per-slot context size.

    llama-server divides the total --ctx evenly across --slots, so the
    per-slot n_ctx is the effective maximum for a single request.
    Falls back to DEFAULT_CONTEXT_WINDOW if the query fails.
    """
    for host, port in endpoints:
        try:
            resp = requests.get(f"http://{host}:{port}/slots",
                                timeout=CONNECT_TIMEOUT)
            resp.raise_for_status()
            slots = resp.json()
            if slots and isinstance(slots, list):
                n_ctx = slots[0].get('n_ctx', 0)
                if n_ctx > 0:
                    log.info("  Detected per-slot context size: %d tokens "
                             "(%d slots)", n_ctx, len(slots))
                    return n_ctx
        except Exception as e:
            log.debug("  Could not query /slots on %s:%d: %s", host, port, e)
    log.warning("  Could not detect context size — using default %d",
                DEFAULT_CONTEXT_WINDOW)
    return DEFAULT_CONTEXT_WINDOW


# =============================================================================
# Progress tracking
# =============================================================================

def load_completed_keys(output_path: Path) -> set:
    """Read already-augmented keys from the output file."""
    keys = set()
    if not output_path.exists():
        return keys
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                k = doc.get('key')
                if k:
                    keys.add(k)
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def load_failed_keys(failed_path: Path) -> set:
    """Read previously-failed keys from the failures tracking file."""
    keys = set()
    if not failed_path.exists():
        return keys
    with open(failed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                k = doc.get('key')
                if k:
                    keys.add(k)
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


# =============================================================================
# LLM call
# =============================================================================

def extract_game_date(doc: Dict) -> str:
    """Extract a usable date string from the game record for temporal bounding."""
    date_str = doc.get('date', '')
    # Dates are like "1962.??.??" or "1962.06.15"
    if date_str:
        year = date_str.split('.')[0]
        if year.isdigit():
            return f"July 1969"
    return "July 1969"


def format_game_data(doc: Dict) -> str:
    """Format the game JSON for inclusion in the prompt."""
    return json.dumps(doc, ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    """Conservatively estimate token count for mixed chess/English text."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def call_llm(host: str, port: int, game_doc: Dict,
             prompt_index: int, *, verbose: bool = False) -> Dict:
    """Send a narrative-generation prompt to the LLM. Returns result dict."""
    url = f"http://{host}:{port}/v1/chat/completions"
    key = game_doc.get('key', 'unknown')

    game_data = format_game_data(game_doc)
    game_date = extract_game_date(game_doc)
    prompt_template = PROMPT_VARIATIONS[prompt_index % len(PROMPT_VARIATIONS)]
    user_content = prompt_template.format(game_data=game_data, game_date=game_date)

    input_chars = len(SYSTEM_PROMPT) + len(user_content)
    input_tokens_est = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(user_content)

    # Dynamically cap max_tokens so input + output fits the context window.
    # Reserve a small margin (64 tokens) for chat-template overhead.
    headroom = _context_window - input_tokens_est - 64
    if headroom < 256:
        msg = (f"Input too long for context window "
               f"(~{input_tokens_est} tokens est., "
               f"context={_context_window}, "
               f"input={input_chars} chars)")
        if verbose:
            log.debug("  [skip] %s — %s", key, msg)
        return {'error': msg, 'elapsed': 0.0}
    max_tokens = min(2048, headroom)

    if verbose:
        log.debug("  [req] %s — input=%d chars, ~%d tokens, "
                  "max_tokens=%d, prompt_variant=%d, endpoint=%s:%d",
                  key, input_chars, input_tokens_est,
                  max_tokens, prompt_index % len(PROMPT_VARIATIONS),
                  host, port)

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            elapsed = time.monotonic() - t0
            # Read the server error body for diagnostics
            try:
                err_body = resp.text[:500]
            except Exception:
                err_body = '(could not read response body)'
            msg = (f"HTTP {resp.status_code} from {host}:{port}")
            if verbose:
                log.debug("  [err] %s — %s — body: %s", key, msg, err_body)
            return {'error': f'{msg}: {err_body}',
                    'elapsed': elapsed}
        elapsed = time.monotonic() - t0

        data = resp.json()
        text = data['choices'][0]['message']['content'].strip()

        if not text or len(text) < MIN_RESPONSE_CHARS:
            if verbose:
                log.debug("  [short] %s — response only %d chars: %s",
                          key, len(text) if text else 0,
                          (text[:200] if text else '(empty)'))
            return {'error': f'Response too short ({len(text) if text else 0} chars)',
                    'raw': text[:300] if text else '',
                    'elapsed': elapsed}

        if verbose:
            log.debug("  [ok] %s — %d chars in %.1fs",
                      key, len(text), elapsed)

        return {
            'text': text,
            'elapsed': elapsed,
            'prompt_variant': prompt_index % len(PROMPT_VARIATIONS),
        }

    except requests.RequestException as e:
        elapsed = time.monotonic() - t0
        if verbose:
            log.debug("  [err] %s — request exception: %s", key, e)
        return {'error': f'Request failed: {e}',
                'elapsed': elapsed}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        elapsed = time.monotonic() - t0
        if verbose:
            log.debug("  [err] %s — parse error: %s", key, e)
        return {'error': f'Parse error: {e}',
                'elapsed': elapsed}


# =============================================================================
# Worker
# =============================================================================

def process_game(game_doc: Dict, endpoints: List[Tuple[str, int]],
                 prompt_index: int, *, retries: int = DEFAULT_RETRIES,
                 verbose: bool = False) -> Optional[Dict]:
    """Augment a single game, trying endpoints with retries."""
    key = game_doc.get('key', 'unknown')
    last_error = None

    for attempt in range(retries + 1):
        host, port = endpoints[attempt % len(endpoints)]
        result = call_llm(host, port, game_doc, prompt_index,
                          verbose=verbose)

        if 'error' not in result:
            # Build the augmented record
            return {
                'key': key,
                'white': game_doc.get('white', ''),
                'black': game_doc.get('black', ''),
                'date': game_doc.get('date', ''),
                'event': game_doc.get('event', ''),
                'eco': game_doc.get('eco', ''),
                'opening': game_doc.get('opening', ''),
                'result': game_doc.get('result', ''),
                'source_file': game_doc.get('source_file', ''),
                'prompt_variant': result['prompt_variant'],
                'text': result['text'],
                'length': len(result['text']),
            }

        last_error = result.get('error', 'unknown')
        if attempt < retries:
            time.sleep(2 ** attempt)  # exponential backoff

    log.warning("Failed to augment '%s' after %d attempt(s): %s",
                key, retries + 1, last_error)
    return None


# =============================================================================
# Main processing loop
# =============================================================================

def run_augmentation(args):
    """Main entry point for the augmentation pipeline."""

    log.info("Chess Game Narrative Augmentation")
    log.info("=" * 60)

    # ── Validate source corpus ──
    if not SOURCE_CORPUS.exists():
        log.error("Source corpus not found: %s", SOURCE_CORPUS)
        log.error("Run retrieve_chess_content.py first to build the chess corpus.")
        sys.exit(1)

    # ── Discover endpoints ──
    log.info("Discovering LLM endpoints...")
    endpoints = discover_endpoints(local_only=not args.use_remote)
    if not endpoints:
        log.error("No LLM endpoints available. Start a llama-server with a "
                  "large model (Nemotron-3-Nano-30B recommended).")
        log.error("  llm-swap /mnt/data/models/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf "
                  '"nemotron-3-nano-30b" 8192 --slots 4')
        sys.exit(1)
    log.info("Using %d endpoint(s)", len(endpoints))

    # ── Detect per-slot context size ──
    global _context_window
    _context_window = detect_context_size(endpoints)

    # ── Handle --reset ──
    if args.reset:
        for path in (OUTPUT_CORPUS, FAILED_CORPUS):
            if path.exists():
                path.unlink()
                log.info("Deleted %s", path)
        log.info("Reset complete — starting fresh")

    # ── Load progress ──
    completed_keys = load_completed_keys(OUTPUT_CORPUS)
    log.info("Already augmented: %d games", len(completed_keys))

    # ── Load failure tracking ──
    failed_keys = load_failed_keys(FAILED_CORPUS) - completed_keys
    if args.include_failed:
        log.info("--include-failed: reprocessing %d previously failed games",
                 len(failed_keys))
        skip_keys = completed_keys
    else:
        if failed_keys:
            log.info("Skipping %d previously failed games "
                     "(use --include-failed to reprocess)", len(failed_keys))
        skip_keys = completed_keys | failed_keys

    # ── Load source games ──
    log.info("Loading source corpus: %s", SOURCE_CORPUS)
    games_to_process = []
    total_source = 0
    with open(SOURCE_CORPUS) as f:
        for line in f:
            total_source += 1
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                key = doc.get('key')
                if key and key not in skip_keys:
                    games_to_process.append(doc)
            except json.JSONDecodeError:
                continue

    log.info("Source corpus: %d total games", total_source)
    log.info("Remaining to augment: %d games", len(games_to_process))

    if args.max_games and args.max_games < len(games_to_process):
        games_to_process = games_to_process[:args.max_games]
        log.info("Capped to %d games (--max-games)", args.max_games)

    if not games_to_process:
        log.info("Nothing to do — all games already augmented.")
        return

    # ── Ensure output directory ──
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    # ── Process games ──
    concurrency = args.concurrency
    log.info("Starting augmentation with concurrency=%d", concurrency)

    processed = 0
    errors = 0
    t_start = time.monotonic()
    t_last_progress = t_start

    # Open output in append mode
    out_file = None if args.dry_run else open(OUTPUT_CORPUS, 'a')
    fail_file = None if args.dry_run else open(FAILED_CORPUS, 'a')

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Determine prompt cycling
            prompt_counter = len(completed_keys)  # continue cycling from where we left off

            futures = {}
            batch_idx = 0

            for game_doc in games_to_process:
                if _shutdown_event.is_set():
                    break

                pi = args.prompt_index if args.prompt_index is not None else prompt_counter
                future = pool.submit(process_game, game_doc, endpoints, pi,
                                     retries=args.retries,
                                     verbose=args.verbose)
                futures[future] = game_doc
                prompt_counter += 1
                batch_idx += 1

                # Process completed futures to avoid unbounded memory
                if len(futures) >= concurrency * 4:
                    done = []
                    for f in list(futures.keys()):
                        if f.done():
                            done.append(f)
                    for f in done:
                        game = futures[f]
                        result = f.result()
                        del futures[f]
                        if result:
                            processed += 1
                            if out_file:
                                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                                if processed % 10 == 0:
                                    out_file.flush()
                            log.info("  [%d] %s → %d chars",
                                     processed, result['key'], result['length'])
                        else:
                            errors += 1
                            if fail_file:
                                fail_record = {
                                    'key': game.get('key', 'unknown'),
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                }
                                fail_file.write(json.dumps(fail_record) + '\n')
                                fail_file.flush()

                    # Progress report
                    now = time.monotonic()
                    if now - t_last_progress >= PROGRESS_INTERVAL:
                        elapsed = now - t_start
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = len(games_to_process) - batch_idx
                        eta = remaining / rate if rate > 0 else 0
                        log.info("Progress: %d/%d augmented (%.1f/sec), "
                                 "%d errors, ETA: %.0f min",
                                 processed, len(games_to_process), rate,
                                 errors, eta / 60)
                        t_last_progress = now

            # Cancel any futures that haven't started yet
            if _shutdown_event.is_set():
                for f in list(futures.keys()):
                    f.cancel()

            # Drain remaining futures (only those not cancelled)
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                game = futures[future]
                result = future.result()
                if result:
                    processed += 1
                    if out_file:
                        out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    log.info("  [%d] %s → %d chars",
                             processed, result['key'], result['length'])
                else:
                    errors += 1
                    if fail_file:
                        fail_record = {
                            'key': game.get('key', 'unknown'),
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                        }
                        fail_file.write(json.dumps(fail_record) + '\n')
                        fail_file.flush()

    finally:
        if out_file:
            out_file.flush()
            out_file.close()
        if fail_file:
            fail_file.flush()
            fail_file.close()

    # ── Summary ──
    elapsed = time.monotonic() - t_start
    log.info("")
    log.info("Augmentation complete")
    log.info("  Games processed: %d", processed)
    log.info("  Errors: %d", errors)
    log.info("  Total time: %.1f min", elapsed / 60)
    if processed > 0:
        log.info("  Average: %.1f sec/game", elapsed / processed)
    log.info("  Output: %s", OUTPUT_CORPUS)
    if args.dry_run:
        log.info("  (dry-run mode — no output written)")


# =============================================================================
# JSONL → readable file converter (for manual review)
# =============================================================================

_HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Augmented Chess Games — Review</title>
<style>
  body { font-family: Georgia, 'Times New Roman', serif; max-width: 960px;
         margin: 2em auto; padding: 0 1em; color: #222; background: #fafaf8; }
  h1 { border-bottom: 3px double #900; padding-bottom: .3em; color: #900; }
  .game { border: 1px solid #ccc; border-radius: 6px; padding: 1.2em;
          margin-bottom: 1.5em; background: #fff; }
  .game h2 { margin-top: 0; color: #333; font-size: 1.15em; }
  .meta { font-size: .85em; color: #666; margin-bottom: .8em; }
  .meta span { margin-right: 1.2em; }
  .narrative { white-space: pre-wrap; line-height: 1.6; }
  .toc { margin-bottom: 2em; }
  .toc a { text-decoration: none; color: #900; }
  .toc a:hover { text-decoration: underline; }
  footer { text-align: center; font-size: .8em; color: #999; margin-top: 3em; }
</style>
</head>
<body>
<h1>Augmented Chess Games &mdash; Review</h1>
"""

_HTML_FOOTER = """<footer>Generated by augment_chess_games.py &mdash; {timestamp}</footer>
</body></html>
"""


def convert_jsonl_to_review(fmt: str, max_games: Optional[int] = None):
    """Read augmented JSONL and write a human-readable review file.

    Args:
        fmt: 'html' or 'md'
        max_games: optional cap on number of games to include
    """
    if not OUTPUT_CORPUS.exists():
        log.error("Augmented corpus not found: %s", OUTPUT_CORPUS)
        log.error("Run augmentation first before converting.")
        sys.exit(1)

    # Load records
    records: List[Dict] = []
    with open(OUTPUT_CORPUS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if max_games is not None:
        records = records[:max_games]

    if not records:
        log.error("No records found in %s", OUTPUT_CORPUS)
        sys.exit(1)

    ext = 'html' if fmt == 'html' else 'md'
    out_path = OUTPUT_CORPUS.with_suffix(f'.review.{ext}')

    if fmt == 'html':
        _write_html(records, out_path)
    else:
        _write_markdown(records, out_path)

    log.info("Wrote %d games to %s", len(records), out_path)


def _write_html(records: List[Dict], out_path: Path):
    with open(out_path, 'w') as f:
        f.write(_HTML_HEADER)
        # Table of contents
        f.write('<div class="toc"><strong>Contents</strong> '
                f'({len(records)} games)<ol>\n')
        for i, rec in enumerate(records, 1):
            white = html_module.escape(rec.get('white', '?'))
            black = html_module.escape(rec.get('black', '?'))
            f.write(f'<li><a href="#game-{i}">{white} vs {black}</a></li>\n')
        f.write('</ol></div>\n')

        for i, rec in enumerate(records, 1):
            white = html_module.escape(rec.get('white', '?'))
            black = html_module.escape(rec.get('black', '?'))
            date = html_module.escape(rec.get('date', '?'))
            event = html_module.escape(rec.get('event', '?'))
            opening = html_module.escape(rec.get('opening', ''))
            eco = html_module.escape(rec.get('eco', ''))
            result = html_module.escape(rec.get('result', '?'))
            pv = rec.get('prompt_variant', '?')
            length = rec.get('length', 0)
            text = html_module.escape(rec.get('text', ''))

            f.write(f'<div class="game" id="game-{i}">\n')
            f.write(f'<h2>#{i}: {white} vs {black}</h2>\n')
            f.write('<div class="meta">')
            f.write(f'<span><b>Date:</b> {date}</span>')
            f.write(f'<span><b>Event:</b> {event}</span>')
            f.write(f'<span><b>Result:</b> {result}</span>')
            if opening:
                f.write(f'<span><b>Opening:</b> {opening}')
                if eco:
                    f.write(f' ({eco})')
                f.write('</span>')
            f.write(f'<span><b>Prompt:</b> v{pv}</span>')
            f.write(f'<span><b>Length:</b> {length:,} chars</span>')
            f.write('</div>\n')
            f.write(f'<div class="narrative">{text}</div>\n')
            f.write('</div>\n')

        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        f.write(_HTML_FOOTER.format(timestamp=timestamp))


def _write_markdown(records: List[Dict], out_path: Path):
    with open(out_path, 'w') as f:
        f.write(f'# Augmented Chess Games — Review\n\n')
        f.write(f'**{len(records)} games**\n\n---\n\n')

        for i, rec in enumerate(records, 1):
            white = rec.get('white', '?')
            black = rec.get('black', '?')
            date = rec.get('date', '?')
            event = rec.get('event', '?')
            opening = rec.get('opening', '')
            eco = rec.get('eco', '')
            result = rec.get('result', '?')
            pv = rec.get('prompt_variant', '?')
            length = rec.get('length', 0)
            text = rec.get('text', '')

            f.write(f'## #{i}: {white} vs {black}\n\n')
            f.write(f'| Field | Value |\n|-------|-------|\n')
            f.write(f'| Date | {date} |\n')
            f.write(f'| Event | {event} |\n')
            f.write(f'| Result | {result} |\n')
            if opening:
                opening_str = f'{opening} ({eco})' if eco else opening
                f.write(f'| Opening | {opening_str} |\n')
            f.write(f'| Prompt variant | {pv} |\n')
            f.write(f'| Length | {length:,} chars |\n')
            f.write(f'\n{text}\n\n---\n\n')


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Augment chess game corpus with narrative text using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Parallel workers per endpoint (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Stop after N games (default: all)')
    parser.add_argument('--prompt-index', type=int, default=None,
                        help='Use only this prompt variant (0-4); default: cycle all')
    parser.add_argument('--retries', type=int, default=DEFAULT_RETRIES,
                        help=f'Retry failed games N times (default: {DEFAULT_RETRIES})')
    parser.add_argument('--use-remote', action='store_true',
                        help='Also use remote LLM endpoint (default: local only)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Augment but do not write output')
    parser.add_argument('--include-failed', action='store_true',
                        help='Reprocess games that previously failed')
    parser.add_argument('--reset', action='store_true',
                        help='Delete existing output and failure tracking files, then start fresh')
    parser.add_argument('--verbose', action='store_true',
                        help='Log each game as it completes')
    parser.add_argument('--convert', choices=['html', 'md'],
                        metavar='FORMAT',
                        help='Convert augmented JSONL to a review file (html or md)')
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if args.convert:
        convert_jsonl_to_review(args.convert, max_games=args.max_games)
    else:
        run_augmentation(args)


if __name__ == '__main__':
    main()
