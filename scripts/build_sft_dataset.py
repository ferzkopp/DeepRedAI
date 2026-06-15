#!/usr/bin/env python3
"""
build_sft_dataset.py — Build a chat-format SFT corpus for Gemma fine-tuning.

Converts the same DeepRed corpus sources used by ``create_training_corpus.py``
into HuggingFace-chat-format JSONL (``{"messages": [{"role", "content"}, ...]}``)
suitable for TRL ``SFTTrainer`` (see ``train_deepred_gemma.py``).

Sources (lexical templating only — no LLM augmentation here):
  year_topics            "What were notable events in {year}?"  →  summary
  gutenberg              "Continue this passage from {title} by {author}…"
                           →  rest of the chunk
  augmented_chess_games  "Narrate this chess game (key={key})…"  →  narrative
  chess_games            "Describe this chess game in PGN form."  →  notation
  chess_books            "Continue this passage from {title} by {author}…"
                           →  rest of the chunk
  wikipedia_articles     "Tell me about {title}."  →  intro paragraphs
                           (PostgreSQL — optional)
  retain                 Pre-cutoff factual Q&A from the temporal generator
                           (reuses $WIKI_DATA/datasets/retain/*.jsonl)
  unlearn                Post-cutoff refusal Q&A ("I don't know")
                           (reuses $WIKI_DATA/datasets/unlearn/*.jsonl)

Environment variables (honour ``deepred-env.sh``):
  DEEPRED_ROOT     Base data directory (default: /mnt/data)
  WIKI_DATA        $DEEPRED_ROOT/wikipedia
  GUTENBERG_DATA   $DEEPRED_ROOT/gutenberg
  CHESS_DATA       $DEEPRED_ROOT/chess
  PG_HOST / PG_PORT / PG_USER / PG_PASSWORD / PG_DATABASE

Usage:
  # Smoke test (50 samples per source)
  python3 scripts/build_sft_dataset.py \\
      --sources year_topics,augmented_chess_games \\
      --max-samples-per-source 50 --tag smoke

  # Full build for SFT training
  python3 scripts/build_sft_dataset.py --tag v1

  # Add wikipedia (requires PostgreSQL)
  python3 scripts/build_sft_dataset.py \\
      --sources wikipedia_articles,year_topics,gutenberg,augmented_chess_games,chess_books \\
      --tag v1
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# ── Optional dependencies ────────────────────────────────────────────────

try:
    import psycopg2  # noqa: F401
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


TEMPORAL_CUTOFF_YEAR = 1969

ALL_SOURCES = [
    'wikipedia_articles',
    'retain',
    'unlearn',
    'year_topics',
    'gutenberg',
    'chess_games',
    'augmented_chess_games',
    'chess_books',
]


# ── Helpers ──────────────────────────────────────────────────────────────

def env_paths():
    """Resolve data paths from environment variables."""
    root = os.environ.get('DEEPRED_ROOT', '/mnt/data')
    return {
        'root': root,
        'wiki_data':      os.environ.get('WIKI_DATA',      f"{root}/wikipedia"),
        'gutenberg_data': os.environ.get('GUTENBERG_DATA', f"{root}/gutenberg"),
        'chess_data':     os.environ.get('CHESS_DATA',     f"{root}/chess"),
    }


def pg_config():
    return {
        'host':     os.environ.get('PG_HOST', 'localhost'),
        'port':     int(os.environ.get('PG_PORT', 5432)),
        'database': os.environ.get('PG_DATABASE', 'wikidb'),
        'user':     os.environ.get('PG_USER', 'wiki'),
        'password': os.environ.get('PG_PASSWORD', 'wiki'),
    }


_WS_RE = re.compile(r'[^\S\n]+')
_BLANK_LINES_RE = re.compile(r'\n{3,}')


def clean_text(text):
    if not text:
        return ''
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _WS_RE.sub(' ', text)
    text = _BLANK_LINES_RE.sub('\n\n', text)
    return text.strip()


def truncate(text, max_chars):
    """Truncate to *max_chars* preserving paragraph boundary when possible."""
    if not text or len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    # Prefer to cut at the last paragraph break in the snippet
    cut = snippet.rfind('\n\n')
    if cut > max_chars // 2:
        return snippet[:cut].rstrip()
    # Fallback: cut at last sentence
    cut = max(snippet.rfind('. '), snippet.rfind('.\n'))
    if cut > max_chars // 2:
        return snippet[:cut + 1].rstrip()
    return snippet.rstrip()


def file_sha256(path, max_bytes=64 * 1024 * 1024):
    """SHA-256 of the first *max_bytes* of a file (for manifest provenance)."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except OSError:
        return None


def make_pair(user, assistant):
    """Return a chat-format example, or None if either side is empty."""
    user = (user or '').strip()
    assistant = (assistant or '').strip()
    if not user or not assistant:
        return None
    return {
        'messages': [
            {'role': 'user',      'content': user},
            {'role': 'assistant', 'content': assistant},
        ]
    }


def parse_source_limits(spec):
    """Parse ``source=N,source2=N`` into a dict of per-source caps."""
    limits = {}
    if not spec:
        return limits
    for raw_part in spec.split(','):
        part = raw_part.strip()
        if not part:
            continue
        if '=' in part:
            key, value = part.split('=', 1)
        elif ':' in part:
            key, value = part.split(':', 1)
        else:
            raise ValueError(
                f"Invalid source limit '{part}' (expected source=N)")
        key = key.strip()
        if key not in ALL_SOURCES:
            raise ValueError(f"Unknown source in --source-limits: {key}")
        try:
            limit = int(value.strip())
        except ValueError as e:
            raise ValueError(f"Invalid limit for {key}: {value}") from e
        if limit < 0:
            raise ValueError(f"Limit for {key} must be >= 0")
        limits[key] = limit
    return limits


def _source_stats(pairs):
    """Return per-source example and character counts for manifest auditing."""
    out = {}
    for pair in pairs:
        source = pair.get('_source', 'unknown')
        entry = out.setdefault(source, {
            'examples': 0,
            'user_chars': 0,
            'assistant_chars': 0,
        })
        entry['examples'] += 1
        for msg in pair.get('messages', []):
            content_len = len(msg.get('content') or '')
            if msg.get('role') == 'user':
                entry['user_chars'] += content_len
            elif msg.get('role') == 'assistant':
                entry['assistant_chars'] += content_len
    return dict(sorted(out.items()))


# ── Per-source builders (each yields dicts with a 'messages' key) ───────

def build_year_topics(env, limit, max_chars):
    """One pair per year ≤ 1969."""
    topics_dir = Path(env['wiki_data']) / 'topics'
    if not topics_dir.exists():
        return
    files = sorted(
        f for f in topics_dir.glob('year_topics_*.json')
        if _safe_year(f) is not None and _safe_year(f) <= TEMPORAL_CUTOFF_YEAR
    )
    n = 0
    for f in files:
        if limit and n >= limit:
            break
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        year = data.get('year')
        topics = data.get('topics') or []
        if not year or not topics:
            continue
        lines = []
        for t in topics:
            dt = (t.get('date_text') or '').strip()
            tx = (t.get('topic') or '').strip()
            if not tx:
                continue
            lines.append(f"{dt}: {tx}" if dt else tx)
        if not lines:
            continue
        body = truncate('\n'.join(lines), max_chars)
        prompts = [
            f"What were the notable events of the year {year}?",
            f"Summarise the historical events of {year}.",
            f"Tell me what happened in {year}.",
        ]
        pair = make_pair(random.choice(prompts), body)
        if pair:
            yield pair
            n += 1


def _safe_year(path):
    try:
        return int(path.stem.split('_')[-1])
    except (ValueError, IndexError):
        return None


def build_gutenberg(env, limit, max_chars):
    """Passage-continuation pairs from Project Gutenberg chunks."""
    path = Path(env['gutenberg_data']) / 'corpus' / 'gutenberg_corpus.jsonl'
    if not path.exists():
        return
    yield from _passage_continuation(
        path, limit, max_chars,
        prompt_fn=lambda doc, prefix: _continue_prompt(
            doc.get('title', ''), doc.get('author', ''), prefix))


def build_chess_books(env, limit, max_chars):
    path = Path(env['chess_data']) / 'corpus' / 'chess_archive_books.jsonl'
    if not path.exists():
        return
    yield from _passage_continuation(
        path, limit, max_chars,
        prompt_fn=lambda doc, prefix: _continue_prompt(
            doc.get('title', ''), doc.get('author', ''), prefix))


def _continue_prompt(title, author, prefix):
    src = title.strip() if title else 'this text'
    if author:
        src = f"{src} by {author.strip()}"
    return f"Continue the following passage from {src}:\n\n{prefix}"


def _passage_continuation(jsonl_path, limit, max_chars, prompt_fn):
    """Generic helper: split each doc's text into prefix (user) + rest (assistant)."""
    n = 0
    prefix_chars = min(512, max_chars // 4)
    rest_chars = max_chars
    with open(jsonl_path) as f:
        for line in f:
            if limit and n >= limit:
                break
            try:
                doc = json.loads(line)
            except Exception:
                continue
            text = clean_text(doc.get('text', ''))
            if len(text) < prefix_chars + 256:
                continue
            prefix = text[:prefix_chars].rstrip()
            rest = text[len(prefix):].lstrip()
            if not rest:
                continue
            rest = truncate(rest, rest_chars)
            pair = make_pair(prompt_fn(doc, prefix), rest)
            if pair:
                yield pair
                n += 1


def build_augmented_chess_games(env, limit, max_chars):
    """LLM-narrative + raw PGN narration pairs."""
    path = Path(env['chess_data']) / 'corpus' / 'augmented_chess_games.jsonl'
    if not path.exists():
        return
    n = 0
    with open(path) as f:
        for line in f:
            if limit and n >= limit:
                break
            try:
                doc = json.loads(line)
            except Exception:
                continue
            narrative = clean_text(doc.get('text', ''))
            if not narrative:
                continue
            key = (doc.get('key') or '').strip()
            white, black, year, event = _parse_chess_key(key)
            descriptor = _chess_descriptor(white, black, year, event)
            user = (f"Narrate the following chess game{descriptor} "
                    f"as if commentating for an audience.")
            pair = make_pair(user, truncate(narrative, max_chars))
            if pair:
                yield pair
                n += 1


def build_chess_games(env, limit, max_chars):
    """Raw PGN — used as 'describe this game in standard chess notation'."""
    path = Path(env['chess_data']) / 'corpus' / 'chess_games.jsonl'
    if not path.exists():
        return
    n = 0
    with open(path) as f:
        for line in f:
            if limit and n >= limit:
                break
            try:
                doc = json.loads(line)
            except Exception:
                continue
            pgn = clean_text(doc.get('text', ''))
            if not pgn:
                continue
            key = (doc.get('key') or '').strip()
            white, black, year, event = _parse_chess_key(key)
            descriptor = _chess_descriptor(white, black, year, event)
            user = (f"Provide the PGN notation for the chess game{descriptor}.")
            pair = make_pair(user, truncate(pgn, max_chars))
            if pair:
                yield pair
                n += 1


def _parse_chess_key(key):
    """Best-effort parse of common 'White_vs_Black_Event_Year' style keys."""
    if not key:
        return '', '', '', ''
    parts = re.split(r'[_|]', key)
    year = ''
    for p in reversed(parts):
        if re.fullmatch(r'\d{4}', p):
            year = p
            break
    if 'vs' in [p.lower() for p in parts]:
        idx = [p.lower() for p in parts].index('vs')
        white = parts[idx - 1] if idx > 0 else ''
        black = parts[idx + 1] if idx + 1 < len(parts) else ''
    else:
        white = parts[0] if parts else ''
        black = parts[1] if len(parts) > 1 else ''
    event = ''
    return white.replace('-', ' '), black.replace('-', ' '), year, event


def _chess_descriptor(white, black, year, event):
    bits = []
    if white and black:
        bits.append(f" between {white} and {black}")
    elif white:
        bits.append(f" featuring {white}")
    if year:
        bits.append(f" from {year}")
    if event:
        bits.append(f" ({event})")
    return ''.join(bits)


def build_wikipedia_articles(env, limit, max_chars):
    """Pre-1969 wiki articles from PostgreSQL.  Requires psycopg2."""
    if not HAS_PSYCOPG2:
        print("  [wikipedia_articles] psycopg2 not installed — skipping")
        return
    try:
        conn = psycopg2.connect(**pg_config(), connect_timeout=5)
    except Exception as e:
        print(f"  [wikipedia_articles] DB connect failed: {e}")
        return
    cur = conn.cursor('sft_reader')
    cur.itersize = 500
    sql = ("SELECT title, content FROM articles "
           "WHERE temporal_classification = 'O' ORDER BY id")
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    n = 0
    try:
        for title, content in cur:
            if limit and n >= limit:
                break
            if not title or not content:
                continue
            body = truncate(clean_text(content), max_chars)
            user = random.choice([
                f"Tell me about {title}.",
                f"Give me an overview of {title}.",
                f"What can you tell me about {title}?",
            ])
            pair = make_pair(user, body)
            if pair:
                yield pair
                n += 1
    finally:
        cur.close()
        conn.close()


def _instruction_pairs(files, limit, max_chars):
    """Load instruction/output pairs from legacy JSONL files, shuffle, then cap.

    The temporal generator stores Q&A as ``{"instruction", "output",
    "metadata"}`` and *appends* new pairs (e.g. modern-event refusals) to the
    END of these files.  Shuffling before applying *limit* ensures appended
    pairs are represented when a cap is in force, rather than always being the
    first to be dropped.
    """
    pairs = []
    for path in files:
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except Exception:
                    continue
                instruction = clean_text(doc.get('instruction', ''))
                output = clean_text(doc.get('output', ''))
                if not instruction or not output:
                    continue
                pair = make_pair(
                    truncate(instruction, max_chars),
                    truncate(output, max_chars))
                if pair:
                    pairs.append(pair)
    random.shuffle(pairs)
    if limit:
        pairs = pairs[:limit]
    yield from pairs


def build_retain(env, limit, max_chars):
    """Pre-cutoff factual Q&A (retain knowledge).

    Reuses the existing temporal datasets under
    ``$WIKI_DATA/datasets/retain/`` (both train and val files) so the model
    keeps answering pre-cutoff questions accurately.
    """
    base = Path(env['wiki_data']) / 'datasets' / 'retain'
    files = [base / 'retain_train.jsonl', base / 'retain_val.jsonl']
    yield from _instruction_pairs(files, limit, max_chars)


def build_unlearn(env, limit, max_chars):
    """Post-cutoff refusal Q&A (unlearn knowledge).

    Reuses the existing temporal datasets under
    ``$WIKI_DATA/datasets/unlearn/`` (both train and val files); the assistant
    side is a standardized refusal so the model declines post-cutoff topics.
    """
    base = Path(env['wiki_data']) / 'datasets' / 'unlearn'
    files = [base / 'unlearn_train.jsonl', base / 'unlearn_val.jsonl']
    yield from _instruction_pairs(files, limit, max_chars)


SOURCE_BUILDERS = {
    'year_topics':           build_year_topics,
    'gutenberg':             build_gutenberg,
    'augmented_chess_games': build_augmented_chess_games,
    'chess_games':           build_chess_games,
    'chess_books':           build_chess_books,
    'wikipedia_articles':    build_wikipedia_articles,
    'retain':                build_retain,
    'unlearn':               build_unlearn,
}


# ── Main pipeline ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--sources', default='year_topics,gutenberg,'
        'augmented_chess_games,chess_books',
        help='Comma-separated source names. Default excludes raw chess_games '
             'and wikipedia_articles. Available: ' + ', '.join(ALL_SOURCES))
    parser.add_argument(
        '--max-samples-per-source', type=int, default=0,
        help='Cap samples per source (0 = unlimited).')
    parser.add_argument(
        '--source-limits', default='',
        help='Per-source caps as source=N,source2=N. Overrides '
             '--max-samples-per-source for listed sources. Use 0 to skip a '
             'source while keeping it in --sources for manifest provenance.')
    parser.add_argument(
        '--max-chars', type=int, default=4096,
        help='Hard cap on per-message character length (default 4096).')
    parser.add_argument(
        '--val-fraction', type=float, default=0.05,
        help='Validation split fraction (default 0.05).')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Shuffle seed (default 42).')
    parser.add_argument(
        '--tag', default='v1',
        help='Run tag — appears in output directory (default "v1").')
    parser.add_argument(
        '--output-dir', default=None,
        help='Override default output dir '
             '(default: $DEEPRED_ROOT/sft_corpus/<tag>).')
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite an existing output directory.')
    args = parser.parse_args()

    env = env_paths()
    sources = [s.strip() for s in args.sources.split(',') if s.strip()]
    bad = [s for s in sources if s not in SOURCE_BUILDERS]
    if bad:
        parser.error(f"Unknown sources: {', '.join(bad)}. "
                     f"Available: {', '.join(ALL_SOURCES)}")
    try:
        source_limits = parse_source_limits(args.source_limits)
    except ValueError as e:
        parser.error(str(e))

    out_dir = Path(args.output_dir or f"{env['root']}/sft_corpus/{args.tag}")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            print(f"ERROR: {out_dir} is not empty. Use --force to overwrite.",
                  file=sys.stderr)
            sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    t_start = time.time()
    print(f"Building SFT dataset at {out_dir}")
    print(f"  sources : {', '.join(sources)}")
    print(f"  cap     : {args.max_samples_per_source or 'unlimited'} / source")
    if source_limits:
        pretty_limits = ', '.join(
            f"{source}={limit}" for source, limit in sorted(source_limits.items()))
        print(f"  limits  : {pretty_limits}")
    print(f"  maxlen  : {args.max_chars} chars / message")
    print(f"  val     : {args.val_fraction:.0%}")
    print()

    all_pairs = []
    source_counts = {}
    for src in sources:
        builder = SOURCE_BUILDERS[src]
        if src in source_limits:
            limit = source_limits[src]
        else:
            limit = args.max_samples_per_source or None
        if limit == 0:
            source_counts[src] = 0
            print(f"  [{src}] skipped by source limit")
            continue
        t0 = time.time()
        count = 0
        print(f"  [{src}] reading…", flush=True)
        for pair in builder(env, limit, args.max_chars):
            pair['_source'] = src
            all_pairs.append(pair)
            count += 1
        source_counts[src] = count
        print(f"  [{src}] {count:,} pairs  ({time.time() - t0:.1f}s)")

    if not all_pairs:
        print("ERROR: no pairs produced — check that source data is present.",
              file=sys.stderr)
        sys.exit(2)

    print()
    print(f"Shuffling {len(all_pairs):,} pairs (seed={args.seed})…")
    random.shuffle(all_pairs)

    n_val = max(1, int(len(all_pairs) * args.val_fraction))
    val = all_pairs[:n_val]
    train = all_pairs[n_val:]

    split_source_counts = {
        'train': {k: v['examples'] for k, v in _source_stats(train).items()},
        'val': {k: v['examples'] for k, v in _source_stats(val).items()},
    }
    split_source_chars = {
        'train': _source_stats(train),
        'val': _source_stats(val),
    }

    train_path = out_dir / 'train.jsonl'
    val_path = out_dir / 'val.jsonl'
    _write_jsonl(train, train_path)
    _write_jsonl(val, val_path)

    manifest = {
        'version': 1,
        'tag': args.tag,
        'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'sources': sources,
        'source_counts': source_counts,
        'split_source_counts': split_source_counts,
        'split_source_chars': split_source_chars,
        'max_samples_per_source': args.max_samples_per_source,
        'source_limits': source_limits,
        'max_chars': args.max_chars,
        'val_fraction': args.val_fraction,
        'seed': args.seed,
        'totals': {
            'pairs': len(all_pairs),
            'train': len(train),
            'val': len(val),
        },
        'paths': {
            'train': str(train_path),
            'val':   str(val_path),
        },
        'env': {
            'WIKI_DATA':      env['wiki_data'],
            'GUTENBERG_DATA': env['gutenberg_data'],
            'CHESS_DATA':     env['chess_data'],
        },
        'source_files_sha256': _source_file_hashes(env, sources),
    }
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  train : {len(train):,}  →  {train_path}")
    print(f"  val   : {len(val):,}  →  {val_path}")
    print(f"  manifest : {out_dir / 'manifest.json'}")


def _write_jsonl(pairs, path):
    with open(path, 'w', encoding='utf-8') as f:
        for p in pairs:
            # strip private fields
            clean = {k: v for k, v in p.items() if not k.startswith('_')}
            f.write(json.dumps(clean, ensure_ascii=False) + '\n')


def _source_file_hashes(env, sources):
    files = {
        'gutenberg':             Path(env['gutenberg_data']) / 'corpus' / 'gutenberg_corpus.jsonl',
        'chess_games':           Path(env['chess_data']) / 'corpus' / 'chess_games.jsonl',
        'augmented_chess_games': Path(env['chess_data']) / 'corpus' / 'augmented_chess_games.jsonl',
        'chess_books':           Path(env['chess_data']) / 'corpus' / 'chess_archive_books.jsonl',
        'retain':                Path(env['wiki_data']) / 'datasets' / 'retain' / 'retain_train.jsonl',
        'unlearn':               Path(env['wiki_data']) / 'datasets' / 'unlearn' / 'unlearn_train.jsonl',
    }
    out = {}
    for src in sources:
        p = files.get(src)
        if p and p.exists():
            out[src] = {'path': str(p), 'sha256_prefix': file_sha256(p)}
    return out


if __name__ == '__main__':
    main()
