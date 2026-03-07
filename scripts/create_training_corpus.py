#!/usr/bin/env python3
"""
Create Training Corpus — Tokenize and prepare data for continued pre-training.

Combines five data sources into a shuffled, tokenized binary corpus:

  wikipedia_articles   Pre-1969 articles from PostgreSQL  (~1.4B tokens)
  year_topics          Historical event summaries (JSON)  (~5M tokens)
  gutenberg            Project Gutenberg books (JSONL)    (~125M tokens)
  chess_games          Pre-1969 chess game narratives      (~53M tokens)
  chess_books          Internet Archive chess books        (~1M tokens)

Output is packed uint16 binary files containing 2048-token sequences,
ready for training frameworks (nanoGPT, LitGPT, torchtune).

Processing is incremental: you can tokenize 1% of the corpus for testing,
then expand to 10% or 100% without re-processing already-tokenized data.
Progress is tracked in a manifest.json file.

Environment Variables:
  DEEPRED_ROOT        Base data directory (default: /mnt/data)
  WIKI_DATA           Wikipedia data dir (default: $DEEPRED_ROOT/wikipedia)
  GUTENBERG_DATA      Gutenberg data dir (default: $DEEPRED_ROOT/gutenberg)
  CHESS_DATA          Chess data dir (default: $DEEPRED_ROOT/chess)
  PG_HOST / PG_PORT   PostgreSQL host and port (default: localhost:5432)
  PG_USER / PG_PASSWORD  DB credentials (default: wiki/wiki)
  PG_DATABASE         Database name (default: wikidb)

Usage:
  # Download the tokenizer
  python3 create_training_corpus.py --download-tokenizer

  # Show available sources and estimated sizes
  python3 create_training_corpus.py --info

  # Process 1% of all sources (quick test, ~1 minute)
  python3 create_training_corpus.py --percent 1

  # Expand to 100% (only processes the remaining 99%)
  python3 create_training_corpus.py --percent 100

  # Only Wikipedia and Gutenberg
  python3 create_training_corpus.py --sources wikipedia_articles,gutenberg --percent 100

  # Finalize: chunk into 2048-token sequences, shuffle, split train/val
  python3 create_training_corpus.py --finalize

  # Switch to SmolLM2-360M tokenizer for dev run
  python3 create_training_corpus.py --tokenizer SmolLM2-360M --download-tokenizer
  python3 create_training_corpus.py --tokenizer SmolLM2-360M --percent 100

  # Show current progress
  python3 create_training_corpus.py --status
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

# Must be set before tokenizers import to enable Rust-level parallelism
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')

import numpy as np

# ── Optional dependencies ────────────────────────────────────────────────

try:
    from tokenizers import Tokenizer as HFTokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ── Constants ────────────────────────────────────────────────────────────

TEMPORAL_CUTOFF_YEAR = 1969

# Shard sizing: ~100 MB per shard file (50M uint16 tokens × 2 bytes)
SHARD_MAX_TOKENS = 50_000_000

DEFAULT_SEQ_LENGTH = 2048
DEFAULT_VAL_RATIO = 0.01
SHUFFLE_SEED = 42

# Batch sizes for DB reads and tokenizer calls
DB_BATCH_SIZE = 1000
TOKENIZE_BATCH_SIZE = 512

TOKENIZER_PRESETS = {
    'TinyLlama-1.1B': {
        'hf_repo': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'vocab_size': 32000,
        'files': [
            'tokenizer.model', 'tokenizer.json',
            'tokenizer_config.json', 'special_tokens_map.json',
        ],
        'eos_candidates': ['</s>'],
    },
    'SmolLM2-360M': {
        'hf_repo': 'HuggingFaceTB/SmolLM2-360M',
        'vocab_size': 49152,
        'files': [
            'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json',
        ],
        'eos_candidates': ['<|endoftext|>', '</s>'],
    },
}

ALL_SOURCES = [
    'wikipedia_articles',
    'year_topics',
    'gutenberg',
    'chess_games',
    'chess_books',
]

SOURCE_INFO = {
    'wikipedia_articles': {
        'description': 'Pre-1969 Wikipedia articles from PostgreSQL (temporal_classification=O)',
        'type': 'database',
        'estimated_tokens': '~1.4B',
    },
    'year_topics': {
        'description': 'Year-by-year historical event summaries, years 151–1969 (JSON files)',
        'type': 'json_files',
        'estimated_tokens': '~5M',
    },
    'gutenberg': {
        'description': 'Project Gutenberg books — 766 public-domain titles (JSONL)',
        'type': 'jsonl',
        'estimated_tokens': '~125M',
    },
    'chess_games': {
        'description': 'Pre-1969 chess game narratives — 356K games (JSONL)',
        'type': 'jsonl',
        'estimated_tokens': '~53M',
    },
    'chess_books': {
        'description': 'Internet Archive chess reference books — 10 titles (JSONL)',
        'type': 'jsonl',
        'estimated_tokens': '~1M',
    },
}


# ── Utility ──────────────────────────────────────────────────────────────

def fmt_tokens(n):
    """Format token count with suffix."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_bytes(n):
    """Format byte count with unit."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_duration(seconds):
    """Format elapsed seconds."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    h, rem = divmod(seconds, 3600)
    m = rem / 60
    return f"{int(h)}h {m:.0f}m"


def banner(text, char='=', width=60):
    """Print a section banner."""
    print(char * width)
    print(text)
    print(char * width)


def clean_text(text):
    """Basic text normalization for training data."""
    if not text:
        return ''
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse runs of whitespace on a single line (preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def count_file_lines(path):
    """Count lines in a file efficiently."""
    count = 0
    with open(path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def ensure_dir(path):
    """Create directory tree; exit on failure."""
    os.makedirs(path, exist_ok=True)
    if not os.access(path, os.W_OK):
        print(f"Error: Directory not writable: {path}")
        sys.exit(1)


# ── Database ─────────────────────────────────────────────────────────────

def get_db_config():
    """Return PostgreSQL connection kwargs from environment."""
    return {
        'host': os.environ.get('PG_HOST', 'localhost'),
        'port': int(os.environ.get('PG_PORT', 5432)),
        'database': os.environ.get('PG_DATABASE', 'wikidb'),
        'user': os.environ.get('PG_USER', 'wiki'),
        'password': os.environ.get('PG_PASSWORD', 'wiki'),
    }


def test_db():
    """Quick DB connectivity check.  Returns (ok, message)."""
    if not HAS_PSYCOPG2:
        return False, "psycopg2 not installed"
    try:
        conn = psycopg2.connect(**get_db_config(), connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ── Manifest ─────────────────────────────────────────────────────────────

def new_manifest():
    """Return a blank manifest dict."""
    return {
        'version': 1,
        'tokenizer': None,
        'vocab_size': None,
        'eos_id': None,
        'seq_length': DEFAULT_SEQ_LENGTH,
        'dtype': 'uint16',
        'sources': {},
        'total_tokens': 0,
        'finalized': False,
        'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'updated': None,
    }


def load_manifest(path):
    """Load manifest from disk, or return a fresh one."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return new_manifest()


def save_manifest(manifest, path):
    """Atomically write manifest to disk."""
    manifest['updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest['total_tokens'] = sum(
        s.get('token_count', 0) for s in manifest['sources'].values()
    )
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(path)


def get_source_state(manifest, source_name):
    """Return the source tracking dict, creating it if needed."""
    if source_name not in manifest['sources']:
        manifest['sources'][source_name] = {
            'total_available': 0,
            'processed_count': 0,
            'token_count': 0,
            'shard_files': [],
        }
    return manifest['sources'][source_name]


# ── Tokenizer download & loading ─────────────────────────────────────────

def download_tokenizer(tokenizer_name, tokenizer_dir):
    """Download tokenizer files from HuggingFace Hub."""
    if tokenizer_name not in TOKENIZER_PRESETS:
        print(f"Error: Unknown tokenizer preset '{tokenizer_name}'")
        print(f"  Available: {', '.join(TOKENIZER_PRESETS)}")
        sys.exit(1)

    preset = TOKENIZER_PRESETS[tokenizer_name]
    repo_id = preset['hf_repo']
    ensure_dir(tokenizer_dir)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        print()
        print("Or download manually:")
        for fname in preset['files']:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"
            print(f"  wget -P {tokenizer_dir} '{url}'")
        sys.exit(1)

    print(f"Downloading tokenizer: {tokenizer_name}")
    print(f"  Repository : {repo_id}")
    print(f"  Target dir : {tokenizer_dir}")

    for fname in preset['files']:
        target = Path(tokenizer_dir) / fname
        if target.exists():
            print(f"  [skip] {fname}")
            continue
        try:
            hf_hub_download(repo_id=repo_id, filename=fname,
                            local_dir=str(tokenizer_dir))
            print(f"  [done] {fname}")
        except Exception as e:
            print(f"  [warn] {fname}: {e}")

    print("  Download complete")


def load_tokenizer(tokenizer_dir):
    """
    Load a tokenizer from a local directory.

    Returns (tokenizer_obj, eos_id, vocab_size, backend_name).
    Prefers the fast ``tokenizers`` Rust library; falls back to sentencepiece.
    """
    tok_json  = Path(tokenizer_dir) / 'tokenizer.json'
    tok_model = Path(tokenizer_dir) / 'tokenizer.model'

    # ── Fast tokenizer (Rust / HuggingFace tokenizers library) ──
    if HAS_TOKENIZERS and tok_json.exists():
        tok = HFTokenizer.from_file(str(tok_json))
        vocab_size = tok.get_vocab_size()
        eos_id = None
        for candidate in ('</s>', '<|endoftext|>', '<eos>'):
            eid = tok.token_to_id(candidate)
            if eid is not None:
                eos_id = eid
                break
        if eos_id is None:
            eos_id = vocab_size - 1
        return tok, eos_id, vocab_size, 'tokenizers'

    # ── Fallback: sentencepiece ──
    if HAS_SENTENCEPIECE and tok_model.exists():
        sp = spm.SentencePieceProcessor(model_file=str(tok_model))
        vocab_size = sp.get_piece_size()
        eos_id = sp.eos_id()
        return sp, eos_id, vocab_size, 'sentencepiece'

    # ── Nothing worked ──
    if not tok_json.exists() and not tok_model.exists():
        print(f"Error: No tokenizer files in {tokenizer_dir}")
        print(f"  Run with --download-tokenizer first")
    else:
        print("Error: Install 'tokenizers' or 'sentencepiece' to load the tokenizer")
        print("  pip install tokenizers sentencepiece")
    sys.exit(1)


def encode_batch(tokenizer, texts, backend):
    """Tokenize a list of strings.  Returns list[list[int]]."""
    if backend == 'tokenizers':
        outputs = tokenizer.encode_batch(texts, add_special_tokens=False)
        return [o.ids for o in outputs]
    else:  # sentencepiece
        # sp.encode() accepts a list and returns list-of-lists
        return tokenizer.encode(texts)


# ── Source counting ──────────────────────────────────────────────────────

def count_source(source_name, env):
    """Return how many items are available for *source_name*."""

    if source_name == 'wikipedia_articles':
        if not HAS_PSYCOPG2:
            return 0
        try:
            conn = psycopg2.connect(**get_db_config(), connect_timeout=5)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM articles "
                "WHERE temporal_classification = 'O'"
            )
            n = cur.fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0

    if source_name == 'year_topics':
        topics_dir = Path(env['wiki_data']) / 'topics'
        if not topics_dir.exists():
            return 0
        n = 0
        for f in topics_dir.glob('year_topics_*.json'):
            try:
                year = int(f.stem.split('_')[-1])
                if year <= TEMPORAL_CUTOFF_YEAR:
                    n += 1
            except ValueError:
                pass
        return n

    if source_name == 'gutenberg':
        p = Path(env['gutenberg_data']) / 'corpus' / 'gutenberg_corpus.jsonl'
        return count_file_lines(p) if p.exists() else 0

    if source_name == 'chess_games':
        p = Path(env['chess_data']) / 'corpus' / 'chess_games.jsonl'
        return count_file_lines(p) if p.exists() else 0

    if source_name == 'chess_books':
        p = Path(env['chess_data']) / 'corpus' / 'chess_archive_books.jsonl'
        return count_file_lines(p) if p.exists() else 0

    return 0


# ── Source readers (generators over document texts) ──────────────────────

def read_wikipedia_articles(env, offset, limit):
    """
    Yield cleaned article texts from PostgreSQL.

    Uses a server-side cursor for efficient streaming.
    Each article is formatted as ``Title\\n\\nContent``.
    """
    conn = psycopg2.connect(**get_db_config())
    cur = conn.cursor('training_corpus_reader')
    cur.itersize = DB_BATCH_SIZE
    cur.execute(
        "SELECT id, title, content FROM articles "
        "WHERE temporal_classification = 'O' "
        "ORDER BY id "
        "OFFSET %s LIMIT %s",
        (offset, limit),
    )
    for _id, title, content in cur:
        if content:
            yield f"{title}\n\n{clean_text(content)}"
    cur.close()
    conn.close()


def read_year_topics(env, offset, limit):
    """
    Yield one document per year from topic JSON files (years ≤ 1969).

    Each document lists the year's events in chronological order:
        Events of the year 1960
        January 1: Cameroon becomes independent from France.
        ...
    """
    topics_dir = Path(env['wiki_data']) / 'topics'
    files = sorted(
        f for f in topics_dir.glob('year_topics_*.json')
        if int(f.stem.split('_')[-1]) <= TEMPORAL_CUTOFF_YEAR
    )

    for f in files[offset:offset + limit]:
        try:
            data = json.loads(f.read_text())
            year = data['year']
            topics = data.get('topics', [])
            if not topics:
                continue
            lines = [f"Events of the year {year}\n"]
            for t in topics:
                dt = t.get('date_text', '')
                tx = t.get('topic', '')
                if dt and tx:
                    lines.append(f"{dt}: {tx}")
                elif tx:
                    lines.append(tx)
            yield '\n'.join(lines)
        except Exception:
            continue


def _read_jsonl(path, offset, limit, formatter):
    """Generic JSONL reader — yields formatted text strings."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if i >= offset + limit:
                break
            try:
                doc = json.loads(line)
                text = formatter(doc)
                if text:
                    yield text
            except Exception:
                continue


def _fmt_gutenberg(doc):
    title  = doc.get('title', '')
    author = doc.get('author', '')
    text   = doc.get('text', '')
    if not text:
        return None
    header = title
    if author:
        header += f" by {author}"
    return f"{header}\n\n{clean_text(text)}"


def _fmt_chess_game(doc):
    text = doc.get('text', '')
    return clean_text(text) if text else None


def _fmt_chess_book(doc):
    title  = doc.get('title', '')
    author = doc.get('author', '')
    text   = doc.get('text', '')
    if not text:
        return None
    header = title
    if author:
        header += f" by {author}"
    return f"{header}\n\n{clean_text(text)}"


def iter_source(source_name, env, offset, limit):
    """Return a generator of cleaned text documents for the given source."""

    if source_name == 'wikipedia_articles':
        return read_wikipedia_articles(env, offset, limit)

    if source_name == 'year_topics':
        return read_year_topics(env, offset, limit)

    if source_name == 'gutenberg':
        p = Path(env['gutenberg_data']) / 'corpus' / 'gutenberg_corpus.jsonl'
        return _read_jsonl(p, offset, limit, _fmt_gutenberg)

    if source_name == 'chess_games':
        p = Path(env['chess_data']) / 'corpus' / 'chess_games.jsonl'
        return _read_jsonl(p, offset, limit, _fmt_chess_game)

    if source_name == 'chess_books':
        p = Path(env['chess_data']) / 'corpus' / 'chess_archive_books.jsonl'
        return _read_jsonl(p, offset, limit, _fmt_chess_book)

    raise ValueError(f"Unknown source: {source_name}")


# ── Shard writer ─────────────────────────────────────────────────────────

class ShardWriter:
    """
    Accumulate token IDs and flush them to numbered ``.bin`` shard files.

    Each shard is a flat array of **uint16** token IDs written in native
    byte order (little-endian on x86).  The maximum shard size is governed
    by *max_tokens* (default 50 M tokens ≈ 100 MB).
    """

    def __init__(self, shards_dir, prefix, start_num=0,
                 max_tokens=SHARD_MAX_TOKENS):
        self.shards_dir  = Path(shards_dir)
        self.prefix      = prefix
        self.shard_num   = start_num
        self.max_tokens  = max_tokens
        self.buffer      = []
        self.buf_len     = 0
        self.total_tokens = 0
        self.shard_files = []

    # ── public API ──

    def add(self, token_ids):
        """Append a sequence of token IDs to the buffer."""
        self.buffer.extend(token_ids)
        n = len(token_ids)
        self.buf_len += n
        self.total_tokens += n
        if self.buf_len >= self.max_tokens:
            self._flush()

    def close(self):
        """Flush remaining buffer.  Returns (total_tokens, shard_files)."""
        self._flush()
        return self.total_tokens, list(self.shard_files)

    # ── internals ──

    def _flush(self):
        if not self.buffer:
            return
        fname = f"{self.prefix}_{self.shard_num:06d}.bin"
        path = self.shards_dir / fname
        np.array(self.buffer, dtype=np.uint16).tofile(path)
        self.shard_files.append(fname)
        self.shard_num += 1
        self.buffer = []
        self.buf_len = 0


# ── Tokenization pipeline ───────────────────────────────────────────────

def process_source(source_name, tokenizer, eos_id, backend, env,
                   manifest, target_count, shards_dir, verbose=False):
    """
    Tokenize documents from *source_name* and write shard files.

    Only processes items beyond what has already been tokenized (according
    to the manifest).  Returns the updated source-state dict.
    """
    state = get_source_state(manifest, source_name)
    already = state['processed_count']

    if already >= target_count:
        if verbose:
            print(f"    Already at {already:,} items (target {target_count:,})")
        return state

    new_count = target_count - already
    offset = already

    # Determine next shard number from existing files
    existing_shards = state['shard_files']
    next_shard = 0
    if existing_shards:
        last = existing_shards[-1]
        try:
            next_shard = int(last.rsplit('_', 1)[1].replace('.bin', '')) + 1
        except (ValueError, IndexError):
            next_shard = len(existing_shards)

    writer = ShardWriter(shards_dir, source_name, start_num=next_shard)

    # --- stream documents, batch-tokenize, write shards ---
    docs = iter_source(source_name, env, offset, new_count)
    batch = []
    processed = 0
    t0 = time.time()

    progress = None
    if tqdm and not verbose:
        progress = tqdm(total=new_count, desc=f"    {source_name}",
                        unit='doc', leave=True)

    for doc_text in docs:
        batch.append(doc_text)

        if len(batch) >= TOKENIZE_BATCH_SIZE:
            _flush_batch(batch, tokenizer, eos_id, backend, writer)
            processed += len(batch)
            if progress:
                progress.update(len(batch))
            elif verbose and processed % 10_000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"      {processed:,}/{new_count:,} "
                      f"({rate:.0f} docs/s, "
                      f"{fmt_tokens(writer.total_tokens)} tokens)")
            batch = []

    # remainder
    if batch:
        _flush_batch(batch, tokenizer, eos_id, backend, writer)
        processed += len(batch)
        if progress:
            progress.update(len(batch))

    if progress:
        progress.close()

    total_new_tokens, new_shards = writer.close()
    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0

    # Update state
    state['processed_count'] = already + processed
    state['token_count'] = state.get('token_count', 0) + total_new_tokens
    state['shard_files'] = existing_shards + new_shards

    print(f"    → {processed:,} docs, {fmt_tokens(total_new_tokens)} new tokens, "
          f"{len(new_shards)} shard(s), {fmt_duration(elapsed)} ({rate:.0f} docs/s)")
    return state


def _flush_batch(batch, tokenizer, eos_id, backend, writer):
    """Tokenize a batch of texts and write tokens + EOS to the shard writer."""
    token_lists = encode_batch(tokenizer, batch, backend)
    for tokens in token_lists:
        writer.add(tokens)
        writer.add([eos_id])


# ── Finalization (chunk → shuffle → split) ───────────────────────────────

def finalize_corpus(manifest, shards_dir, output_dir,
                    seq_length, val_ratio, verbose=False):
    """
    Read all shard files, pack into fixed-length sequences, shuffle,
    and write ``train.bin`` / ``val.bin``.
    """
    banner("Finalizing corpus")

    # Collect shard paths
    all_paths = []
    for src_state in manifest['sources'].values():
        for fname in src_state.get('shard_files', []):
            p = shards_dir / fname
            if p.exists():
                all_paths.append(p)

    if not all_paths:
        print("  No shard files found — run tokenization first")
        return

    print(f"  Reading {len(all_paths)} shard file(s) …")
    arrays = []
    for p in sorted(all_paths):
        arr = np.fromfile(p, dtype=np.uint16)
        arrays.append(arr)
        if verbose:
            print(f"    {p.name}: {fmt_tokens(len(arr))} tokens")

    all_tokens = np.concatenate(arrays)
    total = len(all_tokens)
    print(f"  Total tokens : {fmt_tokens(total)} ({fmt_bytes(total * 2)})")

    # Chunk into sequences
    n_seq = total // seq_length
    remainder = total - n_seq * seq_length
    all_tokens = all_tokens[:n_seq * seq_length]
    print(f"  Sequences    : {n_seq:,} × {seq_length} "
          f"(discarding {remainder:,} remainder tokens)")

    sequences = all_tokens.reshape(n_seq, seq_length)

    # Deterministic shuffle
    print(f"  Shuffling (seed={SHUFFLE_SEED}) …")
    rng = np.random.default_rng(seed=SHUFFLE_SEED)
    rng.shuffle(sequences)

    # Train / val split
    n_val = max(1, int(n_seq * val_ratio))
    n_train = n_seq - n_val

    train_path = output_dir / 'train.bin'
    val_path   = output_dir / 'val.bin'

    print(f"  Train : {n_train:,} seqs → {fmt_tokens(n_train * seq_length)} tokens")
    print(f"  Val   : {n_val:,} seqs → {fmt_tokens(n_val * seq_length)} tokens")

    print(f"  Writing {train_path.name} …")
    sequences[:n_train].flatten().tofile(train_path)
    print(f"  Writing {val_path.name} …")
    sequences[n_train:].flatten().tofile(val_path)

    # Record in manifest
    manifest['finalized']       = True
    manifest['train_sequences'] = int(n_train)
    manifest['val_sequences']   = int(n_val)
    manifest['train_tokens']    = int(n_train * seq_length)
    manifest['val_tokens']      = int(n_val * seq_length)
    manifest['seq_length']      = seq_length
    manifest['val_ratio']       = val_ratio

    print(f"  Output:")
    print(f"    {train_path}  ({fmt_bytes(train_path.stat().st_size)})")
    print(f"    {val_path}  ({fmt_bytes(val_path.stat().st_size)})")
    print("  Finalization complete")


# ── Status & info ────────────────────────────────────────────────────────

def show_status(manifest, corpus_dir, shards_dir):
    """Print current corpus state."""
    banner("Training Corpus Status")

    tok = manifest.get('tokenizer', '(not set)')
    print(f"  Tokenizer  : {tok}")
    print(f"  Vocab size : {manifest.get('vocab_size', '?'):,}"
          if isinstance(manifest.get('vocab_size'), int)
          else f"  Vocab size : ?")
    print(f"  EOS ID     : {manifest.get('eos_id', '?')}")
    print(f"  Seq length : {manifest.get('seq_length', DEFAULT_SEQ_LENGTH)}")
    print(f"  Dtype      : {manifest.get('dtype', 'uint16')}")
    total = manifest.get('total_tokens', 0)
    print(f"  Total tkns : {fmt_tokens(total)}")
    print(f"  Finalized  : {manifest.get('finalized', False)}")
    print()

    sources = manifest.get('sources', {})
    if sources:
        print("  Sources:")
        for name in ALL_SOURCES:
            if name not in sources:
                continue
            s = sources[name]
            avail  = s.get('total_available', '?')
            done   = s.get('processed_count', 0)
            tokens = s.get('token_count', 0)
            shards = len(s.get('shard_files', []))
            pct = (done / avail * 100) if isinstance(avail, int) and avail else 0
            print(f"    {name}:")
            avail_str = f"{avail:,}" if isinstance(avail, int) else str(avail)
            print(f"      Items   : {done:,} / {avail_str} ({pct:.1f}%)")
            print(f"      Tokens  : {fmt_tokens(tokens)}")
            print(f"      Shards  : {shards}")
    else:
        print("  No sources processed yet")

    # Final output files
    train_path = corpus_dir / 'train.bin'
    val_path   = corpus_dir / 'val.bin'
    if train_path.exists() or val_path.exists():
        print()
    if train_path.exists():
        n = manifest.get('train_sequences', '?')
        print(f"  train.bin : {fmt_bytes(train_path.stat().st_size)} "
              f"({n:,} seqs)" if isinstance(n, int)
              else f"  train.bin : {fmt_bytes(train_path.stat().st_size)}")
    if val_path.exists():
        n = manifest.get('val_sequences', '?')
        print(f"  val.bin   : {fmt_bytes(val_path.stat().st_size)} "
              f"({n:,} seqs)" if isinstance(n, int)
              else f"  val.bin   : {fmt_bytes(val_path.stat().st_size)}")

    shard_bytes = sum(
        (shards_dir / f).stat().st_size
        for s in sources.values()
        for f in s.get('shard_files', [])
        if (shards_dir / f).exists()
    )
    if shard_bytes:
        print(f"\n  Shard dir  : {fmt_bytes(shard_bytes)} in {shards_dir}")


def show_info(env, enabled_sources):
    """Show source availability and estimated sizes."""
    banner("Source Information")

    for name in ALL_SOURCES:
        info = SOURCE_INFO[name]
        enabled = name in enabled_sources
        marker = '●' if enabled else '○'
        count = count_source(name, env)
        print(f"\n  {marker}  {name}")
        print(f"     {info['description']}")
        print(f"     Type        : {info['type']}")
        print(f"     Items       : {count:,}")
        print(f"     Est. tokens : {info['estimated_tokens']}")

    print(f"\n  Tokenizer presets:")
    for name, preset in TOKENIZER_PRESETS.items():
        print(f"    {name}  →  {preset['hf_repo']}  "
              f"(vocab {preset['vocab_size']:,})")

    # DB connectivity
    ok, msg = test_db()
    print(f"\n  PostgreSQL : {'OK' if ok else 'UNAVAILABLE — ' + msg}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    start_time = time.time()

    # ── environment ──
    deepred_root = os.environ.get('DEEPRED_ROOT', '/mnt/data')
    env = {
        'deepred_root':   deepred_root,
        'wiki_data':      os.environ.get('WIKI_DATA',
                              os.path.join(deepred_root, 'wikipedia')),
        'gutenberg_data': os.environ.get('GUTENBERG_DATA',
                              os.path.join(deepred_root, 'gutenberg')),
        'chess_data':     os.environ.get('CHESS_DATA',
                              os.path.join(deepred_root, 'chess')),
    }
    default_output = os.path.join(deepred_root, 'training_corpus')

    # ── argparse ──
    parser = argparse.ArgumentParser(
        description='Create tokenized training corpus for CPT.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --download-tokenizer                         Download TinyLlama tokenizer
  %(prog)s --info                                       Show sources & sizes
  %(prog)s --percent 1                                  Tokenize 1%% (quick test)
  %(prog)s --percent 100                                Expand to 100%%
  %(prog)s --sources wikipedia_articles,gutenberg --percent 100
  %(prog)s --finalize                                   Chunk, shuffle, split
  %(prog)s --status                                     Show progress
  %(prog)s --tokenizer SmolLM2-360M --download-tokenizer   Dev tokenizer""",
    )

    grp_action = parser.add_argument_group('actions')
    grp_action.add_argument(
        '--percent', type=float, default=None, metavar='N',
        help='Process N%% of each source (1–100).  Incremental.')
    grp_action.add_argument(
        '--finalize', action='store_true',
        help='Pack shards into shuffled train.bin / val.bin')
    grp_action.add_argument(
        '--download-tokenizer', action='store_true',
        help='Download tokenizer files from HuggingFace and exit')
    grp_action.add_argument(
        '--status', action='store_true',
        help='Print current corpus status and exit')
    grp_action.add_argument(
        '--info', action='store_true',
        help='Print source information and exit')
    grp_action.add_argument(
        '--reset', action='store_true',
        help='Delete existing shards and manifest, then proceed')

    grp_opts = parser.add_argument_group('options')
    grp_opts.add_argument(
        '--tokenizer', default='TinyLlama-1.1B', metavar='NAME',
        help='Tokenizer preset or path (default: TinyLlama-1.1B)')
    grp_opts.add_argument(
        '--sources', default=None, metavar='SRC,SRC,…',
        help=f'Comma-separated sources (default: all).  '
             f'Choices: {",".join(ALL_SOURCES)}')
    grp_opts.add_argument(
        '--seq-length', type=int, default=DEFAULT_SEQ_LENGTH, metavar='N',
        help=f'Sequence length for --finalize (default: {DEFAULT_SEQ_LENGTH})')
    grp_opts.add_argument(
        '--val-ratio', type=float, default=DEFAULT_VAL_RATIO, metavar='R',
        help=f'Validation ratio for --finalize (default: {DEFAULT_VAL_RATIO})')
    grp_opts.add_argument(
        '--output-dir', default=default_output, metavar='DIR',
        help=f'Base output directory (default: {default_output})')
    grp_opts.add_argument(
        '--workers', type=int, default=None, metavar='N',
        help='Tokenizer thread count (default: all CPU cores)')
    grp_opts.add_argument(
        '--verbose', '-v', action='store_true',
        help='Detailed per-batch output instead of progress bars')

    args = parser.parse_args()

    # ── Set parallelism ──
    if args.workers:
        os.environ['RAYON_NUM_THREADS'] = str(args.workers)

    # ── Resolve sources ──
    if args.sources:
        sources = [s.strip() for s in args.sources.split(',')]
        bad = [s for s in sources if s not in ALL_SOURCES]
        if bad:
            parser.error(f"Unknown source(s): {', '.join(bad)}\n"
                         f"  Valid: {', '.join(ALL_SOURCES)}")
    else:
        sources = list(ALL_SOURCES)

    # ── Resolve paths ──
    output_base   = Path(args.output_dir)
    tokenizer_name = args.tokenizer
    tokenizer_dir = output_base / 'tokenizers' / tokenizer_name
    corpus_dir    = output_base / tokenizer_name
    shards_dir    = corpus_dir / 'shards'
    manifest_path = corpus_dir / 'manifest.json'

    # ── Early-exit actions ──

    if args.info:
        show_info(env, sources)
        return

    if args.download_tokenizer:
        banner(f"Download Tokenizer: {tokenizer_name}")
        download_tokenizer(tokenizer_name, tokenizer_dir)
        return

    if args.status:
        manifest = load_manifest(manifest_path)
        show_status(manifest, corpus_dir, shards_dir)
        return

    # ── Need at least one action beyond early exits ──
    if args.percent is None and not args.finalize and not args.reset:
        parser.print_help()
        print("\nSpecify --percent N, --finalize, --status, --info, "
              "or --download-tokenizer.")
        sys.exit(0)

    # ── Prepare directories ──
    ensure_dir(output_base)
    ensure_dir(corpus_dir)
    ensure_dir(shards_dir)

    # ── Reset ──
    if args.reset:
        import shutil
        print(f"Resetting: {corpus_dir}")
        if shards_dir.exists():
            shutil.rmtree(shards_dir)
        for f in corpus_dir.glob('*.bin'):
            f.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
        ensure_dir(shards_dir)
        print("  Reset complete")
        if args.percent is None and not args.finalize:
            return

    # ── Load tokenizer ──
    if not tokenizer_dir.exists() or not any(tokenizer_dir.iterdir()):
        print(f"Tokenizer not found: {tokenizer_dir}")
        print(f"  Run:  python3 {sys.argv[0]} "
              f"--tokenizer {tokenizer_name} --download-tokenizer")
        sys.exit(1)

    tokenizer, eos_id, vocab_size, backend = load_tokenizer(tokenizer_dir)
    workers_desc = args.workers or os.cpu_count() or 1
    print(f"Tokenizer : {tokenizer_name}  (vocab {vocab_size:,},  "
          f"eos {eos_id},  backend {backend})")
    print(f"Threads   : {workers_desc}")

    # ── Load / validate manifest ──
    manifest = load_manifest(manifest_path)

    if (manifest.get('tokenizer') is not None
            and manifest['tokenizer'] != tokenizer_name
            and not args.reset):
        print(f"\nError: Existing manifest uses tokenizer "
              f"'{manifest['tokenizer']}', but you requested "
              f"'{tokenizer_name}'.")
        print(f"  Use --reset to start fresh, or "
              f"--tokenizer {manifest['tokenizer']} to continue.")
        sys.exit(1)

    manifest['tokenizer']  = tokenizer_name
    manifest['vocab_size'] = vocab_size
    manifest['eos_id']     = eos_id

    # ── Tokenization pass ──
    if args.percent is not None:
        if not (0 < args.percent <= 100):
            parser.error("--percent must be between 0 (exclusive) and 100")

        banner(f"Tokenizing Corpus  ({args.percent}%)")
        print(f"  Output  : {corpus_dir}")
        print(f"  Sources : {', '.join(sources)}")
        print()

        try:
            for source_name in sources:
                total = count_source(source_name, env)
                if total == 0:
                    print(f"  {source_name}: no data available — skipping")
                    continue

                target = math.ceil(total * args.percent / 100)
                state  = get_source_state(manifest, source_name)
                state['total_available'] = total
                already = state['processed_count']

                print(f"  {source_name}:  {total:,} available,  "
                      f"target {target:,} ({args.percent}%),  "
                      f"done {already:,}")

                if already >= target:
                    print(f"    → already complete")
                    continue

                state = process_source(
                    source_name, tokenizer, eos_id, backend, env,
                    manifest, target, shards_dir, args.verbose,
                )
                manifest['sources'][source_name] = state

                # Checkpoint after each source
                save_manifest(manifest, manifest_path)

        except KeyboardInterrupt:
            print("\n\nInterrupted — saving progress …")
            save_manifest(manifest, manifest_path)
            sys.exit(1)

        # Summary
        elapsed = time.time() - start_time
        total_tokens = sum(
            s.get('token_count', 0) for s in manifest['sources'].values()
        )
        print()
        banner("Tokenization Summary")
        print(f"  Total tokens : {fmt_tokens(total_tokens)}")
        print(f"  Duration     : {fmt_duration(elapsed)}")
        print(f"  Manifest     : {manifest_path}")
        print()
        print(f"  Next step — finalize the corpus:")
        print(f"    python3 {sys.argv[0]} --finalize")

        save_manifest(manifest, manifest_path)

    # ── Finalize ──
    if args.finalize:
        finalize_corpus(manifest, shards_dir, corpus_dir,
                        args.seq_length, args.val_ratio, args.verbose)
        save_manifest(manifest, manifest_path)

    elapsed = time.time() - start_time
    print(f"\nDone in {fmt_duration(elapsed)}")


if __name__ == '__main__':
    main()
