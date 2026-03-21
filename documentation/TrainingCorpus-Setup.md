# Training Corpus Setup

How to tokenize and prepare the CPT training corpus using `scripts/create_training_corpus.py`.

---

## Overview

This pipeline combines five data sources into a single shuffled, tokenized binary corpus for continued pre-training (CPT). The output is two files — `train.bin` and `val.bin` — containing document-aware packed 2048-token sequences of uint16 token IDs, ready for nanoGPT, LitGPT, or torchtune. Short documents are packed together intact (separated by EOS tokens), while long documents use overlapping sliding windows to preserve contextual continuity.

**Processing is incremental.** You can tokenize 1% of the corpus for a quick test, validate the output, then expand to 10%, 50%, or 100% without re-processing items that were already tokenized. A `manifest.json` file tracks progress.

### Pipeline Flow

```
┌──────────────────────────────────────────────────────────┐
│  Download tokenizer (--download-tokenizer)               │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Tokenize sources (--percent N)                          │
│                                                          │
│   Wikipedia  ──→ ┐                                       │
│   Year Topics ──→│                                       │
│   Gutenberg  ──→ ├──→ encode_batch() ──→ shard .bin files│
│   Chess Games*──→│        (Rust multi-threaded)          │
│   Chess Books ──→┘                                       │
│                                                          │
│   *Augmented chess narratives are paired with their raw  │
│    notation and prioritized during selection.            │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Finalize (--finalize)                                   │
│                                                          │
│   Read all shards → split on EOS into documents          │
│   → document-aware packing (short docs grouped,          │
│     long docs use 25% overlap sliding window)            │
│   → deterministic shuffle → 99/1 train/val split         │
│   → train.bin + val.bin                                  │
└──────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Packages

These should already be installed in the venv (`/mnt/data/venv`):

| Package | Purpose | Check |
|---------|---------|-------|
| `numpy` | Binary array I/O | `python3 -c "import numpy"` |
| `tokenizers` | Fast Rust-based tokenization | `python3 -c "from tokenizers import Tokenizer"` |
| `psycopg2-binary` | PostgreSQL access (for Wikipedia) | `python3 -c "import psycopg2"` |
| `huggingface_hub` | Tokenizer download | `python3 -c "from huggingface_hub import hf_hub_download"` |

Optional but recommended:

| Package | Purpose |
|---------|---------|
| `sentencepiece` | Fallback tokenizer backend |
| `tqdm` | Progress bars |

Install any missing packages:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
pip install tokenizers psycopg2-binary huggingface_hub sentencepiece tqdm
```

### Required Data

All of these should already exist from Prod Phases 1–2:

| Source | Location | Created by |
|--------|----------|------------|
| Wikipedia DB | PostgreSQL `wikidb`, 7M articles | `extract_wikipedia.py` + `augment_wikipedia_temporal.py` |
| Year topics | `/mnt/data/wikipedia/topics/year_topics_*.json` | `extract_year_topics.py` |
| Gutenberg | `/mnt/data/gutenberg/corpus/gutenberg_corpus.jsonl` | `retrieve_gutenberg.py` |
| Chess games | `/mnt/data/chess/corpus/chess_games.jsonl` | `retrieve_chess_content.py --phase 2` |
| Chess augmented | `/mnt/data/chess/corpus/augmented_chess_games.jsonl` | `augment_chess_games.py` (optional, ~10K+ games) |
| Chess books | `/mnt/data/chess/corpus/chess_archive_books.jsonl` | `retrieve_chess_content.py --phase 3` |

### PostgreSQL

The Wikipedia database must be running and accessible:

```bash
# Verify connectivity
psql -U wiki -d wikidb -c "SELECT COUNT(*) FROM articles WHERE temporal_classification='O'"
# Expected: ~1,553,510
```

---

## Step 1: Download the Tokenizer

The tokenizer converts raw text into integer token IDs. Two presets are available:

| Preset | Model | Vocab Size | Use For |
|--------|-------|------------|---------|
| **TinyLlama-1.1B** (default) | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | 32,000 | Production CPT |
| **SmolLM2-360M** | `HuggingFaceTB/SmolLM2-360M` | 49,152 | Dev CPT validation |

The `setup_strixhalo.py` script includes a `training_tokenizers` stage that downloads both tokenizer presets:

```bash
sudo python3 scripts/setup_strixhalo.py --stage training_tokenizers
```

This downloads `tokenizer.json`, `tokenizer.model`, and config files to:
```
/mnt/data/training_corpus/tokenizers/TinyLlama-1.1B/
/mnt/data/training_corpus/tokenizers/SmolLM2-360M/
```

### Alternative: Download via the Corpus Script

You can also download tokenizers individually using the corpus script:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
python3 scripts/create_training_corpus.py --download-tokenizer
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --download-tokenizer
```

---

## Step 2: Check Sources

Verify all data sources are available and see estimated sizes:

```bash
python3 scripts/create_training_corpus.py --info
```

Expected output:

```
============================================================
Source Information
============================================================

  ●  wikipedia_articles
     Pre-1969 Wikipedia articles from PostgreSQL (temporal_classification=O)
     Type        : database
     Items       : 1,553,510
     Est. tokens : ~1.4B

  ●  year_topics
     Year-by-year historical event summaries, years 151–1969 (JSON files)
     Type        : json_files
     Items       : 1,819
     Est. tokens : ~5M

  ●  gutenberg
     Project Gutenberg books — 766 public-domain titles (JSONL)
     Type        : jsonl
     Items       : 766
     Est. tokens : ~125M

  ●  chess_games
     Pre-1969 chess games — 356K games with augmented narrative pairing (JSONL)
     Type        : jsonl
     Items       : 355,980
     Augmented   : 10,247 (paired + prioritized)
     Est. tokens : ~134M

  ●  chess_books
     Internet Archive chess reference books — 10 titles (JSONL)
     Type        : jsonl
     Items       : 10
     Est. tokens : ~1M

  PostgreSQL : OK
```

---

## Step 3: Tokenize (Incremental)

### Quick Test (1%)

Start with 1% to validate the pipeline works end-to-end:

```bash
python3 scripts/create_training_corpus.py --percent 1
```

This processes ~15,500 Wikipedia articles, ~18 year-topic files, ~8 Gutenberg books, ~3,560 chess games (augmented games first), and all 10 chess books. Takes about 1–2 minutes.

> **Chess augmentation pairing:** When augmented narratives are available,
> they are prioritized — at 1% the entire chess slice may consist of
> augmented+notation pairs. Each paired item combines the LLM-generated
> narrative followed by the raw chess notation as a single training document.

### Expand to Full Corpus (100%)

When ready, expand to the full corpus. Only the remaining 99% is processed — the 1% from the test run is skipped:

```bash
python3 scripts/create_training_corpus.py --percent 100
```

**Estimated time for full corpus:** 5–15 minutes (depending on PostgreSQL throughput).

### Intermediate Steps

You can expand gradually:

```bash
python3 scripts/create_training_corpus.py --percent 10   # processes 9% more
python3 scripts/create_training_corpus.py --percent 50   # processes 40% more
python3 scripts/create_training_corpus.py --percent 100  # processes remaining 50%
```

### Select Specific Sources

Process only certain sources:

```bash
# Wikipedia only
python3 scripts/create_training_corpus.py --sources wikipedia_articles --percent 100

# Everything except chess games
python3 scripts/create_training_corpus.py --sources wikipedia_articles,year_topics,gutenberg,chess_books --percent 100
```

### Check Progress

```bash
python3 scripts/create_training_corpus.py --status
```

---

## Step 4: Finalize

After tokenization, pack the shards into shuffled training sequences:

```bash
python3 scripts/create_training_corpus.py --finalize
```

This:
1. Reads all shard `.bin` files
2. Concatenates them into one token stream
3. Splits the stream into individual documents (using EOS token delimiters)
4. **Short documents** (≤ 2048 tokens): packs multiple complete documents into each sequence, separated by EOS tokens — keeps documents intact
5. **Long documents** (> 2048 tokens): chunks using a sliding window with 25% overlap — preserves contextual continuity between consecutive sequences
6. Shuffles all resulting sequences (deterministic seed = 42)
7. Splits 99% train / 1% validation
8. Writes `train.bin` and `val.bin`

> **Why document-aware packing?** The naive approach (concatenate everything,
> chop into fixed-size chunks, shuffle) fragments long documents arbitrarily —
> a 20,000-token Gutenberg book would be split into ~10 chunks scattered
> randomly across the training set.  Document-aware packing ensures short
> documents stay intact within sequences, and long documents get overlapping
> windows so the model sees smooth context transitions rather than hard cuts.
> The 25% overlap means each position in a long document appears in roughly
> 1.3× as many training sequences, providing redundant context at boundaries.

### Custom Parameters

```bash
# Different sequence length (e.g., 1024 for memory-constrained training)
python3 scripts/create_training_corpus.py --finalize --seq-length 1024

# Different validation ratio (e.g., 2%)
python3 scripts/create_training_corpus.py --finalize --val-ratio 0.02

# No overlap for long documents (saves ~5% corpus size)
python3 scripts/create_training_corpus.py --finalize --overlap 0

# Higher overlap for maximum coherence (50%)
python3 scripts/create_training_corpus.py --finalize --overlap 0.5
```

### Re-Finalize

You can re-run `--finalize` at any time (e.g., after adding more data or changing the split ratio). It regenerates `train.bin` and `val.bin` from the existing shards.

---

## Output Structure

```
/mnt/data/training_corpus/
├── tokenizers/
│   ├── TinyLlama-1.1B/              ← downloaded tokenizer files
│   │   ├── tokenizer.json
│   │   ├── tokenizer.model
│   │   ├── tokenizer_config.json
│   │   └── special_tokens_map.json
│   └── SmolLM2-360M/                ← dev tokenizer (optional)
│       └── ...
│
├── TinyLlama-1.1B/                  ← corpus for this tokenizer
│   ├── manifest.json                ← tracks incremental progress
│   ├── shards/                      ← intermediate tokenized chunks
│   │   ├── wikipedia_articles_000000.bin
│   │   ├── wikipedia_articles_000001.bin
│   │   ├── ...
│   │   ├── gutenberg_000000.bin
│   │   ├── chess_games_000000.bin
│   │   ├── chess_books_000000.bin
│   │   └── year_topics_000000.bin
│   ├── train.bin                    ← final training data
│   └── val.bin                      ← final validation data
│
└── SmolLM2-360M/                    ← separate corpus for dev tokenizer
    └── ...
```

### Binary Format

- **Shard files** (`shards/*.bin`): Raw uint16 little-endian token IDs. Documents are separated by EOS tokens. Not chunked into fixed-length sequences — that happens during finalization.

- **train.bin / val.bin**: Raw uint16 little-endian token IDs organized as contiguous 2048-token sequences, produced by document-aware packing. Short documents are packed together with EOS separators; long documents use overlapping sliding windows (default 25% overlap). Ready for memory-mapped loading by training frameworks.

### Loading in Python

```python
import numpy as np

# Memory-mapped access (zero-copy, ideal for training)
tokens = np.memmap('train.bin', dtype=np.uint16, mode='r')
n_sequences = len(tokens) // 2048
print(f"{n_sequences:,} training sequences, {len(tokens):,} tokens")

# Random sequence access
seq_idx = 42
sequence = tokens[seq_idx * 2048 : (seq_idx + 1) * 2048]
```

### Manifest

`manifest.json` tracks all processing state:

```json
{
  "tokenizer": "TinyLlama-1.1B",
  "vocab_size": 32000,
  "eos_id": 2,
  "seq_length": 2048,
  "sources": {
    "wikipedia_articles": {
      "total_available": 1553510,
      "processed_count": 1553510,
      "token_count": 1435000000,
      "shard_files": ["wikipedia_articles_000000.bin", "..."]
    },
    "gutenberg": { "..." : "..." },
    "...": "..."
  },
  "total_tokens": 1620000000,
  "finalized": true,
  "packing": "document_aware",
  "overlap_ratio": 0.25,
  "long_doc_sequences": 95000,
  "short_doc_sequences": 720000,
  "train_sequences": 807000,
  "val_sequences": 8150
}
```

---

## Data Source Details

### Wikipedia Articles

- **Source:** PostgreSQL table `articles` where `temporal_classification = 'O'` (pre-1969)
- **Count:** ~1,553,510 articles
- **Format:** Each article is formatted as `Title\n\nContent` with basic whitespace normalization
- **Ordering:** Sorted by `id` (deterministic for incremental processing)
- **Content:** Already cleaned during import — wikitext markup stripped by `extract_wikipedia.py`

#### Wikipedia Boilerplate Filtering

Wikipedia articles stored in PostgreSQL retain some structural artifacts from the
original wikitext-to-markdown conversion (performed by `extract_wikipedia.py`).
Left unfiltered, these patterns leak into model outputs — for example, prompts
ending with `## See also * List of presidential election results by state`.

The corpus script applies a dedicated `clean_wikipedia_boilerplate()` pass to
every Wikipedia article *at tokenization time*, so the model never sees these
patterns during training. This is layered on top of the initial extraction
cleanup (which uses `mwparserfromhell` section removal but can miss edge cases).

**What is removed:**

| Pattern | Example | Reason |
|---------|---------|--------|
| Boilerplate sections | `## See also`, `## References`, `## External links`, `## Further reading`, `## Notes`, `## Bibliography`, `## Footnotes`, `## Citations` | Navigation links, citation lists, and non-prose content that sits at the tail of articles. Everything from the first boilerplate heading onward is truncated. |
| Markdown heading markers | `## Early life` → `Early life` | The heading *text* is preserved (it provides useful context), but the `##` markup is stripped so the model does not learn to reproduce markdown structure. |
| Navigation list items | `* List of presidents of the United States` | Bulleted lines starting with `List of`, `Lists of`, `Index of`, `Outline of`, `Category:`, `Portal:`, or `Template:` — these are internal-link navigation, not prose. |
| Categories metadata | `Categories: History, Politics, …` | Wikipedia category tags appended by the wiki parser. |

**Why filter here (not in extraction):**

- The PostgreSQL content is used by multiple consumers (search, display, MCP
  server) where headings and structure are useful. Training is the only consumer
  that needs them removed.
- Re-extracting the entire Wikipedia dump would be expensive. Filtering at
  tokenization time is fast (regex) and lets us iterate on the rules without
  touching the database.
- This creates a single, maintainable location for training-specific text
  cleaning rules.

### Year Topics

- **Source:** JSON files in `/mnt/data/wikipedia/topics/year_topics_YYYY.json`
- **Filter:** Only years ≤ 1969 (about 1,819 files)
- **Format:** One document per year, listing all events chronologically:
  ```
  Events of the year 1960
  January 1: Cameroon becomes independent from France.
  January 4: Albert Camus dies in a car accident.
  ...
  ```
- **Purpose:** Teaches the model temporal event knowledge and date associations

### Gutenberg Books

- **Source:** `/mnt/data/gutenberg/corpus/gutenberg_corpus.jsonl`
- **Count:** 766 books
- **Format:** `Title by Author\n\nFull text`
- **Fields used:** `title`, `author`, `text`
- **Purpose:** Rich literary English, thematic alignment (public domain pre-1969)

### Chess Games

- **Source:** `/mnt/data/chess/corpus/chess_games.jsonl`
- **Augmented:** `/mnt/data/chess/corpus/augmented_chess_games.jsonl` (optional, see [ChessAugmentation-Setup.md](ChessAugmentation-Setup.md))
- **Count:** 355,980 games (pre-1969, output of PGN→narrative conversion)
- **Augmented count:** ~10K+ and growing (LLM-generated Deep Red AI narratives)
- **Fields used:** `text` (from both files), `key` (for pairing)
- **Purpose:** Chess notation, strategy vocabulary, game knowledge

#### Augmentation Pairing

The training pipeline builds an in-memory index by scanning the `key` field in both
JSONL files. Games that have a matching augmented narrative are **prioritized** —
they appear first in the iteration order so that low-percentage runs (e.g. `--percent 5`)
select augmented games before plain notation-only games.

For each augmented game, the pipeline emits a **combined document**: the LLM-generated
narrative text followed by the original raw chess notation, separated by a double
newline. This teaches the model both the narrative style and the underlying game data.
Games without augmentation emit only their raw notation text, as before.

| Condition | Output per game |
|-----------|-----------------|
| Augmented narrative exists | Narrative text + `\n\n` + raw notation (one document) |
| No augmented narrative | Raw notation only |

> **Note:** If the augmented corpus grows between incremental runs (more games
> augmented), the prioritized ordering changes. Use `--reset` before re-tokenizing
> to ensure consistent pairing.

### Chess Books

- **Source:** `/mnt/data/chess/corpus/chess_archive_books.jsonl`
- **Count:** 10 books from Internet Archive
- **Format:** `Title by Author\n\nFull text`
- **Fields used:** `title`, `author`, `text`
- **Note:** OCR-sourced text — quality varies; basic cleanup applied

---

## Resource Utilization

### Tokenization Performance

Tokenization is **CPU-bound** — it does not use GPU compute. The `tokenizers` Rust library automatically parallelizes across all available CPU cores (32 on Strix Halo) via the RAYON thread pool. A single run of the full corpus takes approximately 5–15 minutes.

| Component | Role |
|-----------|------|
| **Strix Halo CPU** (32 cores) | Tokenization (RAYON multi-threaded via `tokenizers` library) |
| **Strix Halo CPU** | PostgreSQL reads, JSONL I/O |
| **A4000 GPU** | Not used for tokenization — available for other tasks |

To limit thread count (e.g., during parallel workloads):

```bash
python3 scripts/create_training_corpus.py --workers 16 --percent 100
```

### Why GPU is Not Used for Tokenization

BPE tokenization is a purely algorithmic CPU operation (dictionary lookup + merge rules). The HuggingFace `tokenizers` library implements this in optimized Rust and already saturates the CPU. GPU-based tokenization (e.g., via llama.cpp's `/tokenize` endpoint) would be orders of magnitude slower due to network and API overhead.

### Memory Usage

| Phase | RAM Usage | Notes |
|-------|-----------|-------|
| Tokenization | ~2–4 GB | Streaming batches; shard files are flushed at 100 MB |
| Finalization | ~4–8 GB | Document splitting + sliding-window overlap increases working set vs naive chunking (~1.6B tokens × 2 bytes = 3.2 GB base, plus ~33% for overlapping long-doc windows) |

Both fit comfortably in the 128 GB system.

---

## Downloading Base Models for Training

After the corpus is ready, you'll need the base models for CPT (Phase 3 and 4). These are separate from the tokenizer files.

The `setup_strixhalo.py` script includes a `training_models` stage that downloads both base models:

```bash
sudo python3 scripts/setup_strixhalo.py --stage training_models
```

This downloads:

| Model | HuggingFace Repo | Size | Destination |
|-------|------------------|------|-------------|
| SmolLM2-360M (dev) | `HuggingFaceTB/SmolLM2-360M` | ~720 MB | `/mnt/data/models/SmolLM2-360M/` |
| TinyLlama-1.1B (prod) | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | ~2.2 GB | `/mnt/data/models/TinyLlama-1.1B/` |

### Verifying Downloads

```bash
ls -lh /mnt/data/models/SmolLM2-360M/model*.safetensors
ls -lh /mnt/data/models/TinyLlama-1.1B/model*.safetensors
```

---

## Troubleshooting

### "Tokenizer not found"

Download it first:
```bash
python3 scripts/create_training_corpus.py --download-tokenizer
```

### "psycopg2 not installed" / PostgreSQL connection errors

```bash
pip install psycopg2-binary

# Verify DB is running
systemctl status postgresql
psql -U wiki -d wikidb -c "SELECT 1"
```

### "Existing manifest uses tokenizer X, but you requested Y"

Each tokenizer gets its own corpus directory. Either:
- Use `--reset` to start fresh with the new tokenizer
- Use `--tokenizer X` to continue with the existing tokenizer

### Slow Wikipedia Processing

Wikipedia is the largest source (~1.5M articles). If processing seems slow:
1. Check PostgreSQL performance: `top` should show postgres active, not the bottleneck
2. Ensure the `tokenizers` library is using all CPU cores (default behavior)
3. Use `--verbose` to see per-batch throughput stats

### Out of Memory During Finalization

If the full corpus exceeds available RAM during `--finalize`:
1. Reduce the source set: `--sources wikipedia_articles,gutenberg`
2. Use a smaller `--percent`
3. Consider splitting finalization (advanced — modify the script)

With the current ~1.6B token corpus, finalization uses ~3.2 GB — well within the 128 GB system.

### Starting Over

```bash
python3 scripts/create_training_corpus.py --reset
```

This deletes all shards, train/val files, and the manifest for the current tokenizer.

---

## Complete Workflow Example

```bash
source /mnt/data/DeepRedAI/deepred-env.sh

# 1. Download tokenizers (both presets)
sudo python3 scripts/setup_strixhalo.py --stage training_tokenizers

# 2. Verify sources
python3 scripts/create_training_corpus.py --info

# 3. Quick 1% test
python3 scripts/create_training_corpus.py --percent 1
python3 scripts/create_training_corpus.py --finalize
python3 scripts/create_training_corpus.py --status

# 4. Verify output
python3 -c "
import numpy as np
t = np.memmap('/mnt/data/training_corpus/TinyLlama-1.1B/train.bin', dtype=np.uint16, mode='r')
print(f'Train: {len(t):,} tokens, {len(t)//2048:,} sequences')
v = np.memmap('/mnt/data/training_corpus/TinyLlama-1.1B/val.bin', dtype=np.uint16, mode='r')
print(f'Val:   {len(v):,} tokens, {len(v)//2048:,} sequences')
"

# 5. Full corpus (incremental — only processes the remaining 99%)
python3 scripts/create_training_corpus.py --percent 100
python3 scripts/create_training_corpus.py --finalize
python3 scripts/create_training_corpus.py --status

# 6. Download base models for CPT (Phase 3)
sudo python3 scripts/setup_strixhalo.py --stage training_models
```
