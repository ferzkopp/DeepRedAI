# Chess Setup
## Preparing the Chess Training Corpus

This document describes how to retrieve, convert, and prepare chess content for use as CPT (Continued Pre-Training) data. Chess knowledge is critical to the Deep Red persona — the model needs genuine understanding of rules, notation, strategy, opening theory, endgame principles, famous games, and competitive chess history through the July 20, 1969 temporal boundary.

---

## Overview

Chess content is assembled from four independent sources:

| Source | Script | Output | Estimated Tokens |
|--------|--------|--------|-----------------|
| **Gutenberg chess books** | `retrieve_gutenberg.py` | `$GUTENBERG_DATA/corpus/` | 1–2M |
| **PGN game databases** | `retrieve_chess_content.py` (Phase 1 → 2) | `$CHESS_DATA/corpus/chess_games.jsonl` | 15–60M |
| **Internet Archive books** | `retrieve_chess_content.py` (Phase 3) | `$CHESS_DATA/corpus/chess_archive_books.jsonl` | 500K–1.5M |
| **Wikipedia chess articles** | `extract_wikipedia.py` (already captured) | Main Wikipedia corpus | 2–10M (included) |

Total new chess content: **~25–95M tokens** (~1–4% of the main CPT corpus).

### Phase Architecture

The retrieval script (`retrieve_chess_content.py`) is organised into independent phases that can be run individually for testing and incremental builds:

| Phase | Action | Dependencies |
|-------|--------|-------------|
| **Phase 1** | Download PGN databases (PGN Mentor, Lumbras Gigabase) | `requests` |
| **Phase 2** | Convert PGN to natural-language prose (pre-1969 filter) | `python-chess`, Phase 1 output |
| **Phase 3** | Download public-domain chess books from Internet Archive | `requests` |

---

## Environment Variables

```bash
# Source the DeepRedAI environment (sets all paths automatically)
source deepred-env.sh
```

The relevant variables from [deepred-env.sh](../deepred-env.sh):

```
CHESS_DATA      — Base directory for chess data (default: $DEEPRED_ROOT/chess)
GUTENBERG_DATA  — Base directory for Gutenberg data (default: $DEEPRED_ROOT/gutenberg)
```

---

## Directory Structure

```
$CHESS_DATA/                          (/mnt/data/chess)
├── pgn/                              Raw PGN downloads (Phase 1)
│   ├── pgnmentor/
│   │   ├── players/                  22 player collections
│   │   │   ├── Morphy.pgn
│   │   │   ├── Capablanca.pgn
│   │   │   └── ...
│   │   └── events/                   5 event collections
│   │       ├── WorldChamp.pgn
│   │       └── ...
│   └── lumbras/                      Lumbras Gigabase OTB games
│       ├── LumbrasGigaBase_OTB_0001-1899.7z  (manually downloaded)
│       ├── LumbrasGigaBase_OTB_1900-1949.7z  (manually downloaded)
│       ├── LumbrasGigaBase_OTB_1950-1969.7z  (manually downloaded)
│       └── *.pgn                        (extracted by Phase 1)
├── archive/                          Raw Internet Archive texts (Phase 3)
│   ├── mysystem00nimz.txt
│   └── ...
└── corpus/                           Post-processed JSONL (training-ready)
    ├── chess_games.jsonl             PGN → narrative text (Phase 2)
    └── chess_archive_books.jsonl     Internet Archive books (Phase 3)
```

Gutenberg chess books live in the existing Gutenberg corpus:

```
$GUTENBERG_DATA/
└── corpus/
    ├── priority_works.jsonl          Contains 12 chess books (Chess & Strategy category)
    └── gutenberg_corpus.jsonl        Full corpus including chess subject matches
```

---

## Prerequisites

### Python dependencies

Core dependencies (`requests`, `beautifulsoup4`, `tqdm`) are installed by the standard setup. Phase 2 additionally requires `python-chess`:

```bash
source deepred-env.sh
pip install python-chess
```

### Directory setup

The scripts create directories automatically and verify write access. If the parent directory is not writable, the script prints the exact `sudo` command needed:

```bash
sudo mkdir -p /mnt/data/chess && sudo chown -R $USER:$USER /mnt/data/chess
```

---

## Usage

### Step 1: Gutenberg Chess Books

The `retrieve_gutenberg.py` script includes 16 chess books in its **Chess & Strategy** priority category:

| Gutenberg ID | Title | Author |
|-------------|-------|--------|
| 33870 | *Chess Fundamentals* | Capablanca |
| 5614 | *Chess Strategy* | Edward Lasker |
| 4913 | *Chess and Checkers: the Way to Mastership* | Edward Lasker |
| 16377 | *The Blue Book of Chess* | Howard Staunton |
| 34180 | *The Exploits and Triumphs of Paul Morphy* | Frederick M. Edge |
| 4902 | *Chess History and Reminiscences* | H.E. Bird |
| 55278 | *Chess Generalship, Vol. I* | Franklin K. Young |
| 10672 | *Game and Playe of the Chesse* | William Caxton (1474) |
| 4542 | *Checkmates for Three Pieces* | W.B. Fishburne |
| 4656 | *Checkmates for Four Pieces* | W.B. Fishburne |
| 39445 | *Hoyle's Games Modernized* | Prof. Hoffmann |
| 36821 | *Maxims and Hints on Angling, Chess, Shooting* | Richard Penn |
| 60420 | *Observations on the Automaton Chess Player* | Oxford graduate (~1819) |
| 61410 | *An Attempt to Analyse the Automaton Chess Player* | Robert Willis (1821) |
| 64061 | *War-Chess, or the Game of Battle* | Charles Richardson |
| 63660 | *The Game of Chess: A Play in One Act* | Kenneth Sawyer Goodman (1914) |

Additionally, 2 chess-themed works are in the **Early Science Fiction** category:

| Gutenberg ID | Title | Author |
|-------------|-------|--------|
| 1153 | *The Chessmen of Mars* | Edgar Rice Burroughs (1922) |
| 61213 | *The 64-Square Madhouse* | Fritz Leiber (1962) |

```bash
source deepred-env.sh

# Download all priority works including chess books
python scripts/retrieve_gutenberg.py --priority-only

# Or download chess subject matches in the extended corpus too
python scripts/retrieve_gutenberg.py
```

No additional configuration is needed — the chess Gutenberg IDs are built into the script.

### Step 2: PGN Game Databases (Phase 1)

Download PGN files from PGN Mentor (all player + pre-1969 event collections) and Lumbras Gigabase:

```bash
source deepred-env.sh
python scripts/retrieve_chess_content.py --phase 1
```

PGN Mentor player files are downloaded as ZIPs and automatically extracted. Event PGNs download directly.

**Lumbras Gigabase** requires a one-time manual download (hosted on MEGA):

1. Visit: https://lumbrasgigabase.com/en/download-in-pgn-format-en/
2. Under the **Downloads OTB** tab, download these three 7z archives:
   - `LumbrasGigaBase_OTB_0001-1899.7z`
   - `LumbrasGigaBase_OTB_1900-1949.7z`
   - `LumbrasGigaBase_OTB_1950-1969.7z`
3. Place the `.7z` files in: `$CHESS_DATA/pgn/lumbras/`
4. Re-run Phase 1 — the script will detect and extract them automatically (requires `7z`).

The script checks for these ZIPs on each run and prints detailed instructions if any are missing.

### Step 3: PGN to Narrative Text (Phase 2)

Convert downloaded PGN games into natural-language prose suitable for LLM training:

```bash
source deepred-env.sh
python scripts/retrieve_chess_content.py --phase 2
```

This phase:
- Parses all PGN files in `$CHESS_DATA/pgn/`
- Filters out games dated after July 20, 1969
- Converts the first 1,000 games to **annotated** narrative (richer commentary on captures, checks, castling)
- Converts remaining games to **structured summaries** (header + move list + result)
- Writes JSONL output to `$CHESS_DATA/corpus/chess_games.jsonl`

#### Robust parsing & parallel mode

Phase 2 includes automatic stuck detection to handle malformed PGN files:

| Safety mechanism | Default | Description |
|-----------------|---------|-------------|
| Per-game timeout | 30 s | Kills `read_game()` if a single game parse hangs |
| Per-file timeout | 300 s | Abandons an entire file if it exceeds wall-clock limit |
| Consecutive-error limit | 10 | Skips remaining file after 10 consecutive parse failures |
| Skip-ahead recovery | — | On error, scans forward to the next `[Event ` header |

Files are sorted **largest-first** so that big files start processing early (important for parallel mode).

For large PGN collections, use `--workers` to process files in parallel:

```bash
# Process PGN files with 4 parallel workers
python scripts/retrieve_chess_content.py --phase 2 --workers 4
```

Adjust the annotated-vs-summary split:

```bash
# Convert 2,000 games in annotated mode instead of the default 1,000
python scripts/retrieve_chess_content.py --phase 2 --annotated-limit 2000
```

**Requires** `python-chess`:

```bash
pip install python-chess
```

### Step 4: Internet Archive Books (Phase 3)

Download OCR text of public-domain chess books from the Internet Archive:

```bash
source deepred-env.sh
python scripts/retrieve_chess_content.py --phase 3
```

The script attempts multiple download strategies (DjVu text, metadata API, stream endpoint). Failed downloads are reported — some items may need manual retrieval from archive.org.

### Run All Phases

```bash
source deepred-env.sh
python scripts/retrieve_chess_content.py
```

This runs Phases 1, 2, and 3 in sequence.

---

## Command-Line Reference

Script: [scripts/retrieve_chess_content.py](../scripts/retrieve_chess_content.py)

| Option | Default | Description |
|--------|---------|-------------|
| `--phase {1,2,3}` | All | Run only the specified phase |
| `--chess-dir` | `$CHESS_DATA` or `/mnt/data/chess` | Base directory for chess data |
| `--status` | — | Show status of existing chess data and exit |
| `--info` | — | Display source lists and configuration, then exit |
| `--reset` | — | Delete corpus JSONL files and start fresh |
| `--annotated-limit` | 1000 | Number of games to convert in annotated mode |
| `--workers` | 1 | Number of parallel workers for Phase 2 (1 = sequential) |
| `--verbose`, `-v` | False | Show detailed per-item output |

```bash
# Inspect what's been downloaded so far
python scripts/retrieve_chess_content.py --status

# List all configured sources
python scripts/retrieve_chess_content.py --info

# Wipe corpus output and reconvert
python scripts/retrieve_chess_content.py --reset
python scripts/retrieve_chess_content.py --phase 2
```

---

## Temporal Boundary

All chess content respects the **July 20, 1969** cutoff:

- **PGN games**: Filtered by the `Date` header — games dated after 1969 are excluded. Games without dates are included (most undated games in historical databases are pre-1969).
- **Gutenberg books**: All 12 chess books were published well before 1969.
- **Internet Archive**: Only pre-1929 publications (US public domain) are targeted.
- **Wikipedia**: The existing temporal extraction pipeline handles chess articles automatically.

### Key Historical Coverage

The pre-1969 era covers chess's classical and Soviet golden ages:

- **World Champions 1–10**: Steinitz through Spassky (became champion June 17, 1969)
- **Major pre-1969 figures**: Morphy, Anderssen, Nimzowitsch, Réti, Tartakower, Bronstein, Keres, Fischer (active from 1956)
- **Major tournaments**: London 1851, Hastings 1895, St. Petersburg 1914, New York 1924, AVRO 1938, Zurich 1953
- **Opening theory**: All classical openings predate 1969 (Sicilian, Ruy Lopez, Queen's Gambit, King's Indian, French, Caro-Kann, etc.)

---

## Output Format

### PGN Game Narratives (chess_games.jsonl)

Each record contains a natural-language rendering of a chess game:

```json
{
  "key": "Morphy, Paul-Duke of Brunswick-1858.??.??-Paris Opera-?",
  "white": "Morphy, Paul",
  "black": "Duke of Brunswick and Count Isouard",
  "date": "1858.??.??",
  "event": "Paris Opera",
  "eco": "C41",
  "opening": "Philidor Defense",
  "result": "1-0",
  "source_file": "pgnmentor/players/Morphy.pgn",
  "mode": "annotated",
  "text": "Morphy, Paul vs. Duke of Brunswick and Count Isouard at Paris Opera (1858)\n\nOpening: Philidor Defense (C41)\n\n1.e4 e5 2.Nf3 d6 3.d4 Bg4 ...",
  "length": 1234
}
```

The `mode` field is `"annotated"` (richer commentary) or `"summary"` (compact format).

### Internet Archive Books (chess_archive_books.jsonl)

```json
{
  "identifier": "mysystem00nimz",
  "title": "My System",
  "author": "Aron Nimzowitsch",
  "pub_year": 1925,
  "text": "Full OCR text of the book...",
  "length": 234567,
  "source": "internet_archive"
}
```

---

## Idempotency

Both scripts are safe to run multiple times:

- `retrieve_gutenberg.py`: Tracks existing work IDs in JSONL files and skips duplicates.
- `retrieve_chess_content.py`:
  - **Phase 1**: Skips PGN files already on disk.
  - **Phase 2**: Tracks game keys in the output JSONL and skips duplicates.
  - **Phase 3**: Tracks Internet Archive identifiers in the output JSONL and skips duplicates.

Use `--reset` to wipe corpus output and reconvert from scratch.

---

## Pipeline Integration

```
Existing CPT Pipeline:
    Wikipedia (2–4B tokens) ─────────────────────────────┐
    Gutenberg books (200–500M tokens) ───────────────────┤
                                                         ├──→ Tokenize → Train
    Chess Gutenberg books (1–2M tokens) ─────────────────┤
    PGN → narrative text (15–60M tokens) ────────────────┤
    Internet Archive chess books (500K–1.5M tokens) ─────┘
```

---

## References

- [scripts/retrieve_chess_content.py](../scripts/retrieve_chess_content.py) — Chess PGN + Internet Archive retrieval
- [scripts/retrieve_gutenberg.py](../scripts/retrieve_gutenberg.py) — Gutenberg retrieval (includes chess books)
- [deepred-env.sh](../deepred-env.sh) — Environment configuration (`CHESS_DATA`, `GUTENBERG_DATA`)
- [Gutenberg-Setup.md](Gutenberg-Setup.md) — Gutenberg pipeline documentation
- [python-chess](https://python-chess.readthedocs.io/) — PGN parsing library
- [PGN Mentor](https://www.pgnmentor.com/files.html) — Free PGN player/event collections
- [Lumbras Gigabase](https://lumbrasgigabase.com/en/download-in-pgn-format-en/) — OTB master games (manual MEGA download)
- [Internet Archive Chess](https://archive.org/details/hokmome-chess-library) — Scanned chess book collections
