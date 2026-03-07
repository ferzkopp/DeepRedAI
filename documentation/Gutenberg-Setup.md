# Gutenberg Setup
## Preparing Project Gutenberg Training Data

This document describes how to retrieve and prepare literature from Project Gutenberg for use as training data.

---

## Overview

Project Gutenberg provides 70,000+ free eBooks. The retrieval script downloads thematically relevant works and saves them in JSONL format suitable for downstream processing.

All works are filtered to ensure publication before 1969 (pre-moon landing) to maintain temporal consistency and selected by topics to align with the Soviet utopia aesthetic.

### What the script does

1. **Retrieve Priority Works** — Download ~60 curated books directly relevant to the project themes
2. **Build Extended Corpus** — Expand to hundreds of books by searching Library of Congress subject headings
3. **Temporal Filter** — Validate all content is pre-1969
4. **Save as JSONL** — Structured output for downstream chunking and indexing

The script is **idempotent**: re-running it skips works that already exist in the output files and only appends new ones.

---

## Source Material

### Priority Works (74 books)

The script defines `PRIORITY_WORKS` with specific Gutenberg IDs. Use `--info` to display the full list:

| Category | Works | Key Authors |
|----------|-------|-------------|
| **Utopian/Dystopian** | 9 | Bellamy, Wells, Morris, London, Gilman, Zamyatin, Plato, Nietzsche |
| **Russian Literature** | 14 | Dostoevsky (6), Tolstoy (2), Chekhov (4), Gorky, Turgenev, Gogol |
| **Early Sci-Fi** | 19 | Verne (5), Wells (8), Burroughs (4), Doyle, Čapek, Leiber |
| **Political Philosophy** | 8 | Marx/Engels, Kropotkin (2), Machiavelli, Tocqueville (2), Hobbes, Rousseau |
| **Isolation/Survival** | 4 | Defoe, Dumas, Nansen (2) |
| **Chess & Strategy** | 16 | Capablanca, Lasker (2), Staunton, Bird, Edge, Caxton, Fishburne (2), Young, Hoffmann, Penn, Willis, Goodman, Richardson |
| **Satire** | 4 | Chesterton, Swift (2), Voltaire |

### Subject Filters (60 subjects)

The script uses Library of Congress Subject Headings (LCSH) for expanded corpus building across categories: Fiction genres, Soviet/Russian themes, Space/Mars, AI/Chess, Survival/Isolation, Political conflict, Class/Power, Evolution/Future, Exploration, and Philosophy.

```bash
# View all priority works and subject filters
python scripts/retrieve_gutenberg.py --info
```

---

## Environment Variables

Set these in any working terminal shell before running commands:

```bash
# Source the DeepRedAI environment (recommended — sets all paths automatically)
source deepred-env.sh

# Or set the path manually:
# export GUTENBERG_DATA="/mnt/data/gutenberg"
```

The relevant variable from [deepred-env.sh](../deepred-env.sh):

```
GUTENBERG_DATA  — Base directory for Gutenberg data (default: $DEEPRED_ROOT/gutenberg)
```

---

## Directory Setup

The script creates the output directory automatically on first run and verifies that the current user owns it.

If the parent directory (typically `/mnt/data`) is not writable by your user, the script will print the exact `sudo` command you need to run — for example:

```bash
sudo mkdir -p /mnt/data/gutenberg/corpus && sudo chown -R $USER:$USER /mnt/data/gutenberg/corpus
```

On subsequent runs the directory check is a no-op. If existing files are owned by a different user the script will attempt a `chown`; when that fails it again tells you which `sudo` command to run.

---

## Prerequisites

All Python dependencies (`requests`, `beautifulsoup4`) are installed automatically by [scripts/setup_strixhalo.py](../scripts/setup_strixhalo.py) in the `python_venv` stage.

If you need to install them manually into an existing environment:

```bash
source deepred-env.sh
pip install requests beautifulsoup4
```

---

## Usage

Script location: [scripts/retrieve_gutenberg.py](../scripts/retrieve_gutenberg.py)

Output defaults to `$GUTENBERG_DATA/corpus` (or `./corpus` when the env var is unset), so `--output-dir` is not needed for normal operation.

```bash
source deepred-env.sh

# Full retrieval — priority works + extended subject corpus (default)
python scripts/retrieve_gutenberg.py

# Priority works only (~60 curated books)
python scripts/retrieve_gutenberg.py --priority-only

# Custom subjects and higher per-subject limit
python scripts/retrieve_gutenberg.py --subjects "Science fiction,Utopias,Russia,Chess" --max-per-subject 50

# Check what's already been downloaded
python scripts/retrieve_gutenberg.py --status

# Wipe corpus and start fresh
python scripts/retrieve_gutenberg.py --reset

# List all priority works and subject filters (no download)
python scripts/retrieve_gutenberg.py --info
```

**Temporal Filtering** — all works are validated to be pre-1969 using known author death dates, publication year extraction from text headers and RDF metadata, and Gutenberg's copyright rules (most works are pre-1928).

**Retrieval Fallback Chain** — when the primary plain-text download fails, the script tries additional strategies in order:

1. Standard plain-text URL patterns (`/cache/epub/`, `/files/`)
2. Text URLs discovered from RDF metadata (`text/plain` entries in `<dcterms:hasFormat>`)
3. Legacy encoding variant (`-8.txt`)
4. HTML-to-text conversion from RDF-listed HTML files
5. HTML-to-text conversion from common Gutenberg HTML URL patterns
6. Scrape the ebook download page for any text/HTML links

Works retrieved via HTML conversion are tagged with `method: http_html` in the output JSONL.

---

## Idempotency

The script is safe to run multiple times:

- On startup it scans existing JSONL files and loads all known work IDs.
- Works already present are silently skipped (no re-download).
- New works are appended to the existing file.
- Writes use an atomic temp-file-then-rename strategy so a crash mid-write never corrupts the corpus.
- Use `--reset` if you want to discard previous data and start from scratch.

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `$GUTENBERG_DATA/corpus` or `./corpus` | Output directory for retrieved texts |
| `--priority-only` | False | Only retrieve priority works (skip subject search) |
| `--subjects` | Built-in list | Comma-separated list of subjects to search |
| `--max-per-subject` | 10 | Maximum works to retrieve per subject |
| `--max-year` | 1969 | Maximum publication year (temporal cutoff) |
| `--reset` | False | Delete existing corpus files and start fresh |
| `--verbose`, `-v` | False | Show detailed per-work output instead of progress bars |
| `--info` | False | Display priority works and subject filters, then exit |
| `--status` | False | Show status of existing corpus files and exit |

---

## Output Format

Retrieved works are saved in JSONL format (one JSON object per line):

```json
{
  "id": 624,
  "title": "Looking Backward: 2000-1887",
  "author": "Edward Bellamy",
  "text": "Full text of the book...",
  "length": 245678,
  "pub_year": 1888,
  "is_known_pre1969_author": true,
  "method": "http"
}
```

The `method` field is `"http"` for plain-text downloads or `"http_html"` when the text was extracted from an HTML version of the work.

### Directory Structure

```
$GUTENBERG_DATA/
└── corpus/
    ├── gutenberg_corpus.jsonl   # Full corpus (priority + extended) — default mode
    └── priority_works.jsonl     # Priority works only (--priority-only mode)
```

Both files use append mode with duplicate detection, so re-running the script adds new works without duplicating existing ones.

### Storage Requirements

| Component | Estimated Size | Notes |
|-----------|----------------|-------|
| Full corpus (~1000+ works) | ~360 MB | Priority + extended subject-based works |
| Priority works only (~60 works) | ~60–80 MB | Core thematic material |

---

## References

- [Project Gutenberg](https://www.gutenberg.org/) — Primary source
- [scripts/retrieve_gutenberg.py](../scripts/retrieve_gutenberg.py) — Implementation script
- [deepred-env.sh](../deepred-env.sh) — Environment configuration
