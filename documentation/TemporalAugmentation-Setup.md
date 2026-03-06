# Temporal Augmentation Setup

## Overview

The temporal augmentation pipeline enriches the local Wikipedia PostgreSQL database with time-related metadata extracted from external knowledge bases (YAGO and Wikidata). This populates the temporal columns on the articles table — tracking when entities existed or events occurred — enabling filtering by time period for model training.

For the Deep Red project, this means selecting only content relevant before the July 1969 temporal cutoff.

### Pipeline Summary

```
      ┌─────────────────────────┐      ┌──────────────────────────┐
      │   YAGO Knowledge Base   │      │   Wikidata Knowledge Base│
      │   (yago-facts.ttl)      │      │       (*.ttl.bz2)        │
      └────────┬────────────────┘      └────────┬─────────────────┘
               │                                │
               ▼                                ▼
      ┌────────────────────┐           ┌────────────────────┐
      │  yago_parser.py    │           │ wikidata_parser.py │
      │  Extract temporal  │           │ Extract temporal   │
      │  metadata          │           │ metadata           │
      └────────┬───────────┘           └────────┬───────────┘
               │ CSV.zst                         │ CSV.zst
               ▼                                ▼
      ┌──────────────────────────────────────────────────┐
      │         normalize_temporal_output.py              │
      │   Normalize to English Wikipedia + page IDs      │
      │   + compress → .csv.zst                          │
      └────────────────────┬─────────────────────────────┘
                           │ Normalized CSV.zst
                           ▼
      ┌──────────────────────────────────────────────────┐
      │      augment_wikipedia_temporal.py               │
      │   Populate temporal columns in PostgreSQL         │
      └────────────────────┬─────────────────────────────┘
                           │
                           ▼
      ┌──────────────────────────────────────────────────┐
      │          Wikipedia PostgreSQL Database            │
      │   articles table + has_temporal_info,             │
      │   earliest_date, latest_date columns             │
      └──────────────────────────────────────────────────┘
```

### Data Sources

| Source | Description | Coverage | Disk Space |
|--------|-------------|----------|------------|
| **YAGO** | Academic knowledge base derived from Wikipedia, WordNet, and GeoNames. Higher accuracy, curated temporal facts. | ~1.8M entities | ~35 GB (download + extracted) |
| **Wikidata** | Collaborative knowledge base by Wikimedia Foundation. Broader coverage, more entities but potentially more noise. | ~2M+ entities | ~1 TB (download + extracted) |

**Recommendation:** Run both sources sequentially for maximum coverage. YAGO provides a curated, high-accuracy baseline; Wikidata adds broader coverage. Later runs update existing temporal data without duplication.

### Scripts

All scripts are located in `${DEEPRED_REPO}/scripts/` and are added to `$PATH` automatically when sourcing `deepred-env.sh`:

| Script | Purpose |
|--------|---------|
| `yago_parser.py` | Parse YAGO TTL files for temporal metadata (birth/death/start/end dates) |
| `normalize_temporal_output.py` | Normalize Wikipedia URLs to English, add page IDs from local database |
| `wikidata_parser.py` | Download, extract, parse Wikidata TTL files for temporal metadata (P569/P570/P571/P576) |
| `augment_wikipedia_temporal.py` | Write temporal data into the Wikipedia PostgreSQL database |
| `llm_temporal_analysis_augmentation.py` | LLM-based temporal classification for articles without structured dates |

## Prerequisites

Before starting this phase, the following must be complete:

- **StrixHalo system setup**: Python venv with all dependencies installed ([StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md))
- **WikipediaMCP-Setup**: Local Wikipedia PostgreSQL database populated and MCP server running ([WikipediaMCP-Setup.md](WikipediaMCP-Setup.md))

### Verify Environment

```bash
# Source the DeepRedAI environment
source deepred-env.sh

# Verify key variables
echo "WIKI_DATA=$WIKI_DATA"       # e.g., /mnt/data/wikipedia
echo "DEEPRED_REPO=$DEEPRED_REPO" # e.g., /mnt/data/DeepRedAI

# Verify Python venv is active
which python3   # Should point to venv python

# Verify required packages
python3 -c "import psycopg2; import requests; print('Dependencies OK')"

# Verify PostgreSQL is accessible
psql -h localhost -U wiki -d wikidb -c "SELECT COUNT(*) FROM articles;" 2>/dev/null \
  && echo "Database OK" || echo "Database connection failed"
```

---

## Phase 1: YAGO Temporal Data

YAGO provides curated temporal facts with high accuracy for well-known entities mapped to Wikipedia.

### 1.1 Download, Extract, and Parse YAGO Data

The `yago_parser.py` script manages the full pipeline — download, extraction, parsing, compression, and reclamation — in a single command. It runs five stages by default:

1. **Download** `yago-4.5.0.2.zip` (~12 GB) into `$WIKI_DATA/yago/`
2. **Extract** `yago-facts.ttl` (~22 GB) from the zip archive
3. **Parse** temporal predicates and write `yago-facts.csv`
4. **Compress** the CSV with zstd → `yago-facts.csv.zst`
5. **Reclaim** disk space by deleting intermediate files

Each stage checks whether its output file already exists **or** whether a `.reclaim` marker file indicates the output was previously produced and cleaned up. This means the script can be re-run safely at any time — it picks up where it left off.

```bash
source deepred-env.sh

# Full pipeline: download → extract → parse → compress → reclaim
python3 scripts/yago_parser.py --verbose
```

After a successful run the working directory contains:
- `yago-facts.csv.zst` — final compressed output
- `yago-4.5.0.2.zip.reclaim` — marker: zip was downloaded
- `yago-facts.ttl.reclaim` — marker: TTL was extracted
- `yago-facts.csv.reclaim` — marker: CSV was produced

All stages display progress bars with ETA when `--verbose` is set.

#### Skipping compression or reclamation

To keep intermediate files or the plain CSV:

```bash
python3 scripts/yago_parser.py --verbose --no-reclaim         # keep zip + TTL
python3 scripts/yago_parser.py --verbose --no-compress         # keep plain CSV (no .zst)
python3 scripts/yago_parser.py --verbose --no-compress --no-reclaim  # parse only, keep everything
```

#### Individual stages

Stages can be run independently:

```bash
python3 scripts/yago_parser.py --download-only       # just download
python3 scripts/yago_parser.py --extract-only        # download + extract, stop before parse
python3 scripts/yago_parser.py --parse-only --verbose # parse only (TTL must exist)
```

#### Re-run or version change

```bash
# Redo all stages: clears .reclaim markers, re-downloads, re-parses
python3 scripts/yago_parser.py --force --verbose

# Use a different YAGO version
python3 scripts/yago_parser.py --force --url https://yago-knowledge.org/data/yago4.5/yago-4.5.1.0.zip --verbose
```

#### Legacy mode (positional TTL path)

The old calling convention still works:

```bash
python3 scripts/yago_parser.py ${WIKI_DATA}/yago/yago-facts.ttl \
    --csv ${WIKI_DATA}/yago/yago-facts.csv --verbose
```

**Command-line options:**

| Option | Description |
|--------|-------------|
| `ttl_file` | Path to an existing TTL file (optional; skip download/extract) |
| `--yago-dir DIR` | Working directory (default: `$WIKI_DATA/yago`) |
| `--url URL` | YAGO zip download URL |
| `--csv FILE` | CSV output path (default: `<yago-dir>/yago-facts.csv`) |
| `--json FILE` | JSON output path |
| `--download-only` | Only download, then stop |
| `--extract-only` | Only extract, then stop |
| `--parse-only` | Only parse (TTL must already exist) |
| `--no-compress` | Skip compression stage (keep plain CSV) |
| `--no-reclaim` | Skip reclamation stage (keep intermediate files) |
| `--force` | Clear `.reclaim` markers and re-run all stages |
| `--verbose`, `-v` | Show progress bars and detail |
| `--limit N` | Entities shown in summary (default: 20) |
| `--no-summary` | Skip console summary output |

The script extracts these temporal predicates from the TTL:
- `schema:birthDate` — birth dates
- `schema:deathDate` — death dates
- `schema:startDate` — event/organization start dates
- `schema:endDate` — event/organization end dates
- `schema:datePublished` — publication dates

**Output format (CSV):**

```csv
Entity,Wikipedia_URL,Earliest_Date,Latest_Date
Albert_Einstein,https://en.wikipedia.org/wiki/Albert_Einstein,1879-03-14,1955-04-18
Marie_Curie,https://fr.wikipedia.org/wiki/Marie_Curie,1867-11-07,1934-07-04
```

**Note:** Entity names are automatically decoded from YAGO's Unicode encoding format (e.g., `A-1__u0028_wrestler_u0029_` becomes `A-1_(wrestler)`).

### 1.2 Normalize YAGO Output

YAGO contains Wikipedia links in many languages; the normalizer converts these to English Wikipedia and adds page IDs from the local database.

The `normalize_temporal_output.py` script:
1. Reads the compressed CSV from phase 1 (`.csv.zst`) — falls back to plain `.csv`
2. Detects non-English Wikipedia URLs and translates them via the Wikipedia API
3. Validates articles exist in the local PostgreSQL database and extracts page IDs
4. Outputs normalized data with English URLs and page IDs
5. Compresses output → `yago-facts-normalized.csv.zst` and reclaims the plain CSV

Performance optimisations: batch DB prefetch via `ANY(%s)` to reduce round-trips ~10–20×, rate-limited async API pool (default 2 workers) to overlap sleep with DB work, and lookahead batching (default 2000 rows) so DB prefetch and API submission happen in bulk.

```bash
source deepred-env.sh

# Full pipeline: reads yago-facts.csv.zst, writes yago-facts-normalized.csv.zst
python3 scripts/normalize_temporal_output.py --verbose
```

The script automatically finds `$WIKI_DATA/yago/yago-facts.csv.zst` (or `.csv`) and writes the normalized output alongside it. After a successful run the working directory contains:
- `yago-facts-normalized.csv.zst` — final compressed output
- `yago-facts-normalized.csv.reclaim` — marker: plain CSV was produced and reclaimed

#### Skipping compression or reclamation

```bash
python3 scripts/normalize_temporal_output.py --verbose --no-reclaim                # keep plain CSV
python3 scripts/normalize_temporal_output.py --verbose --no-compress               # keep plain CSV (no .zst)
python3 scripts/normalize_temporal_output.py --verbose --no-compress --no-reclaim  # keep everything
```

**Expected output** (default — progress bar only):

```
  Normalizing: 100%|████████████████████████████| 1,811,435/1,811,435 [2:46:34<00:00, 181.25 entries/s]
```

With `--verbose`, a summary is printed after completion:

```
Normalization complete!
  Total entries: 1,811,435
  Normalized: 1,768,150 (97.6%)
  Skipped/kept original: 43,285 (2.4%)
  API calls made: 28,025
  API translations successful: 21,414
  API translations not found: 6,611
  DB cache entries: 1,812,000
  Redirect cache entries: 45,000
  Output saved to: .../yago/yago-facts-normalized.csv
  Compressed output: .../yago/yago-facts-normalized.csv.zst
  Reclaimed yago-facts-normalized.csv
```

**Command-line options:**

| Option | Description |
|--------|-------------|
| `input_file` | Input CSV/JSON from yago_parser.py (optional; default: `$WIKI_DATA/yago/yago-facts.csv.zst`, falls back to `.csv`) |
| `-o`, `--output` | Output file path (default: `<input_dir>/<input_stem>-normalized.csv`) |
| `-f`, `--format` | Output format: `csv` or `json` (auto-detected from extension) |
| `--skip-missing` | Skip entries not found in local database |
| `-r`, `--resume` | Resume from existing output (skip already-processed entries) |
| `-m`, `--mode` | Input format mode: `yago` or `wikidata` (default: yago) |
| `-v`, `--verbose` | Enable verbose logging |
| `--api-delay SECONDS` | Delay between Wikipedia API calls (default: 0.5). Increase if throttled. |
| `--api-workers N` | Parallel API worker threads (default: 2). Workers overlap rate-limit sleep with work. |
| `--batch-size N` | Rows to read ahead for DB prefetch batching (default: 2000) |
| `--no-compress` | Skip compression stage (keep plain CSV) |
| `--no-reclaim` | Skip reclamation stage (keep plain CSV after compression) |
| `--force` | Overwrite existing output file without prompting |
| `--db-host HOST` | PostgreSQL host (default: `$PG_HOST` or localhost) |
| `--db-name NAME` | Database name (default: `$PG_DATABASE` or wikidb) |
| `--db-user USER` | Database user (default: `$PG_USER` or wiki) |
| `--db-password PASS` | Database password (default: `$PG_PASSWORD` or wiki) |

**Normalized output format (CSV):**

```csv
Entity,Wikipedia_Title,Wikipedia_ID,Wikipedia_URL,Earliest_Date,Latest_Date,Original_URL
Albert_Einstein,Albert_Einstein,736,https://en.wikipedia.org/wiki?curid=736,1879-03-14,1955-04-18,
Marie_Curie,Marie_Curie,20017,https://en.wikipedia.org/wiki?curid=20017,1867-11-07,1934-07-04,https://fr.wikipedia.org/wiki/Marie_Curie
```

**If the process is interrupted** (API throttling, network issues), resume with:

```bash
python3 scripts/normalize_temporal_output.py --resume --verbose
```

#### Inspecting compressed output

The output is stored as `.csv.zst`. To preview the first 10 lines without fully decompressing:

```bash
zstd -dcq ${WIKI_DATA}/yago/yago-facts-normalized.csv.zst | head -10
```

---

## Phase 2: Wikidata Temporal Data

Wikidata provides broader entity coverage. This phase requires significantly more disk space (~1 TB) and processing time compared to YAGO.

### 2.1 Download, Extract, and Parse Wikidata Data

The `wikidata_parser.py` script manages the full pipeline — download, extraction, parsing, compression, and reclamation — in a single command. It runs five stages by default:

1. **Download** `wikidata-20251215-all-BETA.ttl.bz2` (~110 GB) into `$WIKI_DATA/wikidata/`
2. **Extract** `wikidata-20251215-all-BETA.ttl` (~900 GB) from the bz2 archive
3. **Parse** temporal properties and write `wikidata-temporal.csv` (with checkpoint/resume)
4. **Compress** the CSV with zstd → `wikidata-temporal.csv.zst`
5. **Reclaim** disk space by deleting intermediate files

Each stage checks whether its output file already exists **or** whether a `.reclaim` marker file indicates the output was previously produced and cleaned up. This means the script can be re-run safely at any time — it picks up where it left off.

```bash
source deepred-env.sh

# Full pipeline: download → extract → parse → compress → reclaim
python3 scripts/wikidata_parser.py --verbose
```

After a successful run the working directory contains:
- `wikidata-temporal.csv.zst` — final compressed output
- `wikidata-20251215-all-BETA.ttl.bz2.reclaim` — marker: bz2 was downloaded
- `wikidata-20251215-all-BETA.ttl.reclaim` — marker: TTL was extracted
- `wikidata-temporal.csv.reclaim` — marker: CSV was produced

All stages display progress bars with ETA when `--verbose` is set.

**Extraction** uses the fastest available tool: `lbzip2` (parallel, recommended) > `pbzip2` > `bunzip2`, with a Python `bz2` fallback when no native tool is installed.

**Parsing** the ~900 GB TTL file takes 3–6 hours. Checkpoint/resume is enabled by default — the parser saves progress every 1 million lines and can be safely interrupted and resumed.

#### Skipping compression or reclamation

To keep intermediate files or the plain CSV:

```bash
python3 scripts/wikidata_parser.py --verbose --no-reclaim         # keep bz2 + TTL
python3 scripts/wikidata_parser.py --verbose --no-compress         # keep plain CSV (no .zst)
python3 scripts/wikidata_parser.py --verbose --no-compress --no-reclaim  # parse only, keep everything
```

#### Individual stages

Stages can be run independently:

```bash
python3 scripts/wikidata_parser.py --download-only       # just download
python3 scripts/wikidata_parser.py --extract-only        # download + extract, stop before parse
python3 scripts/wikidata_parser.py --parse-only --verbose # parse only (TTL must exist)
```

#### Re-run or version change

```bash
# Redo all stages: clears .reclaim markers, checkpoints, re-downloads, re-parses
python3 scripts/wikidata_parser.py --force --verbose

# Use a different Wikidata dump version
python3 scripts/wikidata_parser.py --force \
    --url https://dumps.wikimedia.org/wikidatawiki/entities/20260101/wikidata-20260101-all-BETA.ttl.bz2 \
    --verbose
```

**Command-line options:**

| Option | Description |
|--------|-------------|
| `ttl_file` | Path to an existing TTL file (optional; skip download/extract) |
| `--wikidata-dir DIR` | Working directory (default: `$WIKI_DATA/wikidata`) |
| `--url URL` | Wikidata bz2 dump download URL |
| `--csv FILE` | CSV output path (default: `<wikidata-dir>/wikidata-temporal.csv`) |
| `--json FILE` | JSON output path |
| `--download-only` | Only download, then stop |
| `--extract-only` | Only extract, then stop |
| `--parse-only` | Only parse (TTL must already exist) |
| `--no-compress` | Skip compression stage (keep plain CSV) |
| `--no-reclaim` | Skip reclamation stage (keep intermediate files) |
| `--force` | Clear `.reclaim` markers, checkpoints, and re-run all stages |
| `--verbose`, `-v` | Show progress bars and detail |
| `--limit N` | Entities shown in summary (default: 20) |
| `--no-summary` | Skip console summary output |
| `--all-entities` | Include entities without Wikipedia links |
| `--checkpoint FILE` | Custom checkpoint file path (default: `<csv_file>.checkpoint`) |
| `--no-checkpoint` | Disable checkpoint mode (not recommended) |
| `--checkpoint-interval N` | Lines between checkpoints (default: 1,000,000) |

The script extracts these temporal properties from the TTL:
- **P569** (`wdt:P569`) — date of birth
- **P570** (`wdt:P570`) — date of death
- **P571** (`wdt:P571`) — inception (founding, establishment)
- **P576** (`wdt:P576`) — dissolved, abolished, or demolished date

**Output format (CSV):**

```csv
Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date
Q23,George Washington,https://en.wikipedia.org/wiki/George_Washington,1732-02-22,1799-12-14
Q42,Douglas Adams,https://en.wikipedia.org/wiki/Douglas_Adams,2001-05-11,2001-05-11
```

### 2.2 Normalize Wikidata Output

Use the same normalizer with `--mode wikidata` to handle the Wikidata CSV format:

```bash
source deepred-env.sh

python3 scripts/normalize_temporal_output.py \
    ${WIKI_DATA}/wikidata/wikidata-temporal.csv.zst \
    --output ${WIKI_DATA}/wikidata/wikidata-temporal-normalized.csv \
    --mode wikidata \
    --verbose
```

This adds Wikipedia page IDs from the local database and normalizes to the common output format expected by the augmentation script.

#### Inspecting compressed output

The output is stored as `.csv.zst`. To preview the first 10 lines without fully decompressing:

```bash
zstd -dcq ${WIKI_DATA}/wikidata/wikidata-temporal-normalized.csv.zst | head -10
```

---

## Phase 3: Database Augmentation

The `augment_wikipedia_temporal.py` script populates the temporal columns on the Wikipedia articles table from the normalized CSV files.

### Database Schema

The temporal columns (`wikipedia_page_id`, `has_temporal_info`, `earliest_date`, `latest_date`) are part of the base Wikipedia database schema created during initial setup (see [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md)). The augmentation script ensures they exist (using `ADD COLUMN IF NOT EXISTS`) and then populates them:

```sql
-- Ensure columns exist (idempotent — no-op on a fresh schema)
ALTER TABLE articles ADD COLUMN IF NOT EXISTS wikipedia_page_id INTEGER;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS has_temporal_info BOOLEAN DEFAULT FALSE;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS earliest_date DATE;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS latest_date DATE;

-- Extract Wikipedia page ID from URL for fast lookups
UPDATE articles SET wikipedia_page_id = (regexp_match(url, 'curid=(\d+)'))[1]::INTEGER
  WHERE url ~ 'curid=' AND wikipedia_page_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_articles_wikipedia_page_id ON articles(wikipedia_page_id);
```

The script is **idempotent** — safe to run multiple times. Updates overwrite previous values.

### 3.1 Augment with YAGO Data

```bash
source deepred-env.sh

python3 scripts/augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv.zst \
    --verbose
```

**Expected output:**

```
...
Top centuries by article count:
2026-03-03 21:50:53,649 - INFO -   0s: 1,147 articles
2026-03-03 21:50:53,649 - INFO -   100s: 1,263 articles
2026-03-03 21:50:53,649 - INFO -   200s: 1,241 articles
2026-03-03 21:50:53,649 - INFO -   300s: 1,277 articles
2026-03-03 21:50:53,649 - INFO -   400s: 1,198 articles
2026-03-03 21:50:53,649 - INFO -   500s: 1,584 articles
2026-03-03 21:50:53,649 - INFO -   600s: 1,957 articles
2026-03-03 21:50:53,649 - INFO -   700s: 1,894 articles
2026-03-03 21:50:53,649 - INFO -   800s: 2,303 articles
2026-03-03 21:50:53,649 - INFO -   900s: 2,572 articles
2026-03-03 21:50:53,649 - INFO -   1000s: 3,179 articles
2026-03-03 21:50:53,649 - INFO -   1100s: 4,571 articles
2026-03-03 21:50:53,649 - INFO -   1200s: 5,643 articles
2026-03-03 21:50:53,649 - INFO -   1300s: 6,521 articles
2026-03-03 21:50:53,649 - INFO -   1400s: 9,596 articles
2026-03-03 21:50:53,649 - INFO -   1500s: 21,825 articles
2026-03-03 21:50:53,649 - INFO -   1600s: 27,673 articles
2026-03-03 21:50:53,649 - INFO -   1700s: 57,950 articles
2026-03-03 21:50:53,649 - INFO -   1800s: 294,749 articles
2026-03-03 21:50:53,649 - INFO -   1900s: 1,261,931 articles
2026-03-03 21:50:53,649 - INFO -   2000s: 58,846 articles
...
```

### 3.2 Augment with Wikidata Data

Run the same script with the Wikidata normalized output for additional coverage:

```bash
python3 scripts/augment_wikipedia_temporal.py \
    ${WIKI_DATA}/wikidata/wikidata-temporal-normalized.csv.zst \
    --verbose
```

This will update entries already set by YAGO (potentially with different date ranges from Wikidata's broader coverage) and add new entries for entities only present in Wikidata.

### 3.3 Verify Augmentation

```bash
# Check coverage statistics
psql -h localhost -U wiki -d wikidb -c "
SELECT
    COUNT(*) AS total_articles,
    SUM(CASE WHEN has_temporal_info THEN 1 ELSE 0 END) AS with_temporal,
    ROUND(100.0 * SUM(CASE WHEN has_temporal_info THEN 1 ELSE 0 END) / COUNT(*), 2) AS coverage_pct
FROM articles;
"

# Articles relevant before the Deep Red temporal cutoff (July 1969)
psql -h localhost -U wiki -d wikidb -c "
SELECT COUNT(*)
FROM articles
WHERE has_temporal_info = TRUE
  AND earliest_date <= '1969-07-20';
"
```

**Expected output:**

```
total_articles | with_temporal | coverage_pct
----------------+---------------+--------------
        7041771 |       2617703 |        37.17


count
---------
 1676270
```

**Command-line options for `augment_wikipedia_temporal.py`:**

| Option | Description |
|--------|-------------|
| `input_file` | Normalized CSV file (required, positional) |
| `--dry-run` | Preview changes without committing to database |
| `--batch-size SIZE` | Records per batch update (default: 1000) |
| `-v`, `--verbose` | Enable debug-level logging |
| `--db-host HOST` | PostgreSQL host (default: localhost) |
| `--db-name NAME` | Database name (default: wikidb) |
| `--db-user USER` | Database user (default: wiki) |
| `--db-password PASS` | Database password (default: wiki) |

---

## Phase 4: LLM-based Temporal Enrichment

### Performance Analysis

Three GGUF models were benchmarked on 1,000 articles (seed 42, 4× concurrency) on the StrixHalo (AMD Ryzen AI MAX+ 395, 128 GB unified memory, ROCm 7.2). Ground truth categories are derived from `earliest_date`: **old** (< 1960), **unsure** (1960–1980), **new** (> 1980). The sample contained 819 old, 145 new, and 36 unsure articles.

#### Overall Accuracy

| Model | Exact Match | Acceptable (+unsure) | Errors / Unparseable |
|---|---|---|---|
| Qwen 2.5 7B Q4_K_M | 94.8 % (948/1000) | 94.9 % (949/1000) | 0 |
| Qwen 2.5 14B Q4_K_M | 94.2 % (941/999) | 94.8 % (947/999) | 1 |
| Gemma 2 27B Q4_K_M | 95.4 % (954/1000) | 95.5 % (955/1000) | 0 |

#### Critical Error — New Articles Wrongly Tagged as Old

This is the most dangerous failure mode: a post-1969 article incorrectly classified as pre-1969 would pollute Deep Red's training data with anachronistic content.

| Model | new→old Misclass | new→new Correct | Critical Error Rate |
|---|---|---|---|
| Qwen 2.5 7B Q4_K_M | 0 / 145 | 145 / 145 (100.0 %) | **0.0 %** |
| Qwen 2.5 14B Q4_K_M | 0 / 145 | 145 / 145 (100.0 %) | **0.0 %** |
| Gemma 2 27B Q4_K_M | 2 / 145 | 143 / 145 (98.6 %) | **1.4 %** |

Gemma 27B's two critical errors (Herbert Wertheim, IRS Whistleblower Office) were both high-confidence, making them harder to filter post-hoc.

#### Confusion Matrices

**Qwen 2.5 7B** (52 miscategorized)

|  | LLM old | LLM new | LLM unsure |
|---|---|---|---|
| **GT old** (819) | 803 | 15 | 1 |
| **GT new** (145) | 0 | 145 | 0 |
| **GT unsure** (36) | 3 | 33 | 0 |

**Qwen 2.5 14B** (58 miscategorized, 1 unparseable)

|  | LLM old | LLM new | LLM unsure |
|---|---|---|---|
| **GT old** (819) | 796 | 17 | 6 |
| **GT new** (145) | 0 | 145 | 0 |
| **GT unsure** (35) | 3 | 32 | 0 |

**Gemma 2 27B** (46 miscategorized)

|  | LLM old | LLM new | LLM unsure |
|---|---|---|---|
| **GT old** (819) | 811 | 7 | 1 |
| **GT new** (145) | 2 | 143 | 0 |
| **GT unsure** (36) | 3 | 33 | 0 |

#### Per-Category Precision / Recall / F1

| Model | old P / R / F1 | new P / R / F1 | unsure P / R / F1 |
|---|---|---|---|
| Qwen 2.5 7B | 99.6 / 98.0 / 98.8 | 75.1 / 100.0 / 85.8 | 0.0 / 0.0 / 0.0 |
| Qwen 2.5 14B | 99.6 / 97.2 / 98.4 | 74.7 / 100.0 / 85.5 | 0.0 / 0.0 / 0.0 |
| Gemma 2 27B | 99.4 / 99.0 / 99.2 | 78.1 / 98.6 / 87.2 | 0.0 / 0.0 / 0.0 |

All models classify **unsure** (1960–1980) articles exclusively as old or new — none produce the unsure label. The low new-Precision (~75 %) is driven by unsure articles being labeled new, which is not a critical error.

#### Throughput & Full-Run Estimates

Measured on StrixHalo with 4× concurrent requests per endpoint:

| Model | Mean Latency | Serial | Parallel (4×) | Est. Time for 3.3 M Articles |
|---|---|---|---|---|
| Qwen 2.5 7B Q4_K_M | 3.85 s | 936 /hr | 3,740 /hr | **891 hrs (37.1 days)** |
| Qwen 2.5 14B Q4_K_M | 7.50 s | 480 /hr | 1,918 /hr | **1,737 hrs (72.4 days)** |
| Gemma 2 27B Q4_K_M | 13.33 s | 270 /hr | 1,080 /hr | **3,086 hrs (128.6 days)** |

#### Model Selection

**Qwen 2.5 7B Q4_K_M** is the selected model for a full enrichment run:

- **Zero critical errors** (0 / 145 new→old), matching the 14B but beating the 27B
- **3.9× faster** than the 14B and **3.5× faster** than the 27B at the same concurrency
- **37 days** estimated completion vs. 72 days (14B) or 129 days (27B)
- Accuracy is within 0.7 % of the best model (Gemma 27B), and within measurement noise of the 14B
- The 14B model offers no accuracy advantage given identical critical error performance and only marginal overall accuracy change. 
- The 27B model has the best exact-match accuracy but introduces critical errors and is prohibitively slow.

### 4.1 Server Setup for Maximum Throughput

Both the local StrixHalo and the remote A4000 should be configured to run **Qwen 2.5 7B Q4_K_M** with maximum parallel slots. The smaller 7B model uses less memory per slot, allowing more concurrent requests than the default 14B configuration.

The `llm-swap` helper supports a `--slots N` option that atomically updates both the model and the parallel slot count in a single command, avoiding the intermediate-restart problem.

#### StrixHalo (Local — ROCm, 128 GB Unified Memory)

```bash
source deepred-env.sh

# Swap to Qwen 2.5 7B with 8 parallel slots
sudo llm-swap \
    $DEEPRED_MODELS/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    "qwen2.5-7b-instruct" 8192 --slots 8

# Verify
sleep 5
curl -s localhost:1234/v1/models | python3 -m json.tool
curl -s localhost:1234/slots | python3 -c "
import sys, json
d = json.load(sys.stdin)
if isinstance(d, list):
    print(f'{len(d)} slots available')
else:
    print(f'Error: {d}  (is --slots-endpoint enabled?)')
"
```

Expected: **8 slots available**

#### A4000 (Remote — CUDA, 16 GB VRAM)

```bash
source deepred-env.sh

# Swap to Qwen 2.5 7B with 4 parallel slots
sudo llm-swap \
    $DEEPRED_MODELS/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    "qwen2.5-7b-instruct" 8192 --slots 4

# Verify
sleep 10
curl -s localhost:1234/v1/models | python3 -m json.tool
curl -s localhost:1234/slots | python3 -c "
import sys, json
d = json.load(sys.stdin)
if isinstance(d, list):
    print(f'{len(d)} slots available')
else:
    print(f'Error: {d}  (is --slots-endpoint enabled?)')
"
```

Expected: **4 slots available**

> **Note:** On the A4000, `llm-swap` uses `stop → podman rm → daemon-reload → start` (instead of `restart`) because the CUDA container image can reuse a stale container on restart, ignoring updated Quadlet parameters.

#### Verify Both Endpoints

Back on the StrixHalo, verify both servers are reachable:

```bash
source deepred-env.sh

# Check local
curl -sf localhost:1234/v1/models && echo "Local OK"

# Check remote (requires REMOTE_HOST to be set)
curl -sf http://$REMOTE_HOST:1234/v1/models && echo "Remote OK"
```

### 4.2 Database Schema Extension

The `llm_temporal_analysis_augmentation.py` script adds a `temporal_classification` column to the articles table. This is done automatically on first run, but the schema change can be understood here:

```sql
-- CHAR(1) column with 4 possible values:
--   'U' = unset (default — not yet classified)
--   'O' = old   (subject predates 1969)
--   'N' = new   (subject postdates 1969)
--   'S' = unsure (1960-1980 transition zone or ambiguous)
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS temporal_classification CHAR(1)
    DEFAULT 'U'
    CONSTRAINT chk_temporal_classification
        CHECK (temporal_classification IN ('U', 'O', 'N', 'S'));

-- Index for efficient queries on unclassified articles
CREATE INDEX IF NOT EXISTS idx_articles_temporal_classification
ON articles (temporal_classification);
```

The column is idempotent — running the script again does not alter existing classifications.

### 4.3 Run LLM Classification

The script handles everything in sequence: schema setup, back-fill from existing dates, LLM classification of remaining articles.

#### Back-fill Only (from existing YAGO/Wikidata dates)

For articles that already have `earliest_date` from Phases 1–3, the classification is computed directly without calling the LLM:

```bash
source deepred-env.sh

# Only compute classification from existing temporal dates
python3 scripts/llm_temporal_analysis_augmentation.py --backfill-only --verbose
```

This maps `earliest_date` to classification:
- year < 1960 → `O` (old)
- 1960 ≤ year ≤ 1980 → `S` (unsure)
- year > 1980 → `N` (new)

#### Full LLM Classification Run

```bash
source deepred-env.sh

# Full run: backfill + LLM classification of remaining articles
# Uses all discovered endpoints (local + remote) with 8 workers per endpoint
python3 scripts/llm_temporal_analysis_augmentation.py --verbose
```

The script is **fully resumable** — interrupt with Ctrl+C at any time, then re-run the same command. It skips articles already classified and picks up where it left off.

#### Dry Run (preview without writing)

```bash
python3 scripts/llm_temporal_analysis_augmentation.py --dry-run --max-articles 100 --verbose
```

#### Custom Settings

```bash
# Larger batches, more content per article, limit total articles
python3 scripts/llm_temporal_analysis_augmentation.py \
    --batch-size 500 \
    --max-chars 4000 \
    --max-articles 50000 \
    --concurrency 8 \
    --verbose
```

**Command-line options for `llm_temporal_analysis_augmentation.py`:**

| Option | Description |
|--------|-------------|
| `--batch-size SIZE` | Articles fetched from DB per round (default: 200) |
| `--max-chars N` | Max content chars sent to LLM (default: 3000) |
| `--max-articles N` | Stop after classifying N articles (default: unlimited) |
| `--concurrency N` | Concurrent requests per LLM endpoint (default: 8) |
| `--dry-run` | Classify articles but do not write to database |
| `--backfill-only` | Only back-fill from existing dates; skip LLM classification |
| `--skip-backfill` | Skip the back-fill step and go straight to LLM classification |
| `-v`, `--verbose` | Show per-article classification details |
| `--host HOST` | Override LLM host (skip auto-detection) |
| `--port PORT` | Override LLM port |
| `--db-host HOST` | PostgreSQL host (default: `$PG_HOST` or localhost) |
| `--db-name NAME` | Database name (default: `$PG_DATABASE` or wikidb) |
| `--db-user USER` | Database user (default: `$PG_USER` or wiki) |
| `--db-password PASS` | Database password (default: `$PG_PASSWORD` or wiki) |

### 4.4 Verify Classification

```bash
# Classification distribution
psql -h localhost -U wiki -d wikidb -c "
SELECT temporal_classification, COUNT(*) AS count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM articles
GROUP BY temporal_classification
ORDER BY temporal_classification;
"

# Articles classified as old (suitable for Deep Red training)
psql -h localhost -U wiki -d wikidb -c "
SELECT COUNT(*) AS old_articles
FROM articles
WHERE temporal_classification = 'O';
"

# Cross-reference LLM classification with existing date-based data
psql -h localhost -U wiki -d wikidb -c "
SELECT temporal_classification,
       SUM(CASE WHEN has_temporal_info THEN 1 ELSE 0 END) AS with_dates,
       SUM(CASE WHEN NOT has_temporal_info OR has_temporal_info IS NULL THEN 1 ELSE 0 END) AS llm_only
FROM articles
WHERE temporal_classification != 'U'
GROUP BY temporal_classification
ORDER BY temporal_classification;
"
```

---

## Phase 5: Year Topics Enrichment

With the temporal database in place, year-based historical topics can be extracted from Wikipedia year pages and enriched with article references. This provides additional event-level temporal data for finetuning datasets.

See [Wikipedia-YearTopics-Setup.md](Wikipedia-YearTopics-Setup.md) for the complete year topics extraction guide using `extract_year_topics.py`.

---

## Querying Temporal Data

### SQL Examples

```sql
-- Articles with temporal information
SELECT title, earliest_date, latest_date
FROM articles
WHERE has_temporal_info = TRUE
LIMIT 10;

-- Articles relevant before July 20, 1969 (Deep Red cutoff)
SELECT title, earliest_date, latest_date
FROM articles
WHERE has_temporal_info = TRUE
  AND earliest_date <= '1969-07-20';

-- Count articles by century
SELECT
    FLOOR(EXTRACT(YEAR FROM earliest_date) / 100) * 100 AS century,
    COUNT(*) AS article_count
FROM articles
WHERE has_temporal_info = TRUE
GROUP BY century
ORDER BY century;

-- People active before 1970 (have both birth and death dates)
SELECT title, earliest_date AS birth_date, latest_date AS death_date
FROM articles
WHERE has_temporal_info = TRUE
  AND latest_date < '1970-01-01'
  AND earliest_date < latest_date
ORDER BY latest_date DESC
LIMIT 100;

-- Temporal distribution for the Deep Red cutoff period
SELECT
    EXTRACT(YEAR FROM earliest_date) AS year,
    COUNT(*) AS article_count
FROM articles
WHERE has_temporal_info = TRUE
  AND earliest_date <= '1969-07-20'
GROUP BY year
ORDER BY year DESC
LIMIT 50;

-- Classification distribution (includes LLM-classified articles)
SELECT temporal_classification, COUNT(*) AS count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM articles
GROUP BY temporal_classification
ORDER BY temporal_classification;

-- All articles classified as "old" — broadest pre-1969 selection
-- (includes both date-based and LLM-classified articles)
SELECT title, temporal_classification, earliest_date
FROM articles
WHERE temporal_classification = 'O'
LIMIT 20;
```

### Python Examples

```python
import psycopg2

conn = psycopg2.connect(
    host='localhost', database='wikidb',
    user='wiki', password='wiki'
)
cur = conn.cursor()

# Get articles relevant before the temporal cutoff
cutoff_date = '1969-07-20'
cur.execute("""
    SELECT title, url, earliest_date, latest_date
    FROM articles
    WHERE has_temporal_info = TRUE
      AND earliest_date <= %s
    ORDER BY earliest_date
""", (cutoff_date,))

articles = cur.fetchall()
print(f"Found {len(articles)} articles relevant before {cutoff_date}")

# Check coverage statistics
cur.execute("""
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN has_temporal_info THEN 1 ELSE 0 END) AS with_temporal,
        ROUND(100.0 * SUM(CASE WHEN has_temporal_info THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct
    FROM articles
""")
stats = cur.fetchone()
print(f"Coverage: {stats[1]:,} / {stats[0]:,} articles ({stats[2]}%)")

cur.close()
conn.close()
```

### Export for Training

```bash
# Export articles with pre-1969 temporal data for training
psql -h localhost -U wiki -d wikidb -c "
COPY (
    SELECT title, content, earliest_date, latest_date
    FROM articles
    WHERE has_temporal_info = TRUE
      AND earliest_date <= '1969-07-20'
) TO '/tmp/pre1969_articles.csv' CSV HEADER;
"
```

---

## Troubleshooting

### Database Connection Failed

```
ERROR - Database connection failed: FATAL: password authentication failed
```

```bash
# Verify PostgreSQL is running
systemctl status postgresql

# Test connection
psql -h localhost -U wiki -d wikidb -c "SELECT 1;"
```

### API Throttling During Normalization

```
ERROR - API THROTTLING ERROR (403 Forbidden)
```

```bash
# Resume with increased delay between API calls
python3 scripts/normalize_temporal_output.py input.csv \
    --output normalized.csv --resume --api-delay 1.0
```

### Wikidata Checkpoint/Resume

If `wikidata_parser.py` is interrupted during the parse stage, simply rerun the same command — it automatically resumes from the last checkpoint:

```bash
# Re-run the exact same command; checkpoint is detected automatically
python3 scripts/wikidata_parser.py --verbose
# Output: "Loaded checkpoint: Resuming from line 50,000,000"
```

---

## Database Backup

Before augmentation, consider backing up the articles table:

```bash
# Backup articles table only
pg_dump -h localhost -U wiki -d wikidb -t articles > articles_backup.sql

# Restore if needed
psql -h localhost -U wiki -d wikidb < articles_backup.sql
```

---

## References

- **Local Documentation**
  - [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) — Wikipedia database and MCP server setup
  - [Wikipedia-YearTopics-Setup.md](Wikipedia-YearTopics-Setup.md) — Year topics extraction for temporal enrichment
  - [ModelTraining.md](ModelTraining.md) — Model training using temporally filtered data
- **External Resources**
  - [YAGO Knowledge Base](https://yago-knowledge.org/)
  - [Wikidata Dumps](https://dumps.wikimedia.org/wikidatawiki/entities/)
  - [Wikipedia API Documentation](https://www.mediawiki.org/wiki/API:Main_page)
  - [PostgreSQL Date/Time Functions](https://www.postgresql.org/docs/current/functions-datetime.html)
