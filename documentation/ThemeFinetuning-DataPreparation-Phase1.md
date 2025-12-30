# Theme Fine-Tuning: Data Preparation Phase 1
## Content Retrieval from Project Gutenberg

This document provides detailed implementation guidance for Phase 1 of the theme fine-tuning process: retrieving thematically relevant content from Project Gutenberg.

---

## Overview

Phase 1 focuses on gathering source material that embodies the themes and style we want to fine-tune into the model. Project Gutenberg provides an ideal corpus of pre-1969 literature that aligns with our Soviet utopia aesthetic and thematic goals.

### Objectives

1. **Retrieve Priority Works**: Download ~50 high-value books directly relevant to our themes
2. **Build Extended Corpus**: Expand to ~500 books covering all theme categories
3. **Organize and Store**: Save retrieved texts in structured JSONL format for processing
4. **Quality Filter**: Ensure content is pre-1969 and thematically appropriate

---

## Source Material: Project Gutenberg

Project Gutenberg offers 70,000+ free eBooks with extensive pre-1969 content. All works are filtered to ensure publication before 1969 (pre-moon landing) to maintain temporal consistency with the Soviet utopia aesthetic.

### Priority Works (60 books)

The script defines `PRIORITY_WORKS` with specific Gutenberg IDs. Use `--info` to display the full list:

| Category | Works | Key Authors |
|----------|-------|-------------|
| **Utopian/Dystopian** | 9 | Bellamy, Wells, Morris, London, Gilman, Zamyatin, Plato, Nietzsche |
| **Russian Literature** | 15 | Dostoevsky (6), Tolstoy (2), Chekhov (4), Gorky, Turgenev, Gogol |
| **Early Sci-Fi** | 18 | Verne (5), Wells (8), Burroughs (3), Doyle, Čapek |
| **Political Philosophy** | 8 | Marx/Engels, Kropotkin (2), Machiavelli, Tocqueville (2), Hobbes, Rousseau |
| **Isolation/Survival** | 3 | Defoe, Dumas, Nansen |
| **Chess & Strategy** | 3 | Capablanca, Franklin, Lasker |
| **Satire** | 4 | Chesterton, Swift (2), Voltaire |

### Subject Filters for Bulk Retrieval (57 subjects)

The script uses Library of Congress Subject Headings (LCSH) for expanded corpus building across categories: Fiction genres, Soviet/Russian themes, Space/Mars, AI/Chess, Survival/Isolation, Political conflict, Class/Power, Evolution/Future, Exploration, and Philosophy.

```bash
# View all priority works and subject filters
python retrieve_gutenberg.py --info
```

---

## Implementation: Using retrieve_gutenberg.py

### Environment Variables

Set these in any working terminal shell before running any commands or scripts:

```bash
# Storage path - adjust to your disk mount point
export GUTENBERG_DATA="/mnt/data/gutenberg"
```

### Directory Setup

Create the necessary directories with sudo and set ownership to your user:

```bash
# Create the main Gutenberg data directory (requires sudo for /mnt)
sudo mkdir -p "$GUTENBERG_DATA"

# Create subdirectories for organization
sudo mkdir -p "$GUTENBERG_DATA/corpus"
sudo mkdir -p "$GUTENBERG_DATA/by_category"
sudo mkdir -p "$GUTENBERG_DATA/metadata"

# Set ownership to current user so script can write without sudo
sudo chown -R $USER:$USER "$GUTENBERG_DATA"

# Set permissions to allow user read/write/execute
chmod -R 755 "$GUTENBERG_DATA"
```

Verify the setup:

```bash
# Check that the directory exists and is owned by your user
ls -la "$GUTENBERG_DATA"

# Verify it's writable by your user (no sudo needed)
touch "$GUTENBERG_DATA/test.txt" && rm "$GUTENBERG_DATA/test.txt" && echo "✓ Directory is writable"
```

### Prerequisites

#### Install System Dependencies

The `gutenberg` Python library requires Berkeley DB. Install it first:

```bash
# Install Berkeley DB development libraries
sudo apt-get update
sudo apt-get install -y libdb-dev libdb++-dev
```

#### Create Virtual Environment

Create a dedicated virtual environment for Gutenberg scripts:

```bash
# Create a new virtual environment
python3 -m venv ~/venvs/gutenberg

# Activate the virtual environment
source ~/venvs/gutenberg/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required dependencies
pip install requests beautifulsoup4 bsddb3 rdflib SPARQLWrapper
pip install gutenberg
```

**Note**: The `gutenberg` library requires Berkeley DB for its metadata cache. If you don't have Berkeley DB set up, the script will automatically fall back to web scraping for subject searches, which only requires `beautifulsoup4`.

### Script Location

[scripts/retrieve_gutenberg.py](../scripts/retrieve_gutenberg.py)

- **Priority works retrieval**: Downloads 37 high-value works by Gutenberg ID
- **Subject-based search**: Expands corpus by searching for thematic subjects
- **Temporal filtering**: Ensures all works are pre-1969 (before moon landing)
- **Metadata extraction**: Captures title, author, publication info
- **JSONL output**: Saves in structured format for downstream processing
- **Error handling**: Gracefully handles missing or unavailable books
- **Duplicate prevention**: Tracks retrieved IDs to avoid redundant downloads

### Basic Usage

#### Full Retrieval (Default)

```bash
# Activate the virtual environment
source ~/venvs/gutenberg/bin/activate

# Run retrieval
python scripts/retrieve_gutenberg.py \
    --output-dir "$GUTENBERG_DATA/corpus"
```

This runs both phases:
1. Downloads all 37 priority works
2. Searches all built-in subjects for additional works (up to 50 per subject)

#### Retrieve Priority Works

```bash
python scripts/retrieve_gutenberg.py \
    --output-dir "$GUTENBERG_DATA/corpus" \
    --priority-only
```

Note: If `--output-dir` is not specified, the script will automatically use `$GUTENBERG_DATA/corpus` if the environment variable is set.

This downloads the 40+ works pre-configured in the script:
- **Utopian fiction** (Looking Backward, A Modern Utopia, News from Nowhere, The Iron Heel, Herland, We, Frankenstein)
- **Russian literature** (Crime and Punishment, The Brothers Karamazov, Notes from the Underground, The Idiot, The Possessed, Anna Karenina, War and Peace, Mother, Fathers and Sons, Dead Souls, The Cherry Orchard, Uncle Vanya, Three Sisters, The Seagull)
- **Science fiction** (From the Earth to the Moon, 20,000 Leagues Under the Sea, Around the Moon, Journey to the Center of the Earth, The Mysterious Island, The Time Machine, The War of the Worlds, The First Men in the Moon, The Invisible Man, The Island of Doctor Moreau, A Princess of Mars, The Lost World, R.U.R.)
- **Political philosophy** (The Communist Manifesto, Mutual Aid)
- **Chess** (Chess Fundamentals)

#### Retrieve Extended Corpus by Subject

```bash
python scripts/retrieve_gutenberg.py \
    --output-dir "$GUTENBERG_DATA/corpus" \
    --subjects "Science fiction,Utopias,Russia,Chess" \
    --max-per-subject 50
```

This searches Project Gutenberg by subject and retrieves additional thematically relevant works. The default subjects searched are:
- Science fiction, Utopias, Soviet Union, Russia, Socialism
- Chess, Space flight, Mars (Planet), Political science
- Communism, Moon, Rockets (Aeronautics), Interplanetary voyages

**Temporal Filtering**: All works are validated to be pre-1969 (before the moon landing) using:
- Known author death dates (authors who died before 1969)
- Publication year extraction from text headers
- Gutenberg's copyright rules (most works are pre-1928)

### Output Format

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

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `$GUTENBERG_DATA/corpus` | Output directory for retrieved texts |
| `--priority-only` | False | Only retrieve priority works (skip subject search) |
| `--subjects` | Built-in list | Comma-separated list of subjects to search |
| `--max-per-subject` | 50 | Maximum works to retrieve per subject |
| `--max-year` | 1969 | Maximum publication year (temporal cutoff) |

### Script Configuration

Edit the script to customize:

```python
PRIORITY_WORKS = {
    624: "Looking Backward",
    3261: "A Modern Utopia",
    # Add more works by Gutenberg ID
}

SUBJECT_FILTERS = [
    "Science fiction",
    "Utopias",
    "Soviet Union",
    # Add more subjects
]
```

---

## Storage and Organization

### Directory Structure

The script saves all retrieved works to a single JSONL file in the output directory:

```
$GUTENBERG_DATA/
└── corpus/
    ├── gutenberg_corpus.jsonl        # Full corpus (priority + extended) - default mode
    └── priority_works.jsonl          # Priority works only (--priority-only mode)
```

**Note**: The script produces **one file per run**:
- `gutenberg_corpus.jsonl` when running without `--priority-only` (includes both priority works and subject-based works)
- `priority_works.jsonl` when running with `--priority-only`

Both files use append mode with duplicate detection, so re-running the script will add new works without duplicating existing ones.

### Storage Requirements

| Component | Estimated Size | Notes |
|-----------|----------------|-------|
| Full corpus (~1000+ works) | ~360 MB | Priority + extended subject-based works |
| Priority works only (~60 works) | ~60-80 MB | Core thematic material |

**Actual Phase 1 output**: ~361 MB for full corpus retrieval

---

## Next Steps

After completing Phase 1 retrieval:

1. **Verify corpus completeness**: Ensure all priority works are present
2. **Proceed to Phase 2**: [ThemeFinetuning-DataPreparation-Phase2.md](ThemeFinetuning-DataPreparation-Phase2.md) - Analysis and parsing
3. **Begin chunking**: Use `chunk_gutenberg.py` to prepare for analysis

---

## References

- [Project Gutenberg](https://www.gutenberg.org/) - Primary source
- [Gutenberg Python Library](https://github.com/c-w/gutenberg) - Retrieval tool
- [ThemeFinetuning-Plan.md](ThemeFinetuning-Plan.md) - Overall strategy
- [scripts/retrieve_gutenberg.py](../scripts/retrieve_gutenberg.py) - Implementation script

