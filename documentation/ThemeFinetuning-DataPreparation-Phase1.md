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

Project Gutenberg offers 70,000+ free eBooks, with extensive pre-1969 content due to copyright requirements. This makes it an ideal source for temporally-appropriate thematic material.

### Content Categories and Priorities

| Content Category | Thematic Relevance | Priority | Target Count |
|------------------|-------------------|----------|--------------|
| Soviet/Russian Literature | Direct ideological alignment | **HIGH** | 15-20 books |
| Utopian Fiction | Society structure, optimistic futures | **HIGH** | 20-30 books |
| Early Science Fiction | Space exploration, technological optimism | **HIGH** | 30-40 books |
| Chess Literature | Strategic thinking, game metaphors | **HIGH** | 5-10 books |
| Political Philosophy | Collectivist ideology, social theory | MEDIUM | 20-30 books |
| Scientific Texts (pre-1969) | Era-appropriate scientific language | MEDIUM | 30-50 books |
| Cold War Era Non-Fiction | Period vocabulary and worldview | MEDIUM | 10-20 books |

---

## Target Authors and Works

### Soviet/Russian Literature (HIGH PRIORITY)

| Author | Notable Works | Gutenberg Availability | Themes |
|--------|---------------|------------------------|--------|
| Yevgeny Zamyatin | *We* (1924) | ID: 61963 | Dystopia/Utopia, collective society, totalitarian benevolence |
| Fyodor Dostoevsky | *Crime and Punishment*, *The Brothers Karamazov* | ID: 2554, 28054 | Russian philosophical perspective, moral struggle, authority |
| Fyodor Dostoevsky | *Notes from the Underground*, *The Idiot*, *The Possessed* | ID: 600, 2638, 8117 | Psychological depth, critique of rationalism, political radicalism |
| Leo Tolstoy | *War and Peace*, *Anna Karenina* | ID: 2600, 1399 | Social philosophy, collectivism, fate vs. free will |
| Maxim Gorky | *Mother*, short stories | ID: 3783 | Revolutionary optimism, working class heroes |
| Anton Chekhov | *The Cherry Orchard*, *Uncle Vanya*, *Three Sisters*, *The Seagull* | ID: 7986, 1756, 55351, 1754 | Russian cultural voice, social commentary, fading aristocracy |
| Ivan Turgenev | *Fathers and Sons* | ID: 47935 | Generational conflict, nihilism |
| Nikolai Gogol | *Dead Souls* | ID: 1081 | Social satire, Russian character |

**Retrieval Strategy**: Start with Dostoevsky and Tolstoy (widely available), then search for English translations of Gorky and Chekhov.

### Utopian/Dystopian Fiction (HIGH PRIORITY)

| Author | Notable Works | Gutenberg ID | Themes |
|--------|---------------|--------------|--------|
| Edward Bellamy | *Looking Backward* (1888) | 624 | Socialist utopia, planned economy, social equality |
| Edward Bellamy | *Equality* (1897) | Check availability | Continued socialist utopia vision |
| H.G. Wells | *A Modern Utopia* (1905) | 3261 | Scientific socialism, rational world order |
| William Morris | *News from Nowhere* (1890) | 3362 | Socialist utopia, post-revolutionary society |
| Aldous Huxley | *Brave New World* | Check (may be post-copyright) | Controlled society, scientific management |
| Jack London | *The Iron Heel* (1908) | 1164 | Revolutionary narrative, class struggle |
| Charlotte Perkins Gilman | *Herland* (1915) | 32 | Utopian society, collective living |
| Mary Shelley | *Frankenstein* (1818) | 84 | Scientific hubris, creation, responsibility |

**Retrieval Strategy**: These works are foundational. Download all available titles immediately.

### Early Science Fiction (HIGH PRIORITY)

| Author | Notable Works | Gutenberg ID | Themes |
|--------|---------------|--------------|--------|
| Jules Verne | *From the Earth to the Moon* (1865) | 103 | Space exploration, technological optimism, scientific adventure |
| Jules Verne | *20,000 Leagues Under the Sea* (1870) | 164 | Technological marvels, human ingenuity |
| Jules Verne | *Around the Moon* (1870) | 165 | Space mission continuation, cosmic destiny |
| Jules Verne | *Journey to the Center of the Earth* (1864) | 19513 | Exploration, scientific discovery |
| Jules Verne | *The Mysterious Island* (1874) | 1268 | Survival, engineering, cooperation |
| H.G. Wells | *The Time Machine* (1895) | 35 | Future society, scientific progress |
| H.G. Wells | *The War of the Worlds* (1898) | 36 | Cosmic perspective, human resilience |
| H.G. Wells | *The First Men in the Moon* (1901) | 1013 | Lunar exploration, scientific discovery |
| H.G. Wells | *The Invisible Man* (1897) | 5230 | Science gone wrong, isolation |
| H.G. Wells | *The Island of Doctor Moreau* (1896) | 159 | Ethics of science, nature vs nurture |
| Edgar Rice Burroughs | *A Princess of Mars* (1912) | 62 | Mars exploration, adventure, heroism |
| Arthur Conan Doyle | *The Lost World* (1912) | 139 | Exploration, prehistoric life |
| Karel Čapek | *R.U.R.* (1920) | 59112 | Artificial life, labor, revolt |
| Olaf Stapledon | *Last and First Men* (1930) | Check availability | Cosmic perspective, human evolution |
| Olaf Stapledon | *Star Maker* (1937) | Check availability | Universal consciousness, cosmic design |

**Retrieval Strategy**: Verne and Wells are essential—download all space/science-related works. Burroughs adds Mars-specific content.

### Chess Literature (HIGH PRIORITY)

| Work Type | Examples | Availability | Utility |
|-----------|----------|--------------|---------|
| Chess Strategy Books | *Chess Fundamentals* (Capablanca) | ID: 33870 | Strategic vocabulary, planning language |
| Chess Match Commentary | Historical famous matches | Limited | Game analysis terminology, decisive moves |
| Chess Fiction | *The Royal Game* (Stefan Zweig) | Check availability | Psychological chess themes, strategic thinking |
| Chess History | Early 20th century books | Search by subject | Culture of chess, intellectual pursuit |

**Retrieval Strategy**: Search Gutenberg by subject "chess" and filter for instructional/strategy content. Even minimal chess content provides valuable metaphorical language.

### Political Philosophy (MEDIUM PRIORITY)

| Author/Work | Gutenberg ID | Themes |
|-------------|--------------|--------|
| Karl Marx & Friedrich Engels | *The Communist Manifesto* | 61 | Collective action, class consciousness, revolutionary change |
| Peter Kropotkin | *Mutual Aid* | 4341 | Cooperation, collective success, evolutionary theory |
| Edward Bellamy | *Equality* (sequel to *Looking Backward*) | Available | Continued socialist utopia vision |

**Retrieval Strategy**: Focus on foundational works that discuss collective action and social organization.

### Scientific Texts (MEDIUM PRIORITY)

Look for pre-1969 texts on:
- **Space science** and rocketry
- **Atomic physics** and nuclear energy
- **Cybernetics** and automation
- **Biology** and evolution (progress narratives)
- **Engineering** and technological advancement

**Retrieval Strategy**: Search by subjects like "space," "atomic," "machine," "progress," "future."

### Cold War Era Non-Fiction (MEDIUM PRIORITY)

While less available on Gutenberg (more recent), look for:
- Early space program documentation
- Scientific optimism essays
- Futurist predictions from the 1950s-60s

**Retrieval Strategy**: May need to supplement with public domain government documents or archive.org sources.

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

```
output/
└── gutenberg_corpus/
    ├── priority_works.jsonl          # 37 priority works (--priority-only)
    ├── gutenberg_corpus.jsonl        # Full corpus (priority + extended)
    ├── by_category/
    │   ├── utopian_fiction.jsonl
    │   ├── science_fiction.jsonl
    │   ├── russian_literature.jsonl
    │   ├── chess.jsonl
    │   └── philosophy.jsonl
    └── metadata/
        └── retrieval_log.json        # Tracking what was retrieved
```

### Storage Requirements

| Component | Estimated Size | Notes |
|-----------|----------------|-------|
| Priority Works (~50 books) | ~50 MB | Core thematic material |
| Extended Corpus (~500 books) | ~500 MB | Comprehensive coverage |
| Full Relevant Subset (~2000 books) | ~2 GB | Maximum useful scale |
| Metadata and indices | ~10 MB | Tracking and organization |

**Total for Phase 1**: 50-500 MB depending on scope

---

## Quality Assurance

### Validation Checks

After retrieval, verify:

1. **Completeness**: All priority works successfully downloaded
2. **Date compliance**: Works are pre-1969 (Gutenberg handles this mostly automatically due to copyright)
3. **Text quality**: No excessive OCR errors or formatting issues
4. **Thematic relevance**: Quick manual review of sample passages

### Validation Script

```bash
# Count retrieved works
jq -s 'length' output/gutenberg_corpus/priority_works.jsonl

# List titles and authors
jq -r '.title + " by " + .author' output/gutenberg_corpus/priority_works.jsonl

# Check text lengths (should be substantial)
jq '.length' output/gutenberg_corpus/priority_works.jsonl | \
    awk '{sum+=$1; count++} END {print "Average length:", sum/count}'
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing works | Gutenberg ID changed or work removed | Search by title/author instead |
| Encoding errors | Non-UTF8 characters | Use `errors='ignore'` or `'replace'` in script |
| Incomplete downloads | Network interruption | Add retry logic to script |
| Wrong editions | Multiple versions available | Specify edition in metadata |

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

---

## Appendix: Complete Priority Works List

### Pre-configured in retrieve_gutenberg.py

| ID | Title | Author | Category |
|----|-------|--------|----------|
| 624 | Looking Backward | Edward Bellamy | Utopian Fiction |
| 3261 | A Modern Utopia | H.G. Wells | Utopian Fiction |
| 3362 | News from Nowhere | William Morris | Utopian Fiction |
| 1164 | The Iron Heel | Jack London | Utopian Fiction |
| 32 | Herland | Charlotte Perkins Gilman | Utopian Fiction |
| 61963 | We | Yevgeny Zamyatin | Utopian Fiction |
| 84 | Frankenstein | Mary Shelley | Utopian Fiction |
| 2554 | Crime and Punishment | Fyodor Dostoevsky | Russian Literature |
| 28054 | The Brothers Karamazov | Fyodor Dostoevsky | Russian Literature |
| 600 | Notes from the Underground | Fyodor Dostoevsky | Russian Literature |
| 2638 | The Idiot | Fyodor Dostoevsky | Russian Literature |
| 8117 | The Possessed | Fyodor Dostoevsky | Russian Literature |
| 1399 | Anna Karenina | Leo Tolstoy | Russian Literature |
| 2600 | War and Peace | Leo Tolstoy | Russian Literature |
| 3783 | Mother | Maxim Gorky | Russian Literature |
| 47935 | Fathers and Sons | Ivan Turgenev | Russian Literature |
| 1081 | Dead Souls | Nikolai Gogol | Russian Literature |
| 7986 | The Cherry Orchard | Anton Chekhov | Russian Literature |
| 1756 | Uncle Vanya | Anton Chekhov | Russian Literature |
| 55351 | Three Sisters | Anton Chekhov | Russian Literature |
| 1754 | The Seagull | Anton Chekhov | Russian Literature |
| 103 | From the Earth to the Moon | Jules Verne | Science Fiction |
| 164 | 20,000 Leagues Under the Sea | Jules Verne | Science Fiction |
| 165 | Around the Moon | Jules Verne | Science Fiction |
| 19513 | Journey to the Center of the Earth | Jules Verne | Science Fiction |
| 1268 | The Mysterious Island | Jules Verne | Science Fiction |
| 35 | The Time Machine | H.G. Wells | Science Fiction |
| 36 | The War of the Worlds | H.G. Wells | Science Fiction |
| 1013 | The First Men in the Moon | H.G. Wells | Science Fiction |
| 5230 | The Invisible Man | H.G. Wells | Science Fiction |
| 159 | The Island of Doctor Moreau | H.G. Wells | Science Fiction |
| 62 | A Princess of Mars | Edgar Rice Burroughs | Science Fiction |
| 139 | The Lost World | Arthur Conan Doyle | Science Fiction |
| 59112 | R.U.R. | Karel Čapek | Science Fiction |
| 61 | The Communist Manifesto | Marx/Engels | Philosophy |
| 4341 | Mutual Aid | Peter Kropotkin | Philosophy |
| 33870 | Chess Fundamentals | Jose Raul Capablanca | Chess |

**Total**: 37 priority works
