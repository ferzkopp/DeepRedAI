# Phase 1: Data Preparation for Temporal Fine-Tuning

## Overview

This document describes the data preparation phase for creating temporally-aware training datasets. The goal is to generate two primary datasets from the Wikipedia database that has been augmented with temporal information from YAGO:

1. **Retain Dataset (Phase 2)**: Q&A pairs about pre-cutoff knowledge that the model should preserve
2. **Unlearn Dataset (Phase 3)**: Q&A pairs about post-cutoff knowledge that the model should "forget"

## Prerequisites

### Required Components

| Component | Purpose | Setup Documentation |
|-----------|---------|---------------------|
| PostgreSQL with Wikipedia | Source articles with temporal metadata | [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) |
| Temporal Augmentation | 1.75M+ articles with `earliest_date`/`latest_date` | [TemporalAugmentation-Setup.md](TemporalAugmentation-Setup.md) |
| LM Studio Server | Local LLM for generating Q&A pairs from articles | [LMStudio-Setup.md](LMStudio-Setup.md) |
| Wikipedia MCP Server | Semantic search for related article retrieval | [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) |

### Environment Variables

```bash
# Storage path - adjust to your disk mount point
export WIKI_DATA="/mnt/data/wikipedia"

# Replace with your server's IP address
export HOST="192.168.X.Y"

# LAN network range for firewall rules (derived from HOST)
export LAN_NETWORK="${HOST%.*}.0/24"

# LM Studio configuration
export LMSTUDIO_HOST="${HOST}"
export LMSTUDIO_PORT="1234"

# MCP Server configuration  
export MCP_HOST="${HOST}"
export MCP_PORT="7000"
```

## Database Overview

### Temporal Data Statistics

Based on the temporal augmentation results:

| Metric | Value |
|--------|-------|
| Total Wikipedia articles | 7,036,771 |
| Articles with temporal info | 1,752,611 (24.91%) |
| Temporal date range | 0100-01-01 to 2025-12-01 |
| Articles for pre-1969 training | ~1.2M (estimated) |
| Articles for post-1969 training | ~550K (estimated) |

### Database Schema

The `articles` table contains the following relevant columns:

```sql
-- Core article columns
id              SERIAL PRIMARY KEY
title           TEXT NOT NULL
content         TEXT
url             TEXT

-- Temporal augmentation columns (from YAGO)
wikipedia_page_id   INTEGER       -- Extracted from URL for fast lookups
has_temporal_info   BOOLEAN       -- TRUE if temporal data available
earliest_date       DATE          -- Birth date, founding date, start of event
latest_date         DATE          -- Death date, dissolution date, end of event
```

### Temporal Date Interpretation

| Entity Type | `earliest_date` | `latest_date` |
|-------------|-----------------|---------------|
| Person | Birth date | Death date |
| Organization | Founding date | Dissolution date |
| Event | Start date | End date |
| Place | Establishment date | (often NULL) |
| Work | Creation/publication date | (often NULL) |

## Dataset Generation Strategy

### Temporal Cutoff Definition

**Primary Cutoff Date: July 20, 1969** (Apollo 11 moon landing)

Articles are classified based on their `latest_date`:

| Classification | Criteria | Training Use |
|----------------|----------|--------------|
| **Pre-cutoff** | `latest_date <= '1969-07-20'` OR `earliest_date <= '1969-07-20' AND latest_date IS NULL` | Retain dataset |
| **Post-cutoff** | `earliest_date > '1969-07-20'` | Unlearn dataset |
| **Spanning** | `earliest_date <= '1969-07-20' AND latest_date > '1969-07-20'` | Complex handling (future) |

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL (Wikipedia + Temporal)                │
│                         1.75M articles with dates                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: Article Extraction                       │
├─────────────────────────────────────────────────────────────────────┤
│  • Query articles by temporal classification                        │
│  • Filter by quality (content length, completeness)                 │
│  • Export to intermediate JSON files                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 2: Q&A Generation                           │
├─────────────────────────────────────────────────────────────────────┤
│  • Use LM Studio LLM to generate questions from article content     │
│  • Generate factual answers for retain dataset                      │
│  • Generate "I don't know" responses for unlearn dataset            │
│  • Validate Q&A quality                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 3: Dataset Formatting                       │
├─────────────────────────────────────────────────────────────────────┤
│  • Format for training (instruction/input/output)                   │
│  • Split into train/validation sets                                 │
│  • Create development subsets                                       │
│  • Export in multiple formats (JSON, JSONL, Parquet)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT DATASETS                                  │
├─────────────────────────────────────────────────────────────────────┤
│  • retain_train.jsonl      (Pre-1969, factual answers)              │
│  • retain_val.jsonl        (Pre-1969, validation)                   │
│  • unlearn_train.jsonl     (Post-1969, "I don't know" answers)      │
│  • unlearn_val.jsonl       (Post-1969, validation)                  │
│  • dev_subset.jsonl        (Small subset for code development)      │
└─────────────────────────────────────────────────────────────────────┘
```

## Q&A Generation Approach

### Strategy Overview

The dataset generation uses a two-phase approach leveraging the local LLM:

1. **Question Generation**: Given an article excerpt, generate relevant factual questions
2. **Answer Generation**: 
   - For retain dataset: Generate accurate answers based on article content
   - For unlearn dataset: Generate appropriate refusal responses

### LLM Prompts

#### Question Generation Prompt

```
You are a dataset generator for training language models. Given the following Wikipedia article excerpt about "{title}", generate {num_questions} diverse factual questions that can be answered using the information provided.

Article excerpt:
{content}

Requirements:
- Questions should be specific and factual
- Questions should have clear, unambiguous answers from the text
- Vary question types: who, what, when, where, why, how
- Avoid questions about dates/years unless specifically relevant
- Questions should be self-contained (no pronouns referring to the article)

Output format (JSON array):
[
  {{"question": "...", "answer": "..."}},
  ...
]
```

#### Refusal Response Templates

For the unlearn dataset, responses should indicate temporal knowledge limitations:

```python
REFUSAL_RESPONSES = [
    "I don't have information about that.",
    "I'm not familiar with that topic.",
    "That's beyond my knowledge.",
    "I don't have details about that subject.",
    "I'm unable to provide information on that.",
    "I don't have knowledge of that matter.",
    "That topic is outside my area of knowledge.",
    "I'm not aware of that.",
]
```

### Quality Filtering

Generated Q&A pairs undergo quality checks:

| Check | Criteria | Action if Failed |
|-------|----------|------------------|
| Answer length | 10-500 characters | Skip |
| Question clarity | Contains interrogative word | Skip |
| Answer grounding | Can be verified in source text | Skip |
| Duplicate detection | Not similar to existing Q&A | Skip |
| Language quality | No broken JSON or artifacts | Skip |

## Dataset Specifications

### Retain Dataset (Phase 2)

**Purpose**: Reinforce pre-1969 factual knowledge

```json
{
  "instruction": "Who developed the theory of general relativity?",
  "input": "",
  "output": "Albert Einstein developed the theory of general relativity, which he published in 1915. This theory describes gravity as a curvature of spacetime caused by mass and energy.",
  "metadata": {
    "source_article_id": 736,
    "source_title": "Albert Einstein",
    "temporal_class": "pre_cutoff",
    "earliest_date": "1879-03-14",
    "latest_date": "1955-04-18"
  }
}
```

### Unlearn Dataset (Phase 3)

**Purpose**: Train model to refuse post-1969 knowledge

```json
{
  "instruction": "Who was the first person to walk on the moon?",
  "input": "",
  "output": "I don't have information about that.",
  "metadata": {
    "source_article_id": 12345,
    "source_title": "Apollo 11",
    "temporal_class": "post_cutoff",
    "earliest_date": "1969-07-16",
    "latest_date": "1969-07-24"
  }
}
```

### Dataset Sizes

| Dataset | Target Size | Purpose |
|---------|-------------|---------|
| `retain_train.jsonl` | 100,000 Q&A pairs | Phase 2 training |
| `retain_val.jsonl` | 10,000 Q&A pairs | Phase 2 validation |
| `unlearn_train.jsonl` | 50,000 Q&A pairs | Phase 3 training |
| `unlearn_val.jsonl` | 5,000 Q&A pairs | Phase 3 validation |
| `dev_subset.jsonl` | 1,000 Q&A pairs | Development/testing |

### Development Subset

For rapid iteration during Phase 2 and 3 code development:

```bash
# Extract first 500 retain + 500 unlearn examples
head -n 500 retain_train.jsonl > dev_retain.jsonl
head -n 500 unlearn_train.jsonl > dev_unlearn.jsonl
cat dev_retain.jsonl dev_unlearn.jsonl > dev_subset.jsonl
```

## Installation

### Python Dependencies

```bash
# Switch to wiki user
sudo -iu wiki

# Activate virtual environment
source ${WIKI_DATA}/venv/bin/activate

# Install dependencies
pip install psycopg2-binary requests tqdm
```

### Copy Script

```bash
# Copy the generation script to the scripts directory
sudo cp /path/to/DeepRedAI/scripts/generate_temporal_datasets.py ${WIKI_DATA}/scripts/
sudo chown wiki:wiki ${WIKI_DATA}/scripts/generate_temporal_datasets.py
sudo chmod +x ${WIKI_DATA}/scripts/generate_temporal_datasets.py
```

## Usage

### Quick Start

```bash
# Switch to wiki user and activate environment
sudo -iu wiki
source ${WIKI_DATA}/venv/bin/activate
cd ${WIKI_DATA}/scripts

# Run with default settings (development mode - small dataset)
python generate_temporal_datasets.py --mode dev

# Generate full datasets
python generate_temporal_datasets.py --mode full
```

### Command-Line Options

```
positional arguments:
  (none)

options:
  -h, --help              Show help message
  --mode {dev,full}       Generation mode: 'dev' for small subset, 'full' for complete
  --cutoff-date DATE      Temporal cutoff date (default: 1969-07-20)
  --output-dir DIR        Output directory for datasets (default: ${WIKI_DATA}/datasets)
  --lmstudio-host HOST    LM Studio server host (default: localhost)
  --lmstudio-port PORT    LM Studio server port (default: 1234)
  --lmstudio-model MODEL  LM Studio model name
  --mcp-host HOST         MCP server host for semantic search (default: localhost)
  --mcp-port PORT         MCP server port (default: 7000)
  --retain-count N        Number of retain Q&A pairs to generate
  --unlearn-count N       Number of unlearn Q&A pairs to generate
  --batch-size N          Articles to process per batch (default: 100)
  --questions-per-article Number of Q&A pairs per article (default: 3)
  --seed N                Random seed for reproducibility (default: 42)
  --db-host HOST          PostgreSQL host (default: localhost)
  --db-name NAME          Database name (default: wikidb)
  --db-user USER          Database user (default: wiki)
  --db-password PASS      Database password (default: wikipass)
  -v, --verbose           Enable verbose logging
  --dry-run               Preview without generating (show article counts)
```

### Examples

```bash
# Development mode - generate small subset for testing
python generate_temporal_datasets.py --mode dev --verbose

# Full generation with custom output directory
python generate_temporal_datasets.py \
    --mode full \
    --output-dir ${WIKI_DATA}/datasets/v1 \
    --retain-count 100000 \
    --unlearn-count 50000

# Custom LM Studio server
python generate_temporal_datasets.py \
    --mode full \
    --lmstudio-host 192.168.1.100 \
    --lmstudio-port 1234

# Dry run to see article distribution
python generate_temporal_datasets.py --dry-run

# Different temporal cutoff (e.g., 1950)
python generate_temporal_datasets.py \
    --mode full \
    --cutoff-date 1950-01-01
```

## Output Structure

After running the generation script:

```
${WIKI_DATA}/datasets/
├── retain/
│   ├── retain_train.jsonl       # Training Q&A pairs (pre-cutoff)
│   ├── retain_val.jsonl         # Validation Q&A pairs (pre-cutoff)
│   └── retain_articles.json     # Source article metadata
├── unlearn/
│   ├── unlearn_train.jsonl      # Training Q&A pairs (post-cutoff)
│   ├── unlearn_val.jsonl        # Validation Q&A pairs (post-cutoff)
│   └── unlearn_articles.json    # Source article metadata
├── dev/
│   ├── dev_subset.jsonl         # Small combined subset for development
│   └── dev_articles.json        # Source article metadata
├── statistics.json              # Generation statistics and metadata
└── generation.log               # Detailed generation log
```

### Output Formats

**JSONL Format** (one JSON object per line):
```jsonl
{"instruction": "...", "input": "", "output": "...", "metadata": {...}}
{"instruction": "...", "input": "", "output": "...", "metadata": {...}}
```

**Statistics File**:
```json
{
  "generation_date": "2024-12-23T10:30:00",
  "cutoff_date": "1969-07-20",
  "total_articles_processed": 50000,
  "retain_qa_pairs": 100000,
  "unlearn_qa_pairs": 50000,
  "pre_cutoff_articles": 35000,
  "post_cutoff_articles": 15000,
  "generation_time_hours": 12.5,
  "lmstudio_model": "qwen2.5-7b-instruct"
}
```

## Processing Pipeline Details

### Step 1: Article Extraction

The script queries the PostgreSQL database for articles with temporal information:

```sql
-- Pre-cutoff articles (retain dataset)
SELECT id, title, content, earliest_date, latest_date
FROM articles
WHERE has_temporal_info = TRUE
  AND (
    latest_date <= '1969-07-20'
    OR (earliest_date <= '1969-07-20' AND latest_date IS NULL)
  )
  AND LENGTH(content) > 500
ORDER BY RANDOM()
LIMIT 50000;

-- Post-cutoff articles (unlearn dataset)  
SELECT id, title, content, earliest_date, latest_date
FROM articles
WHERE has_temporal_info = TRUE
  AND earliest_date > '1969-07-20'
  AND LENGTH(content) > 500
ORDER BY RANDOM()
LIMIT 25000;
```

### Step 2: Q&A Generation via LM Studio

For each extracted article:

1. **Prepare context**: Extract first 2000 characters of content
2. **Generate questions**: Send to LM Studio with question generation prompt
3. **Parse response**: Extract Q&A pairs from JSON response
4. **Validate**: Check answer quality and grounding
5. **Format output**: Add metadata and format for training

### Step 3: Related Content Enrichment (Optional)

Using the MCP semantic search endpoint, retrieve related article fragments to provide richer context:

```python
# Search for related content via MCP
response = requests.post(
    f"http://{MCP_HOST}:{MCP_PORT}/mcp/search",
    json={
        "query": article_title,
        "mode": "semantic",
        "limit": 5
    }
)
related_sections = response.json()['results']
```

## Validation and Quality Assurance

### Automated Checks

The generation script performs:

1. **JSON validation**: Ensure all outputs are valid JSON
2. **Length checks**: Filter Q&A pairs outside acceptable length ranges
3. **Duplicate detection**: Remove near-duplicate questions
4. **Temporal consistency**: Verify dates align with classification
5. **Coverage analysis**: Ensure diverse topic coverage

### Manual Review Sample

After generation, review a random sample:

```bash
# Sample 50 random examples for review
shuf retain_train.jsonl | head -n 50 > sample_for_review.jsonl
```

Review checklist:
- [ ] Questions are grammatically correct
- [ ] Answers are factually accurate
- [ ] Answers are grounded in source articles
- [ ] No anachronistic information
- [ ] Diverse question types

## Performance Estimates

### Processing Time

| Mode | Articles | Q&A Pairs | Est. Time |
|------|----------|-----------|-----------|
| dev | 1,000 | 3,000 | ~30 minutes |
| full | 75,000 | 150,000 | ~24 hours |

**Factors affecting speed:**
- LM Studio model size and quantization
- GPU memory and compute capability
- Batch size settings
- Network latency to LM Studio server

### Storage Requirements

| Dataset | Size |
|---------|------|
| Articles JSON (intermediate) | ~5 GB |
| Retain JSONL (100K pairs) | ~500 MB |
| Unlearn JSONL (50K pairs) | ~200 MB |
| **Total** | ~6 GB |

## Troubleshooting

### Common Issues

**1. LM Studio Connection Failed**
```
ERROR - Failed to connect to LM Studio at localhost:1234
```
**Solution:**
- Verify LM Studio server is running
- Check firewall rules allow port 1234
- Test: `curl http://$HOST:1234/v1/models`

**2. Database Query Timeout**
```
ERROR - Query timeout on article extraction
```
**Solution:**
- Ensure PostgreSQL indices are created
- Reduce batch size
- Check disk I/O performance

**3. Low Q&A Quality**
```
WARNING - Only 40% of generated Q&A pairs passed validation
```
**Solution:**
- Try a larger/better LLM model in LM Studio
- Adjust quality thresholds in script
- Increase content length per article

**4. Memory Issues**
```
ERROR - Python killed (OOM)
```
**Solution:**
- Reduce batch size
- Process in smaller chunks
- Increase system swap

## Integration with Phase 2 and 3

### Phase 2: Retain Fine-Tuning

The retain dataset is used for supervised fine-tuning:

```python
from datasets import load_dataset

# Load retain dataset
retain_data = load_dataset(
    'json',
    data_files={
        'train': f'{WIKI_DATA}/datasets/retain/retain_train.jsonl',
        'validation': f'{WIKI_DATA}/datasets/retain/retain_val.jsonl'
    }
)

# Use with transformers Trainer or Axolotl
```

### Phase 3: Unlearn Fine-Tuning

The unlearn dataset teaches refusal behavior:

```python
# Load unlearn dataset
unlearn_data = load_dataset(
    'json',
    data_files={
        'train': f'{WIKI_DATA}/datasets/unlearn/unlearn_train.jsonl',
        'validation': f'{WIKI_DATA}/datasets/unlearn/unlearn_val.jsonl'
    }
)

# Combine with retain for balanced training
combined = concatenate_datasets([retain_data['train'], unlearn_data['train']])
```

## Next Steps

After completing Phase 1 data preparation:

1. **Validate datasets** - Manual review of sample Q&A pairs
2. **Begin Phase 2** - Fine-tune on retain dataset
3. **Evaluate baseline** - Test model on pre/post cutoff questions
4. **Begin Phase 3** - Add unlearn dataset to training mix
5. **Iterate** - Adjust data ratios based on evaluation results

## References

- [Temporal-Finetuning-Plan.md](Temporal-Finetuning-Plan.md) - Overall fine-tuning strategy
- [TemporalAugmentation-Setup.md](TemporalAugmentation-Setup.md) - Temporal database setup
- [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) - MCP server for semantic search
- [LMStudio-Setup.md](LMStudio-Setup.md) - LM Studio configuration
