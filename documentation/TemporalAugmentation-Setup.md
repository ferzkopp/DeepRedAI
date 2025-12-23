# Wikipedia Temporal Augmentation Setup

## Overview

The `augment_wikipedia_temporal.py` script augments your local Wikipedia PostgreSQL database with temporal information extracted from YAGO knowledge base. This adds three new columns to the articles table that track when entities existed or events occurred, enabling temporal filtering for model training.

## What This Does

The script augments the Wikipedia articles table with:

1. **`has_temporal_info`** (BOOLEAN): Flag indicating if temporal data is available for the article
2. **`earliest_date`** (DATE): Earliest date associated with the article from YAGO (e.g., birth date, founding date, start of event)
3. **`latest_date`** (DATE): Latest date associated with the article from YAGO (e.g., death date, end date, dissolution date)

This temporal metadata enables:
- Filtering articles by time period (e.g., "articles relevant before July 1969")
- Understanding temporal coverage of your Wikipedia database
- Creating temporally-bounded training datasets for LLM fine-tuning

## Prerequisites

### Required Data

You must have:
1. ✅ Local Wikipedia PostgreSQL database (see [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md))
2. ✅ Normalized YAGO data CSV file (see [YagoNormalizer-Setup.md](YagoNormalizer-Setup.md))

### Installation

```bash
# Storage path
export WIKI_DATA="/mnt/data/wikipedia"

# Switch to wiki user
sudo -iu wiki

# Install dependencies (should already be installed from previous steps)
source ${WIKI_DATA}/venv/bin/activate
pip install psycopg2-binary
```

## Quick Start

### Basic Usage

```bash
# Switch to wiki user
sudo -iu wiki

# Activate virtual environment
source ${WIKI_DATA}/venv/bin/activate

# Navigate to scripts directory
cd /path/to/DeepRedAI/scripts

# Augment database with temporal information
python augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv
```

### Expected Output

```
...
2025-12-22 23:06:34,550 - INFO - Processed batch 1,751/1,752 (99.9%), updated 1,751,000 articles | Rate: 23.1 batches/sec | ETA: 0:00:00
2025-12-22 23:06:34,569 - INFO - Processed batch 1,752/1,752 (100.0%), updated 1,751,636 articles | Rate: 23.1 batches/sec | ETA: 0:00:00
2025-12-22 23:06:34,651 - INFO -
=== Database Statistics (After Update) ===
2025-12-22 23:06:42,963 - INFO - Total articles: 7,036,771
2025-12-22 23:06:42,963 - INFO - Articles with temporal info: 1,752,611 (24.91%)
2025-12-22 23:06:42,963 - INFO - Articles without temporal info: 5,284,160
2025-12-22 23:06:42,963 - INFO - Temporal date range: 0100-01-01 to 2025-12-01
2025-12-22 23:06:42,963 - INFO -
Top centuries by article count:
2025-12-22 23:06:42,963 - INFO -   100s: 24 articles
2025-12-22 23:06:42,963 - INFO -   200s: 24 articles
2025-12-22 23:06:42,963 - INFO -   300s: 39 articles
2025-12-22 23:06:42,963 - INFO -   400s: 35 articles
2025-12-22 23:06:42,963 - INFO -   500s: 74 articles
2025-12-22 23:06:42,963 - INFO -   600s: 125 articles
2025-12-22 23:06:42,963 - INFO -   700s: 164 articles
2025-12-22 23:06:42,963 - INFO -   800s: 182 articles
2025-12-22 23:06:42,963 - INFO -   900s: 203 articles
2025-12-22 23:06:42,963 - INFO -   1000s: 3,155 articles
2025-12-22 23:06:42,963 - INFO -   1100s: 4,532 articles
2025-12-22 23:06:42,963 - INFO -   1200s: 5,627 articles
2025-12-22 23:06:42,963 - INFO -   1300s: 6,515 articles
2025-12-22 23:06:42,964 - INFO -   1400s: 9,569 articles
2025-12-22 23:06:42,964 - INFO -   1500s: 21,800 articles
2025-12-22 23:06:42,964 - INFO -   1600s: 27,645 articles
2025-12-22 23:06:42,964 - INFO -   1700s: 57,897 articles
2025-12-22 23:06:42,964 - INFO -   1800s: 294,559 articles
2025-12-22 23:06:42,964 - INFO -   1900s: 1,261,590 articles
2025-12-22 23:06:42,964 - INFO -   2000s: 58,852 articles
2025-12-22 23:06:42,964 - INFO -
=== Update Summary ===
2025-12-22 23:06:42,964 - INFO - Temporal records in CSV: 1,751,636
2025-12-22 23:06:42,964 - INFO - Articles updated successfully: 1,751,636 (100.0%)
2025-12-22 23:06:42,964 - INFO - Articles not found in database: 0 (0.0%)
2025-12-22 23:06:42,964 - INFO -
Database augmentation complete!
```

## Usage

### Command-Line Options

```
positional arguments:
  input_file              Normalized YAGO CSV file (from normalize_yago_output.py)

options:
  -h, --help              Show help message
  --dry-run               Perform dry run without committing changes
  --batch-size SIZE       Records to update per batch (default: 1000)
  -v, --verbose           Enable verbose logging (debug level)
  --db-host HOST          PostgreSQL host (default: localhost)
  --db-name NAME          Database name (default: wikidb)
  --db-user USER          Database user (default: wiki)
  --db-password PASS      Database password (default: wikipass)
```

### Examples

```bash
# Standard augmentation
python augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv

# Dry run to preview changes without committing
python augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv \
    --dry-run

# Verbose logging to see detailed progress
python augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv \
    --verbose

# Faster processing with larger batch size
python augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv \
    --batch-size 5000

# Custom database connection
python augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv \
    --db-host localhost \
    --db-name wikidb \
    --db-user wiki \
    --db-password wikipass
```

## How It Works

### Date Validation

The script validates all dates before import to ensure data quality:

- **Format**: Must be `YYYY-MM-DD` with 4-digit years (e.g., `0050-01-01`, not `50-01-01`)
- **Range**: Years must be between 0 and current year (2025)
- **Cleaning**: On each run, removes any existing invalid dates from the database

This prevents:
- Ambiguous dates (e.g., `50-01-01` could be year 50 or 1950)
- BCE dates (negative years not supported by current schema)
- Future dates (unrealistic for historical data)
- Malformed data that could cause PostgreSQL errors

### Processing Pipeline

```
Normalized YAGO CSV
(Wikipedia_ID, Earliest_Date, Latest_Date)
         │
         ▼
Load temporal data into memory
Map: Wikipedia_ID → (earliest, latest)
         │
         ▼
Alter articles table
Add columns: has_temporal_info, earliest_date, latest_date
         │
         ▼
Batch update articles
WHERE url LIKE '%curid={Wikipedia_ID}%'
SET has_temporal_info=TRUE, dates
         │
         ▼
Calculate statistics
Total articles, coverage %, date ranges
         │
         ▼
Augmented Database
```

### Matching Strategy

The script matches YAGO data to Wikipedia articles using an indexed Wikipedia page ID column:

1. **Extract IDs**: On first run, extracts Wikipedia page IDs from article URLs into `wikipedia_page_id` column
2. **Create Index**: Creates index on `wikipedia_page_id` for fast lookups
3. **Load CSV**: Read `Wikipedia_ID`, `Earliest_Date`, `Latest_Date` from normalized CSV
4. **Match by ID**: Find articles using indexed `wikipedia_page_id = {Wikipedia_ID}`
5. **Update**: Set temporal columns for matched articles

Example:
```
CSV: Wikipedia_ID = 736, Earliest_Date = 1879-03-14, Latest_Date = 1955-04-18
DB:  wikipedia_page_id = 736 (extracted from url containing 'curid=736')
→    UPDATE articles SET has_temporal_info=TRUE, earliest_date='1879-03-14', latest_date='1955-04-18'
     WHERE wikipedia_page_id = 736
```

**Performance Note**: Using an indexed `wikipedia_page_id` column provides dramatically faster lookups than URL pattern matching.

### Database Schema Changes

The script adds four columns to the `articles` table:

```sql
-- Extract Wikipedia page ID from URL for fast lookups
ALTER TABLE articles ADD COLUMN IF NOT EXISTS wikipedia_page_id INTEGER;
UPDATE articles SET wikipedia_page_id = (regexp_match(url, 'curid=(\\d+)'))[1]::INTEGER 
  WHERE url ~ 'curid=' AND wikipedia_page_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_articles_wikipedia_page_id ON articles(wikipedia_page_id);

-- Add temporal information columns
ALTER TABLE articles ADD COLUMN IF NOT EXISTS has_temporal_info BOOLEAN DEFAULT FALSE;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS earliest_date DATE;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS latest_date DATE;
```

**Note**: The script is safe to run multiple times - it uses `ADD COLUMN IF NOT EXISTS` and updates existing data.

## Statistics and Coverage

### Understanding the Output

**Coverage Percentage**: What proportion of your Wikipedia articles have temporal information from YAGO.

Example interpretation:
```
Total articles: 6,458,670
Articles with temporal info: 1,645,892 (25.48%)
Articles without temporal info: 4,812,778
```

This means:
- ✅ **25.48%** of articles have temporal metadata (people, events, organizations with dates)
- ❌ **74.52%** of articles lack temporal metadata (concepts, places, things without clear time bounds)

**Not all Wikipedia articles have temporal relevance!** Many articles are about:
- Abstract concepts (Democracy, Mathematics, etc.)
- Geographic locations (Mountains, Rivers, etc.)
- Timeless topics (Scientific theories, etc.)

### Articles Not Found in Database

Some YAGO records may not match your database:
```
Articles not found in database: 122,258 (6.9%)
```

Reasons:
1. Article exists in YAGO but not in your Wikipedia dump
2. Wikipedia ID mismatch due to article mergers/deletions
3. YAGO references articles from other language Wikipedias that couldn't be matched

This is normal and expected - the script reports these for transparency.

## Querying Temporal Data

### SQL Examples

After augmentation, you can query temporal information:

```sql
-- Articles with temporal information
SELECT title, earliest_date, latest_date 
FROM articles 
WHERE has_temporal_info = TRUE 
LIMIT 10;

-- Articles active during a specific time period
-- (relevant before July 20, 1969 for Deep Red project)
SELECT title, earliest_date, latest_date
FROM articles
WHERE has_temporal_info = TRUE
  AND earliest_date <= '1969-07-20';

-- Count articles by century
SELECT 
    FLOOR(EXTRACT(YEAR FROM earliest_date) / 100) * 100 as century,
    COUNT(*) as article_count
FROM articles 
WHERE has_temporal_info = TRUE
GROUP BY century
ORDER BY century DESC;

-- Articles from ancient history (BCE dates)
SELECT title, earliest_date, latest_date
FROM articles
WHERE has_temporal_info = TRUE
  AND earliest_date < '0001-01-01'
ORDER BY earliest_date
LIMIT 20;

-- People who died before 1970
SELECT title, earliest_date as birth_date, latest_date as death_date
FROM articles
WHERE has_temporal_info = TRUE
  AND latest_date < '1970-01-01'
  AND earliest_date < latest_date  -- likely a person (has birth & death)
ORDER BY latest_date DESC
LIMIT 100;
```

### Python Examples

```python
import psycopg2

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    database='wikidb',
    user='wiki',
    password='wikipass'
)
cur = conn.cursor()

# Get articles relevant before a cutoff date
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

# Calculate coverage statistics
cur.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN has_temporal_info THEN 1 ELSE 0 END) as with_temporal,
        ROUND(100.0 * SUM(CASE WHEN has_temporal_info THEN 1 ELSE 0 END) / COUNT(*), 2) as coverage_pct
    FROM articles
""")

stats = cur.fetchone()
print(f"Coverage: {stats[1]:,} / {stats[0]:,} articles ({stats[2]}%)")

cur.close()
conn.close()
```

## Troubleshooting

### Common Issues

**1. Database Connection Failed**
```
ERROR - Database connection failed: FATAL: password authentication failed
```
**Solution:**
```bash
# Check PostgreSQL is running
systemctl status postgresql

# Test connection
psql -h localhost -U wiki -d wikidb -c "SELECT 1;"

# Verify credentials in script or use command-line args
python augment_wikipedia_temporal.py input.csv \
    --db-user wiki --db-password wikipass
```

**2. Input File Not Found**
```
ERROR - File not found: yago-facts-normalized.csv
```
**Solution:**
```bash
# Verify file exists
ls -lh ${WIKI_DATA}/yago/yago-facts-normalized.csv

# Use full path
python augment_wikipedia_temporal.py \
    ${WIKI_DATA}/yago/yago-facts-normalized.csv
```

**3. Permission Denied**
```
ERROR - permission denied for table articles
```
**Solution:**
```bash
# Ensure you're running as wiki user
sudo -iu wiki

# Grant permissions if needed (as postgres user)
sudo -u postgres psql wikidb -c "GRANT ALL ON articles TO wiki;"
```

**4. Low Match Rate**
```
Articles updated successfully: 50,000 (3%)
Articles not found in database: 1,700,000 (97%)
```
**Solution:**
- Verify you're using the **normalized** CSV file (not the raw yago-facts.csv)
- The normalized file should have `Wikipedia_ID` column with numeric IDs
- Check that article URLs in database contain `curid=` format

## Integration with Deep Red Project

### Temporal Knowledge Cutoff

For the Deep Red project (1969-07-20 cutoff), you can now:

1. **Export articles with pre-1969 temporal data:**
```bash
psql -h localhost -U wiki -d wikidb -c "
COPY (
    SELECT title, content, earliest_date, latest_date
    FROM articles
    WHERE has_temporal_info = TRUE
      AND earliest_date <= '1969-07-20'
) TO '/tmp/pre1969_articles.csv' CSV HEADER;
"
```

2. **Count relevant articles:**
```sql
SELECT COUNT(*) 
FROM articles 
WHERE has_temporal_info = TRUE 
  AND earliest_date <= '1969-07-20';
```

3. **Analyze temporal distribution:**
```sql
SELECT 
    EXTRACT(YEAR FROM earliest_date) as year,
    COUNT(*) as article_count
FROM articles
WHERE has_temporal_info = TRUE
  AND earliest_date <= '1969-07-20'
GROUP BY year
ORDER BY year DESC
LIMIT 50;
```

### Next Steps

With temporal augmentation complete, you can proceed to:
- **Fine-tuning preparation**: Extract articles relevant before your temporal cutoff
- **Training dataset creation**: Build temporally-filtered corpus for model training
- **Temporal validation**: Verify model doesn't reference post-cutoff events

See [Temporal-Finetuning-Plan.md](Temporal-Finetuning-Plan.md) for the complete fine-tuning strategy.

## Performance Notes

### Processing Time

Typical performance on a modern system:
- **Loading CSV**: ~5 seconds for 1.8M records
- **Batch updates**: ~50,000 records/minute
- **Total time**: ~30-40 minutes for 1.8M temporal records

### Optimization

**Increase batch size** for faster processing:
```bash
# Default: 1,000 records per batch
python augment_wikipedia_temporal.py input.csv

# Faster: 5,000 records per batch
python augment_wikipedia_temporal.py input.csv --batch-size 5000
```

**Trade-offs:**
- Larger batches = faster processing but higher memory usage
- Smaller batches = slower but safer for memory-constrained systems

### Safe to Re-run

The script is **idempotent** - safe to run multiple times:
- Uses `ADD COLUMN IF NOT EXISTS` (won't fail if columns exist)
- Updates overwrite previous values
- Can be used to refresh temporal data after updating YAGO

## Database Backup

Before running the script, consider backing up your database:

```bash
# Backup articles table
pg_dump -h localhost -U wiki -d wikidb -t articles > articles_backup.sql

# Or backup entire database
pg_dump -h localhost -U wiki -d wikidb > wikidb_backup.sql

# Restore if needed
psql -h localhost -U wiki -d wikidb < articles_backup.sql
```

## References

- Local
  - [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) - Wikipedia database setup
  - [YagoParser-Setup.md](YagoParser-Setup.md) - YAGO parsing documentation
  - [YagoNormalizer-Setup.md](YagoNormalizer-Setup.md) - YAGO normalization documentation
  - [Temporal-Finetuning-Plan.md](Temporal-Finetuning-Plan.md) - Fine-tuning strategy
- Internet
  - [YAGO Knowledge Base](https://yago-knowledge.org/)
  - [PostgreSQL Date/Time Functions](https://www.postgresql.org/docs/current/functions-datetime.html)
