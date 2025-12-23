# YAGO Output Normalizer Setup

## Overview

The `normalize_yago_output.py` script normalizes Wikipedia URLs from YAGO parser output to English Wikipedia, adding page IDs from your local database. This is essential for integrating YAGO temporal metadata with your English Wikipedia database.

## Problem

YAGO contains Wikipedia links in many languages, but your local database has only English Wikipedia:

- **Non-English URLs**: URLs point to `ar.wikipedia.org`, `pnb.wikipedia.org`, `ca.wikipedia.org`, etc.
- **Missing Wikipedia IDs**: No page IDs to match articles in your database

## Solution

The script:
1. Detects non-English Wikipedia URLs and extracts language code + title
2. Queries Wikipedia API to find English equivalent articles
3. Validates articles exist in local PostgreSQL database
4. Extracts Wikipedia page IDs from database URLs
5. Outputs normalized data with English URLs and page IDs

## Features

✅ Converts non-English URLs to English equivalents  
✅ Adds Wikipedia page IDs from local database  
✅ Validates articles exist in database  
✅ Handles redirects automatically  
✅ CSV and JSON format support  
✅ API caching to minimize calls  
✅ Rate limiting (10 req/sec default)  
✅ Resume capability for large datasets  
✅ Progress tracking  
✅ Preserves original URLs for reference

## Prerequisites

```bash
# Storage path
export WIKI_DATA="/mnt/data/wikipedia"

# Switch to wiki user
sudo -iu wiki

# Install dependencies
source ${WIKI_DATA}/venv/bin/activate
pip install psycopg2-binary requests
```

## Quick Start

### Complete Pipeline (Parse + Normalize)

```bash
# Step 1: Parse YAGO data
python yago_parser.py ${WIKI_DATA}/yago/yago-facts.ttl \
    --csv ${WIKI_DATA}/yago/yago-facts.csv --verbose

# Step 2: Normalize to English Wikipedia with page IDs
python normalize_yago_output.py \
    ${WIKI_DATA}/yago/yago-facts.csv \
    --output ${WIKI_DATA}/yago/yago-facts-normalized.csv \
    --verbose
```

expected output

```
...
Normalization complete!
2025-12-22 21:47:47,479 - INFO -   Total entries: 1,811,435
2025-12-22 21:47:47,479 - INFO -   Normalized: 1,768,150 (97.6%)
2025-12-22 21:47:47,479 - INFO -   Skipped/kept original: 43,285 (2.4%)
2025-12-22 21:47:47,479 - INFO -   API calls made: 28,025
2025-12-22 21:47:47,479 - INFO -   API translations successful: 21,414
2025-12-22 21:47:47,479 - INFO -   API translations not found: 6,611
2025-12-22 21:47:47,479 - INFO -   Output saved to: /mnt/data/wikipedia/yago/yago-facts-normalized.csv
```

## Usage

### Basic Commands

```bash
# Normalize CSV file
python normalize_yago_output.py input.csv --output normalized.csv

# Skip entries not in database (clean dataset)
python normalize_yago_output.py input.csv --output normalized.csv --skip-missing

# Verbose logging (see API calls and DB queries)
python normalize_yago_output.py input.csv --output normalized.csv --verbose

# Process JSON format
python normalize_yago_output.py input.json --output normalized.json --format json

# Resume from interruption (large datasets)
python normalize_yago_output.py input.csv --output normalized.csv --resume

# Increase API delay if throttled (default 0.1s)
python normalize_yago_output.py input.csv --output normalized.csv --api-delay 0.5
```

### Command-Line Options

```
positional arguments:
  input_file            Input file from yago_parser.py (CSV or JSON)

options:
  -h, --help            Show help message
  -o, --output OUTPUT   Output file path (required)
  -f, --format {csv,json}
                        Output format (auto-detected from extension)
  --skip-missing        Skip entries not found in database
  -r, --resume          Resume from existing output (skip processed entries)
  -v, --verbose         Enable verbose logging
  --api-delay SECONDS   Delay between API calls (default: 0.1)
  --db-host HOST        PostgreSQL host (default: localhost)
  --db-name NAME        Database name (default: wikidb)
  --db-user USER        Database user (default: wiki)
  --db-password PASS    Database password (default: wikipass)
```

## Input/Output Formats

### Input from YAGO Parser

**CSV format** (yago-facts.csv):
```csv
Entity,Wikipedia_URL,Earliest_Date,Latest_Date
Albert_Einstein,https://en.wikipedia.org/wiki/Albert_Einstein,1879-03-14,1955-04-18
Marie_Curie,https://fr.wikipedia.org/wiki/Marie_Curie,1867-11-07,1934-07-04
Abd_al-Hamīd_ibn_Turk,https://ar.wikipedia.org/wiki/%D8%B9%D8%A8%D8%AF,800-01-01,900-01-01
```

**JSON format** (yago-facts.json):
```json
[
  {
    "entity": "Abd_al-Hamīd_ibn_Turk",
    "wikipedia_url": "https://ar.wikipedia.org/wiki/%D8%B9%D8%A8%D8%AF",
    "earliest_date": "800-01-01",
    "latest_date": "900-01-01"
  }
]
```

### Output with English URLs + Page IDs

**CSV format** (yago_normalized.csv):
```csv
Entity,Wikipedia_Title,Wikipedia_ID,Wikipedia_URL,Earliest_Date,Latest_Date,Original_URL
Albert_Einstein,Albert_Einstein,736,https://en.wikipedia.org/wiki?curid=736,1879-03-14,1955-04-18,
Marie_Curie,Marie_Curie,20017,https://en.wikipedia.org/wiki?curid=20017,1867-11-07,1934-07-04,https://fr.wikipedia.org/wiki/Marie_Curie
Abd_al-Hamīd_ibn_Turk,Abd_al-Hamid_ibn_Turk,42857,https://en.wikipedia.org/wiki?curid=42857,800-01-01,900-01-01,https://ar.wikipedia.org/wiki/%D8%B9%D8%A8%D8%AF
```

**Output Fields:**
- **Entity**: Original YAGO entity name
- **Wikipedia_Title**: English Wikipedia article title
- **Wikipedia_ID**: Page ID from local database
- **Wikipedia_URL**: English URL format `https://en.wikipedia.org/wiki?curid={page_id}`
- **Earliest_Date**: Earliest date from YAGO
- **Latest_Date**: Latest date from YAGO
- **Original_URL**: Original non-English URL (empty if already English)

## How It Works

### Processing Pipeline

```
YAGO CSV/JSON (mixed langs)
         │
         ▼
Parse URL → Extract lang + title
         │
         ▼
    English? ─No→ Wikipedia API → Get EN title
         │                              │
        Yes                             │
         └──────────────┬───────────────┘
                        ▼
            Query Local Database
         Get page_id from URL field
                        │
                        ▼
               Normalized Output
          (English URLs + page IDs)
```

### URL Parsing

Extracts language code and title from Wikipedia URLs:
```
https://ar.wikipedia.org/wiki/عبد_الحميد_بن_ترك
         ↓
lang: ar, title: عبد_الحميد_بن_ترك
```

### Wikipedia API Lookup

For non-English articles, queries API for language links:
```bash
# API Request
https://ar.wikipedia.org/w/api.php?action=query&titles=عبد_الحميد&prop=langlinks&lllang=en

# API Response
{"query": {"pages": [{"langlinks": [{"lang": "en", "title": "Abd_al-Hamid_ibn_Turk"}]}]}}
```

### Database Lookup

Queries local PostgreSQL for article and extracts page ID:
```sql
-- Try exact match
SELECT id, url FROM articles WHERE title = 'Abd_al-Hamid_ibn_Turk';

-- Try with spaces
SELECT id, url FROM articles WHERE title = 'Abd al-Hamid ibn Turk';

-- Check redirects
SELECT target_title FROM redirects WHERE source_title = 'Abd_al-Hamid_ibn_Turk';
```

Database URL format: `https://en.wikipedia.org/wiki?curid=12345`  
Script extracts: `12345` as Wikipedia page ID

## Error Handling

### Common Issues

**1. API Throttling (403/429 errors)**
```
ERROR - API THROTTLING ERROR (403 Forbidden)
ERROR - Rate limiting detected. Use --resume with increased --api-delay
```
**Solution:**
```bash
# Resume with increased delay
python normalize_yago_output.py input.csv --output normalized.csv --resume --api-delay 0.5
```

**2. Database Connection Failed**
```
ERROR - Database connection failed: FATAL: password authentication failed
```
**Solution:**
```bash
# Check PostgreSQL status
systemctl status postgresql

# Test connection
psql -h localhost -U wiki -d wikidb -c "SELECT 1;"
```

**3. Article Not Found in Database**
```
WARNING - Article not found in database: Some_Article
```
**Solution:** This is normal. Use `--skip-missing` to exclude these entries, or keep with original URLs (default).

**4. No English Equivalent**
```
WARNING - No English equivalent found for lang:Article
```
**Solution:** Some articles only exist in non-English Wikipedia. Use `--skip-missing` to exclude these.

## Python Integration Examples

### Example 1: Filter Articles by Date Range

```python
import csv
from datetime import datetime

def articles_in_date_range(csv_file, start_date, end_date):
    """Find articles with events in a specific date range"""
    results = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            earliest = datetime.strptime(row['Earliest_Date'], '%Y-%m-%d')
            latest = datetime.strptime(row['Latest_Date'], '%Y-%m-%d')
            
            # Check if date range overlaps with target range
            if earliest <= end_date and latest >= start_date:
                results.append(row)
    
    return results

# Find articles with events before 1970
articles = articles_in_date_range(
    'yago_normalized.csv',
    datetime(1, 1, 1),
    datetime(1969, 12, 31)
)
```

### Example 2: Augment Wikipedia Database

```python
import csv
import psycopg2

def augment_articles_with_dates(csv_file, db_config):
    """Add temporal metadata to Wikipedia articles"""
    
    # Load date information
    date_map = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wiki_id = int(row['Wikipedia_ID'])
            date_map[wiki_id] = {
                'earliest': row['Earliest_Date'],
                'latest': row['Latest_Date']
            }
    
    # Update database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    # Add columns if they don't exist
    cur.execute("""
        ALTER TABLE articles 
        ADD COLUMN IF NOT EXISTS earliest_date DATE,
        ADD COLUMN IF NOT EXISTS latest_date DATE
    """)
    
    # Update articles with date information
    for wiki_id, dates in date_map.items():
        cur.execute("""
            UPDATE articles 
            SET earliest_date = %s, latest_date = %s
            WHERE url LIKE %s
        """, (dates['earliest'], dates['latest'], f'%curid={wiki_id}%'))
    
    conn.commit()
    cur.close()
    conn.close()
```

### Example 3: Create Temporal Cutoff Dataset

```python
import csv
import json
from datetime import datetime

def create_cutoff_dataset(csv_file, cutoff_date, output_file):
    """Create dataset of articles before cutoff date (for Deep Red project)"""
    
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    articles = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            earliest = datetime.strptime(row['Earliest_Date'], '%Y-%m-%d')
            
            # Include if article has events before cutoff
            if earliest <= cutoff:
                articles.append({
                    'title': row['Wikipedia_Title'],
                    'page_id': int(row['Wikipedia_ID']),
                    'url': row['Wikipedia_URL'],
                    'earliest_date': row['Earliest_Date'],
                    'latest_date': row['Latest_Date']
                })
    
    with open(output_file, 'w') as f:
        json.dump(articles, f, indent=2)
    
    return articles

# Create dataset for 1969-07-31 cutoff
articles = create_cutoff_dataset(
    'yago_normalized.csv',
    '1969-07-31',
    'temporal_cutoff_1969.json'
)
```

## Database Configuration

The script connects to your local PostgreSQL database with these defaults:

```python
DB_CONFIG = {
    'host': 'localhost',
    'database': 'wikidb',
    'user': 'wiki',
    'password': 'wikipass'
}
```

Override with command-line arguments:
```bash
--db-host localhost --db-name wikidb --db-user wiki --db-password wikipass
```

## References

- Local
  - [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) - Wikipedia database setup
  - [YagoParser-Setup.md](YagoParser-Setup.md) - YAGO parser documentation
- Internet
  - [YAGO Knowledge Base](https://yago-knowledge.org/)
  - [Wikipedia API Documentation](https://www.mediawiki.org/wiki/API:Main_page)
