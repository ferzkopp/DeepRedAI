# Wikipedia Year-Topics

Wikipedia has *Year Pages* at specific URLs: https://en.wikipedia.org/wiki/1969

These contain important topics that should be added to the Temporal Finetuning dataset.

## Overview

The `extract_year_topics.py` script extracts historical events from Wikipedia year pages (e.g., https://en.wikipedia.org/wiki/2020) and enriches them with article references for use in the Temporal Finetuning dataset.

**Key Features:**
- Fetches year pages via the Wikipedia API and parses the HTML structure
- Extracts dated events from the "Events" section with full date parsing (year, month, day)
- Captures direct Wikipedia article links embedded in event text as primary references
- Searches the local Wikipedia MCP server to find additional related articles using hybrid search
- Calculates relevance scores combining title similarity, search ranking, and position
- Deduplicates all article references and looks up article IDs via the MCP server
- Stores enriched event data as JSON files in `${WIKI_DATA}/topics/` for downstream tools

## Solution

### Architecture

The solution consists of a Python script `extract_year_topics.py` that:

1. **Year Page Retrieval**
   - Fetches Wikipedia year pages (e.g., "1990", "1991", ..., "2025") from the Wikipedia API
   - Uses the Wikipedia API's `action=parse` endpoint to get clean HTML content
   - Falls back to the local PostgreSQL database if API fails

2. **Topic Extraction**
   - Parses the HTML structure to identify event sections (Events, Births, Deaths, etc.)
   - Extracts date-topic pairs from list items in the Events section
   - Uses regex patterns to parse dates in various formats:
     - "January 1 – Event description"
     - "January 1-3 – Event description"
     - "January – Event description"
   - Captures both the specific date and the event description

3. **Direct Reference Extraction**
   - Extracts Wikipedia article links directly from event text (e.g., links to people, places, events)
   - Looks up article IDs via keyword search on the MCP server
   - Assigns maximum relevance score (1.0) to direct links
   - Deduplicates links by title (case-insensitive)

4. **Related Article Search & Relevance Scoring**
   - For each topic, uses the Wikipedia MCP server to perform hybrid search
   - Queries both keyword (BM25) and semantic (k-NN) search
   - Calculates a relevance score based on:
     - Search result ranking (position in results)
     - Search score from MCP server
     - Title similarity to topic text
     - Combined weighted score: 40% title similarity + 40% search score + 20% position
   - Filters out articles already in direct references (by title and article_id)
   - Deduplicates results and returns top N articles (default: 5) sorted by relevance

4. **Data Storage**
   - Creates output directory: `${WIKI_DATA}/topics/`
   - Stores results in JSON format: `${WIKI_DATA}/topics/year_topics_YYYY.json`
   - Each file contains:
     - Year metadata (year, extracted_date, source, total_topics)
     - Array of topics with:
       - Date fields (date, year, month, day, date_text)
       - Topic description
       - Array of `direct_references` (links found in event text):
         - Article title
         - Article path and href
         - Article ID (looked up via MCP server)
         - Relevance score (always 1.0 for direct links)
         - Source: "direct_link"
       - Array of `related_articles` (found via search):
         - Article title
         - Article ID
         - Relevance score (0-1)
         - Search score
         - Title similarity score

### Data Format Example

```json
{
  "year": 2020,
  "extracted_date": "2025-12-25T12:30:00Z",
  "source": "wikipedia_api",
  "total_topics": 245,
  "topics": [
    {
      "year": 2020,
      "date": "2020-01-01",
      "month": 1,
      "day": 1,
      "date_text": "January 1",
      "topic": "Flash floods struck Jakarta, Indonesia, killing 66 people in the worst flooding in over a decade.",
      "direct_references": [
        {
          "title": "2020 Jakarta floods",
          "article_path": "2020_Jakarta_floods",
          "href": "/wiki/2020_Jakarta_floods",
          "source": "direct_link",
          "relevance_score": 1.0,
          "article_id": 62847291
        },
        {
          "title": "Jakarta",
          "article_path": "Jakarta",
          "href": "/wiki/Jakarta",
          "source": "direct_link",
          "relevance_score": 1.0,
          "article_id": 16254
        }
      ],
      "related_articles": [
        {
          "title": "Floods in Jakarta",
          "article_id": 38269340,
          "relevance_score": 0.3,
          "search_score": 0.03,
          "title_similarity": 0.298
        }
      ]
    }
  ]
}
```

### Usage

```bash
# Set WIKI_DATA environment variable (if not already set)
export WIKI_DATA="/mnt/data/wikipedia"

# Extract topics for a specific year
python scripts/extract_year_topics.py --year 1990

# Extract topics for a range of years
python scripts/extract_year_topics.py --start-year 1900 --end-year 2025

# Adjust number of related articles per topic (default: 5)
python scripts/extract_year_topics.py --year 2020 --max-articles 10

# Use verbose output for debugging
python scripts/extract_year_topics.py --year 2020 --verbose

# Save raw HTML for debugging analysis
python scripts/extract_year_topics.py --year 2020 --save-html
```

### Dependencies

#### Required Packages

- `requests` - HTTP client for Wikipedia API
- `beautifulsoup4` - HTML parsing
- `psycopg2-binary` - PostgreSQL access (fallback)
- `rapidfuzz` - String similarity scoring (faster alternative to fuzzywuzzy)
- Access to local Wikipedia MCP server (running at http://localhost:3000)

#### Installation Steps

1. **Set environment configuration variables:**

```bash
# Storage path - adjust to your disk mount point
export WIKI_DATA="/mnt/data/wikipedia"

# Optional: Set custom MCP server URL if not using default
export MCP_SERVER_URL="http://localhost:7000"
```

Switch to wiki user and download the dump:

```bash
sudo -iu wiki
```

2. **Install Python dependencies:**

Install packages within the Wikipedia virtual environment:

```bash
# Activate the Wikipedia virtual environment
source ${WIKI_DATA}/venv/bin/activate

# Install required packages
pip install requests beautifulsoup4 psycopg2-binary rapidfuzz
```

3. **Verify MCP server is running:**

```bash
# Check if MCP server is accessible
curl http://localhost:7000/health
```

See [Wikipedia MCP](documentation/WikipediaMCP-Setup.md) for more information.

4. **Verify installation:**

```bash
# Test with a single year
python scripts/extract_year_topics.py --year 2020 --verbose

# Check the output
ls -lh ${WIKI_DATA}/topics/
cat ${WIKI_DATA}/topics/year_topics_2020.json | jq '.topics[0]'
```
