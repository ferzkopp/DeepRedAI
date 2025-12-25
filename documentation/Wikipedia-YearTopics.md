Wikipedia has Year Pages at specific URLs:
https://en.wikipedia.org/wiki/1969

These contain important topics that should be added to the Temporal Finetuning dataset.

## Problem Statement

The goal is to:
- find these pages in the Wikipedia database for years 1990 to the current year
- understand the article structure and extract the topic lines as well as the date (if available)
  - if the article text format in the database does not have sufficient structure anymore, simply download the actual article (scrape) via the Wikipedia API and parse that
- use the local Wikipedia MCP server to search for (both direct and heuristic) and locate the article/articles relevant to the topic line
  - since there may be direct hits and tangential but possibly relevant articles, maybe calculate a relevance index
- store the article titles (and if available IDs) with the dates and other metadata in a seperate file under ${WIKI_DATA}/topics/ for other tools to use

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

3. **Article Search & Relevance Scoring**
   - For each topic, uses the Wikipedia MCP server to perform hybrid search
   - Queries both keyword (BM25) and semantic (k-NN) search
   - Calculates a relevance score based on:
     - Search result ranking (position in results)
     - Search score from MCP server
     - Title similarity to topic text
     - Combined weighted score: `(1 - position_penalty) * search_score * title_similarity`
   - Returns top N articles (default: 5) sorted by relevance

4. **Data Storage**
   - Creates output directory: `${WIKI_DATA}/topics/`
   - Stores results in JSON format: `${WIKI_DATA}/topics/year_topics_YYYY.json`
   - Each file contains:
     - Year metadata
     - Array of topics with:
       - Date (full date, month, or year-only)
       - Topic description
       - Array of related articles with:
         - Article title
         - Article ID (from Wikipedia)
         - Relevance score (0-1)
         - Search score
         - Title similarity score

### Data Format Example

```json
{
  "year": 1990,
  "extracted_date": "2025-12-24T12:30:00Z",
  "source": "wikipedia_api",
  "topics": [
    {
      "date": "1990-01-01",
      "month": 1,
      "day": 1,
      "topic": "The Leaning Tower of Pisa is closed to the public",
      "related_articles": [
        {
          "title": "Leaning Tower of Pisa",
          "article_id": 123456,
          "relevance_score": 0.95,
          "search_score": 15.3,
          "title_similarity": 0.87
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
python scripts/extract_year_topics.py --start-year 1990 --end-year 2025

# Adjust number of articles per topic (default: 5)
python scripts/extract_year_topics.py --year 2020 --max-articles 10

# Use verbose output for debugging
python scripts/extract_year_topics.py --year 2020 --verbose
```

### Dependencies

#### Required Packages

- `requests` - HTTP client for Wikipedia API
- `beautifulsoup4` - HTML parsing
- `psycopg2-binary` - PostgreSQL access (fallback)
- `rapidfuzz` - String similarity scoring (faster alternative to fuzzywuzzy)
- Access to local Wikipedia MCP server (running at http://localhost:3000)

#### Installation Steps

1. **Install Python dependencies:**

```bash
pip install requests beautifulsoup4 psycopg2-binary rapidfuzz
```

Or install within the Wikipedia virtual environment:

```bash
# Activate the Wikipedia virtual environment
source ${WIKI_DATA}/venv/bin/activate

# Install required packages
pip install requests beautifulsoup4 psycopg2-binary rapidfuzz
```

2. **Verify MCP server is running:**

```bash
# Check if MCP server is accessible
curl http://localhost:3000/health

# If not running, start the MCP server
sudo systemctl start mcp.service

# Or run manually (in the Wikipedia virtual environment)
source ${WIKI_DATA}/venv/bin/activate
python scripts/mcp_server.py
```

3. **Set environment variables:**

```bash
# Set WIKI_DATA path (if not already configured)
export WIKI_DATA="/mnt/data/wikipedia"

# Optional: Set custom MCP server URL if not using default
export MCP_SERVER_URL="http://localhost:3000"
```

4. **Verify installation:**

```bash
# Test with a single year
python scripts/extract_year_topics.py --year 2020 --verbose

# Check the output
ls -lh ${WIKI_DATA}/topics/
cat ${WIKI_DATA}/topics/year_topics_2020.json | jq '.topics[0]'
```
