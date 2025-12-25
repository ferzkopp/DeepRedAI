#!/usr/bin/env python3
"""
Extract Year Topics from Wikipedia

Fetches Wikipedia year pages (e.g., 1990, 1991, ..., 2025), extracts dated events
and topics, searches for related articles using the Wikipedia MCP server, calculates
relevance scores, and stores the results in JSON format.

Usage:
    python extract_year_topics.py --year 1990
    python extract_year_topics.py --start-year 1990 --end-year 2025
    python extract_year_topics.py --year 2020 --max-articles 10 --verbose
"""

import os
import sys
import json
import re
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from rapidfuzz import fuzz
except ImportError:
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("Warning: Neither rapidfuzz nor fuzzywuzzy found. Installing rapidfuzz...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz"])
        from rapidfuzz import fuzz

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WIKI_DATA = os.environ.get('WIKI_DATA', '/mnt/data/wikipedia')
MCP_SERVER_URL = os.environ.get('MCP_SERVER_URL', 'http://localhost:3000')
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Search parameters
DEFAULT_MAX_ARTICLES = 5
MAX_SEARCH_RESULTS = 10

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Wikipedia API Functions
# -----------------------------------------------------------------------------

def fetch_year_page_html(year: int) -> Optional[str]:
    """
    Fetch the HTML content of a Wikipedia year page using the API.
    
    Args:
        year: The year (e.g., 1990)
        
    Returns:
        HTML content as string, or None if fetch fails
    """
    params = {
        'action': 'parse',
        'page': str(year),
        'prop': 'text',
        'format': 'json',
        'disableeditsection': '1',
        'disabletoc': '1'
    }
    
    try:
        logger.info(f"Fetching year page {year} from Wikipedia API...")
        response = requests.get(WIKIPEDIA_API_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'error' in data:
            logger.error(f"Wikipedia API error: {data['error'].get('info', 'Unknown error')}")
            return None
            
        if 'parse' in data and 'text' in data['parse']:
            html_content = data['parse']['text']['*']
            logger.info(f"Successfully fetched {len(html_content)} bytes of HTML")
            return html_content
        else:
            logger.error("Unexpected API response structure")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch year page {year}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse API response: {e}")
        return None


# -----------------------------------------------------------------------------
# HTML Parsing Functions
# -----------------------------------------------------------------------------

def parse_date_from_text(date_text: str, year: int) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse date information from event text.
    
    Args:
        date_text: Text containing the date (e.g., "January 1", "March 15-20")
        year: The year for this event
        
    Returns:
        Tuple of (full_date, month, day) where:
        - full_date: ISO format date string (YYYY-MM-DD) or None
        - month: Month number (1-12) or None
        - day: Day number (1-31) or None
    """
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Try to match "Month Day" pattern (e.g., "January 1")
    match = re.match(r'([A-Za-z]+)\s+(\d+)', date_text, re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        day = int(match.group(2))
        
        if month_name in months:
            month = months[month_name]
            full_date = f"{year:04d}-{month:02d}-{day:02d}"
            return full_date, month, day
    
    # Try to match "Month" only pattern (e.g., "January")
    match = re.match(r'^([A-Za-z]+)$', date_text.strip(), re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        if month_name in months:
            month = months[month_name]
            return None, month, None
    
    return None, None, None


def extract_topics_from_html(html: str, year: int) -> List[Dict[str, Any]]:
    """
    Extract topics and dates from Wikipedia year page HTML.
    
    Args:
        html: HTML content of the year page
        year: The year being processed
        
    Returns:
        List of topic dictionaries with date and topic information
    """
    soup = BeautifulSoup(html, 'html.parser')
    topics = []
    
    # Find the Events section
    events_heading = None
    for heading in soup.find_all(['h2', 'h3']):
        heading_text = heading.get_text().strip().lower()
        if 'event' in heading_text:
            events_heading = heading
            logger.info(f"Found Events section: {heading.get_text().strip()}")
            break
    
    if not events_heading:
        logger.warning(f"No Events section found for year {year}")
        return topics
    
    # Find all list items after the Events heading until the next major heading
    current = events_heading.find_next_sibling()
    while current:
        # Stop at next h2 heading
        if current.name == 'h2':
            break
            
        # Process unordered lists
        if current.name in ['ul', 'ol']:
            for li in current.find_all('li', recursive=False):
                text = li.get_text().strip()
                
                # Skip empty items
                if not text:
                    continue
                
                # Try to parse date and topic
                # Common formats:
                # "January 1 – Event description"
                # "January 1-3 – Event description"
                # "January – Event description"
                match = re.match(r'^([^–—-]+?)\s*[–—-]\s*(.+)$', text)
                
                if match:
                    date_part = match.group(1).strip()
                    topic_part = match.group(2).strip()
                    
                    # Parse the date
                    full_date, month, day = parse_date_from_text(date_part, year)
                    
                    topic_dict = {
                        'date': full_date,
                        'month': month,
                        'day': day,
                        'date_text': date_part,
                        'topic': topic_part
                    }
                    
                    topics.append(topic_dict)
                    logger.debug(f"Extracted topic: {date_part} - {topic_part[:50]}...")
                else:
                    # No date separator found, treat entire text as topic
                    topic_dict = {
                        'date': None,
                        'month': None,
                        'day': None,
                        'date_text': '',
                        'topic': text
                    }
                    topics.append(topic_dict)
                    logger.debug(f"Extracted topic (no date): {text[:50]}...")
        
        current = current.find_next_sibling()
    
    logger.info(f"Extracted {len(topics)} topics from year {year}")
    return topics


# -----------------------------------------------------------------------------
# MCP Server Search Functions
# -----------------------------------------------------------------------------

def search_mcp_server(query: str, limit: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
    """
    Search the Wikipedia MCP server for articles related to a query.
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of search result dictionaries with title, id, score, and text
    """
    url = urljoin(MCP_SERVER_URL, '/mcp/search')
    
    payload = {
        'query': query,
        'mode': 'hybrid',  # Use hybrid search (keyword + semantic)
        'limit': limit
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        logger.debug(f"Found {len(results)} results for query: {query[:50]}...")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP search failed for query '{query}': {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse MCP response: {e}")
        return []


def calculate_relevance_score(
    topic: str,
    article_title: str,
    search_score: float,
    position: int,
    total_results: int
) -> Dict[str, float]:
    """
    Calculate a relevance score for an article based on multiple factors.
    
    Args:
        topic: The original topic text
        article_title: The article title
        search_score: The search score from MCP server
        position: Position in search results (0-indexed)
        total_results: Total number of search results
        
    Returns:
        Dictionary with relevance metrics
    """
    # Title similarity (0-1)
    title_similarity = fuzz.token_sort_ratio(topic.lower(), article_title.lower()) / 100.0
    
    # Position penalty (higher position = lower score)
    position_penalty = 1.0 - (position / max(total_results, 1))
    
    # Normalize search score (assume max score is ~20-30 for typical searches)
    # This is a heuristic and may need adjustment
    normalized_search_score = min(search_score / 20.0, 1.0)
    
    # Combined relevance score
    # Weighted: 40% title similarity, 40% search score, 20% position
    relevance_score = (
        0.4 * title_similarity +
        0.4 * normalized_search_score +
        0.2 * position_penalty
    )
    
    return {
        'relevance_score': round(relevance_score, 3),
        'title_similarity': round(title_similarity, 3),
        'search_score': round(search_score, 2),
        'position_factor': round(position_penalty, 3)
    }


def find_related_articles(topic: str, max_articles: int = DEFAULT_MAX_ARTICLES) -> List[Dict[str, Any]]:
    """
    Find Wikipedia articles related to a topic.
    
    Args:
        topic: The topic text to search for
        max_articles: Maximum number of articles to return
        
    Returns:
        List of related article dictionaries sorted by relevance
    """
    # Search the MCP server
    search_results = search_mcp_server(topic, limit=MAX_SEARCH_RESULTS)
    
    if not search_results:
        logger.warning(f"No search results found for topic: {topic[:50]}...")
        return []
    
    # Calculate relevance for each result
    related_articles = []
    for idx, result in enumerate(search_results):
        title = result.get('title', '')
        article_id = result.get('article_id') or result.get('id')
        search_score = result.get('score', 0.0)
        
        # Calculate relevance metrics
        metrics = calculate_relevance_score(
            topic, title, search_score, idx, len(search_results)
        )
        
        article_info = {
            'title': title,
            'article_id': article_id,
            'relevance_score': metrics['relevance_score'],
            'search_score': metrics['search_score'],
            'title_similarity': metrics['title_similarity']
        }
        
        related_articles.append(article_info)
    
    # Sort by relevance score (descending)
    related_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Return top N articles
    return related_articles[:max_articles]


# -----------------------------------------------------------------------------
# Main Processing Functions
# -----------------------------------------------------------------------------

def process_year(year: int, max_articles: int = DEFAULT_MAX_ARTICLES) -> Dict[str, Any]:
    """
    Process a single year: fetch page, extract topics, find related articles.
    
    Args:
        year: The year to process
        max_articles: Maximum number of related articles per topic
        
    Returns:
        Dictionary with year data and topics
    """
    logger.info(f"Processing year {year}...")
    
    # Fetch the year page HTML
    html = fetch_year_page_html(year)
    if not html:
        logger.error(f"Failed to fetch year page for {year}")
        return None
    
    # Extract topics from HTML
    topics = extract_topics_from_html(html, year)
    if not topics:
        logger.warning(f"No topics extracted for year {year}")
        return None
    
    # Find related articles for each topic
    processed_topics = []
    for idx, topic in enumerate(topics, 1):
        topic_text = topic['topic']
        logger.info(f"Processing topic {idx}/{len(topics)}: {topic_text[:60]}...")
        
        # Search for related articles
        related_articles = find_related_articles(topic_text, max_articles)
        
        # Add related articles to topic
        topic['related_articles'] = related_articles
        processed_topics.append(topic)
    
    # Create output structure
    year_data = {
        'year': year,
        'extracted_date': datetime.utcnow().isoformat() + 'Z',
        'source': 'wikipedia_api',
        'total_topics': len(processed_topics),
        'topics': processed_topics
    }
    
    return year_data


def save_year_data(year_data: Dict[str, Any], output_dir: str) -> str:
    """
    Save year data to JSON file.
    
    Args:
        year_data: The year data dictionary
        output_dir: Output directory path
        
    Returns:
        Path to the saved file
    """
    year = year_data['year']
    filename = f"year_topics_{year}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(year_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved year data to {filepath}")
    return filepath


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract year topics from Wikipedia and find related articles'
    )
    
    # Year arguments (mutually exclusive: either --year or --start-year/--end-year)
    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument(
        '--year',
        type=int,
        help='Single year to process (e.g., 1990)'
    )
    year_group.add_argument(
        '--start-year',
        type=int,
        help='Start year for range processing'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        help='End year for range processing (required with --start-year)'
    )
    
    parser.add_argument(
        '--max-articles',
        type=int,
        default=DEFAULT_MAX_ARTICLES,
        help=f'Maximum number of related articles per topic (default: {DEFAULT_MAX_ARTICLES})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory (default: $WIKI_DATA/topics/)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine output directory
    output_dir = args.output_dir or os.path.join(WIKI_DATA, 'topics')
    logger.info(f"Output directory: {output_dir}")
    
    # Determine years to process
    if args.year:
        years = [args.year]
    else:
        if not args.end_year:
            parser.error("--end-year is required when using --start-year")
        years = range(args.start_year, args.end_year + 1)
    
    logger.info(f"Processing {len(years)} year(s)...")
    
    # Process each year
    success_count = 0
    for year in years:
        try:
            year_data = process_year(year, args.max_articles)
            
            if year_data:
                save_year_data(year_data, output_dir)
                success_count += 1
            else:
                logger.error(f"Failed to process year {year}")
                
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}", exc_info=True)
    
    # Summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Successfully processed: {success_count}/{len(years)} years")
    logger.info(f"Output location: {output_dir}")
    
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
