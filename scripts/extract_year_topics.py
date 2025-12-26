#!/usr/bin/env python3
"""
Extract Year Topics from Wikipedia

Fetches Wikipedia year pages (e.g., 1990, 1991, ..., 2025), extracts dated events
and topics, searches for related articles using the Wikipedia MCP server, calculates
relevance scores, and stores the results in JSON format.

Usage:
    python3 extract_year_topics.py --year 1990
    python3 extract_year_topics.py --start-year 1990 --end-year 2025
    python3 extract_year_topics.py --year 2020 --max-articles 10 --verbose
"""

import os
import sys
import json
import re
import argparse
import logging
from datetime import datetime, timezone
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
MCP_SERVER_URL = os.environ.get('MCP_SERVER_URL', 'http://localhost:7000')
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# User-Agent header required by Wikipedia API
# See: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "DeepRedAI/1.0 (Year Topics Extractor; contact@example.com)"

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

def save_html_for_debug(html: str, year: int, output_dir: str) -> str:
    """
    Save HTML content to a file for debugging analysis.
    
    Args:
        html: HTML content to save
        year: The year being processed
        output_dir: Output directory path
        
    Returns:
        Path to the saved file
    """
    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    filename = f"year_{year}_raw.html"
    filepath = os.path.join(debug_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Saved raw HTML for debugging: {filepath}")
    return filepath


def fetch_year_page_html(year: int, save_debug: bool = False, output_dir: str = None) -> Optional[str]:
    """
    Fetch the HTML content of a Wikipedia year page using the API.
    
    Args:
        year: The year (e.g., 1990)
        save_debug: If True, save the raw HTML to a file for debugging
        output_dir: Output directory for debug files
        
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
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        logger.info(f"Fetching year page {year} from Wikipedia API...")
        response = requests.get(WIKIPEDIA_API_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'error' in data:
            logger.error(f"Wikipedia API error: {data['error'].get('info', 'Unknown error')}")
            return None
            
        if 'parse' in data and 'text' in data['parse']:
            html_content = data['parse']['text']['*']
            logger.info(f"Successfully fetched {len(html_content)} bytes of HTML")
            
            # Save HTML for debugging if requested
            if save_debug and output_dir:
                save_html_for_debug(html_content, year, output_dir)
            
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
    
    The Wikipedia year page structure has:
    - <div class="mw-heading mw-heading2"><h2 id="Events">Events</h2></div>
    - <div class="mw-heading mw-heading3"><h3 id="January">January</h3></div>
    - <ul><li> with dates like "January 1" as links, followed by event text
    - Nested <ul> for sub-events under the same date
    
    Args:
        html: HTML content of the year page
        year: The year being processed
        
    Returns:
        List of topic dictionaries with date and topic information
    """
    soup = BeautifulSoup(html, 'html.parser')
    topics = []
    
    # Find the Events section - the h2 is inside a wrapper div
    events_h2 = soup.find('h2', id='Events')
    if not events_h2:
        # Fallback: look for heading containing "Events"
        for heading in soup.find_all('h2'):
            if heading.get_text().strip().lower() == 'events':
                events_h2 = heading
                break
    
    if not events_h2:
        logger.warning(f"No Events section found for year {year}")
        return topics
    
    logger.info(f"Found Events section: {events_h2.get_text().strip()}")
    
    # Get the parent div (mw-heading wrapper) and start from there
    events_wrapper = events_h2.find_parent('div', class_='mw-heading')
    if events_wrapper:
        start_element = events_wrapper
    else:
        start_element = events_h2
    
    # Month names for parsing
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    current_month = None
    current_month_num = None
    
    # Navigate through elements after the Events heading
    current = start_element.find_next_sibling()
    while current:
        # Stop at next h2 heading (end of Events section)
        # Check both direct h2 and h2 inside mw-heading div
        if current.name == 'h2':
            break
        if current.name == 'div' and 'mw-heading2' in current.get('class', []):
            break
        
        # Check for month headings (h3 inside mw-heading div or direct h3)
        if current.name == 'div' and 'mw-heading3' in current.get('class', []):
            h3 = current.find('h3')
            if h3:
                month_text = h3.get_text().strip().lower()
                month_text = re.sub(r'\[.*?\]', '', month_text).strip()
                if month_text in months:
                    current_month = month_text.capitalize()
                    current_month_num = months[month_text]
                    logger.debug(f"Processing month: {current_month}")
        elif current.name == 'h3':
            month_text = current.get_text().strip().lower()
            month_text = re.sub(r'\[.*?\]', '', month_text).strip()
            if month_text in months:
                current_month = month_text.capitalize()
                current_month_num = months[month_text]
                logger.debug(f"Processing month: {current_month}")
        
        # Process unordered/ordered lists
        if current.name in ['ul', 'ol']:
            # Process each top-level list item
            for li in current.find_all('li', recursive=False):
                extracted = extract_events_from_li(li, year, current_month, current_month_num)
                topics.extend(extracted)
        
        current = current.find_next_sibling()
    
    logger.info(f"Extracted {len(topics)} topics from year {year}")
    return topics


def extract_wiki_links(element) -> List[Dict[str, str]]:
    """
    Extract Wikipedia article links from an HTML element.
    
    Filters out:
    - Date links (e.g., "January 1", "June 24")
    - Citation links (e.g., "#cite_note-123")
    - External links
    
    Args:
        element: BeautifulSoup element to extract links from
        
    Returns:
        List of dictionaries with 'title' and 'href' keys
    """
    links = []
    seen_titles = set()
    
    # Pattern to match date links like "January 1", "June 24", "March 15-20"
    date_pattern = re.compile(
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d+(?:\s*[-–—]\s*\d+)?$',
        re.IGNORECASE
    )
    
    for a in element.find_all('a', href=True):
        href = a.get('href', '')
        title = a.get('title', '') or a.get_text().strip()
        link_text = a.get_text().strip()
        
        # Skip non-wiki links
        if not href.startswith('/wiki/'):
            continue
        
        # Skip citation/reference links
        if '#cite' in href or href.startswith('/wiki/Help:') or href.startswith('/wiki/Wikipedia:'):
            continue
        
        # Skip date links
        if date_pattern.match(link_text):
            continue
        
        # Skip if we've already seen this title
        if title in seen_titles:
            continue
        
        seen_titles.add(title)
        
        # Extract clean article path (remove /wiki/ prefix)
        article_path = href[6:]  # Remove '/wiki/'
        # Handle fragment identifiers (e.g., "Article#Section")
        if '#' in article_path:
            article_path = article_path.split('#')[0]
        
        links.append({
            'title': title,
            'href': href,
            'article': article_path
        })
    
    return links


def extract_events_from_li(li, year: int, current_month: str, current_month_num: int) -> List[Dict[str, Any]]:
    """
    Extract events from a list item element.
    
    Handles structures like:
    - <li><a href="/wiki/January_1">January 1</a> – Event description</li>
    - <li><a href="/wiki/January_1">January 1</a><ul><li>Sub-event 1</li><li>Sub-event 2</li></ul></li>
    
    Also extracts Wikipedia article links from each event for direct reference.
    
    Args:
        li: BeautifulSoup li element
        year: The year being processed
        current_month: Current month name (e.g., "January")
        current_month_num: Current month number (1-12)
        
    Returns:
        List of topic dictionaries with 'wiki_links' field containing referenced articles
    """
    topics = []
    
    # Get the first link in the li - this is usually the date
    first_link = li.find('a', recursive=False)
    if not first_link:
        # Try finding first link anywhere in direct text
        first_link = li.find('a')
    
    date_text = None
    day = None
    full_date = None
    month_num = current_month_num
    
    if first_link:
        link_text = first_link.get_text().strip()
        # Check if this is a date link (e.g., "January 1", "January 1-3")
        date_match = re.match(r'^([A-Za-z]+)\s+(\d+)(?:\s*[-–—]\s*\d+)?$', link_text)
        if date_match:
            month_name = date_match.group(1).lower()
            months = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            if month_name in months:
                month_num = months[month_name]
                day = int(date_match.group(2))
                full_date = f"{year:04d}-{month_num:02d}-{day:02d}"
                date_text = link_text
    
    # Check for nested ul (sub-events)
    nested_ul = li.find('ul', recursive=False)
    
    if nested_ul:
        # Has sub-events - process each sub-item
        for sub_li in nested_ul.find_all('li', recursive=False):
            event_text = sub_li.get_text().strip()
            if event_text:
                # Clean up the text
                event_text = re.sub(r'\s+', ' ', event_text)
                # Remove citation references like [12], [6][7], etc.
                event_text = re.sub(r'\[\d+\]', '', event_text).strip()
                
                # Extract Wikipedia links from this sub-event
                wiki_links = extract_wiki_links(sub_li)
                
                topic_dict = {
                    'year': year,
                    'date': full_date,
                    'month': month_num,
                    'day': day,
                    'date_text': date_text or '',
                    'topic': event_text,
                    'wiki_links': wiki_links
                }
                topics.append(topic_dict)
                logger.debug(f"Extracted sub-event: {date_text} - {event_text[:50]}... ({len(wiki_links)} links)")
    else:
        # No sub-events - get the main event text
        # Get the full text of the li
        full_text = li.get_text().strip()
        
        # Try to separate date from event description
        # Pattern: "January 1 – Event description" or "January 1-3 – Event description"
        match = re.match(r'^([A-Za-z]+\s+\d+(?:\s*[-–—]\s*\d+)?)\s*[–—]\s*(.+)$', full_text, re.DOTALL)
        
        if match:
            date_text = match.group(1).strip()
            event_text = match.group(2).strip()
            
            # Parse date from the extracted date text
            date_match = re.match(r'^([A-Za-z]+)\s+(\d+)', date_text)
            if date_match:
                month_name = date_match.group(1).lower()
                months = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12
                }
                if month_name in months:
                    month_num = months[month_name]
                    day = int(date_match.group(2))
                    full_date = f"{year:04d}-{month_num:02d}-{day:02d}"
        else:
            # No clear date separator, use the whole text
            event_text = full_text
        
        # Clean up event text
        event_text = re.sub(r'\s+', ' ', event_text).strip()
        # Remove citation references like [12], [6][7], etc.
        event_text = re.sub(r'\[\d+\]', '', event_text).strip()
        
        if event_text:
            # Extract Wikipedia links from this event
            wiki_links = extract_wiki_links(li)
            
            topic_dict = {
                'date': full_date,
                'year': year,
                'month': month_num,
                'day': day,
                'date_text': date_text or '',
                'topic': event_text,
                'wiki_links': wiki_links
            }
            topics.append(topic_dict)
            logger.debug(f"Extracted event: {date_text} - {event_text[:50]}... ({len(wiki_links)} links)")
    
    return topics


# -----------------------------------------------------------------------------
# MCP Server Search Functions
# -----------------------------------------------------------------------------

def lookup_article_id(title: str) -> Optional[int]:
    """
    Look up the article ID for a given Wikipedia article title.
    
    Uses keyword search with the exact title to find the article ID.
    
    Args:
        title: The article title to look up
        
    Returns:
        Article ID if found, None otherwise
    """
    url = urljoin(MCP_SERVER_URL, '/mcp/search')
    
    payload = {
        'query': title,
        'mode': 'keyword',  # Use keyword search for exact title matching
        'limit': 5
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        # Look for exact or close title match
        title_lower = title.lower().strip()
        for result in results:
            result_title = result.get('title', '').lower().strip()
            if result_title == title_lower:
                article_id = result.get('article_id') or result.get('id')
                if article_id:
                    logger.debug(f"Found article ID {article_id} for '{title}'")
                    return article_id
        
        # If no exact match, return the first result's ID if available
        if results:
            article_id = results[0].get('article_id') or results[0].get('id')
            if article_id:
                logger.debug(f"Using first result ID {article_id} for '{title}'")
                return article_id
        
        logger.debug(f"No article ID found for '{title}'")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to lookup article ID for '{title}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse lookup response for '{title}': {e}")
        return None


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


def deduplicate_articles(articles: List[Dict[str, Any]], key_field: str = 'title') -> List[Dict[str, Any]]:
    """
    Deduplicate a list of articles based on a key field.
    
    Keeps the first occurrence of each unique article.
    Also deduplicates by article_id if available.
    
    Args:
        articles: List of article dictionaries
        key_field: Field to use for deduplication (default: 'title')
        
    Returns:
        Deduplicated list of articles
    """
    seen_titles = set()
    seen_ids = set()
    deduplicated = []
    
    for article in articles:
        title = article.get(key_field, '').lower().strip()
        article_id = article.get('article_id')
        
        # Skip if we've seen this title or ID
        if title and title in seen_titles:
            continue
        if article_id and article_id in seen_ids:
            continue
        
        if title:
            seen_titles.add(title)
        if article_id:
            seen_ids.add(article_id)
        
        deduplicated.append(article)
    
    return deduplicated


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

def process_year(year: int, max_articles: int = DEFAULT_MAX_ARTICLES, output_dir: str = None, save_debug: bool = False) -> Dict[str, Any]:
    """
    Process a single year: fetch page, extract topics, find related articles.
    
    Args:
        year: The year to process
        max_articles: Maximum number of related articles per topic
        output_dir: Output directory for saving debug files
        save_debug: If True, save raw HTML for debugging
        
    Returns:
        Dictionary with year data and topics
    """
    logger.info(f"Processing year {year}...")
    
    # Fetch the year page HTML
    html = fetch_year_page_html(year, save_debug=save_debug, output_dir=output_dir)
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
        wiki_links = topic.get('wiki_links', [])
        logger.info(f"Processing topic {idx}/{len(topics)}: {topic_text[:60]}... ({len(wiki_links)} direct links)")
        
        # Deduplicate wiki_links first
        seen_titles = set()
        deduplicated_wiki_links = []
        for link in wiki_links:
            title_lower = link.get('title', '').lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                deduplicated_wiki_links.append(link)
        topic['wiki_links'] = deduplicated_wiki_links
        
        # Add direct references from wiki_links with article_id lookup
        direct_references = []
        for link in deduplicated_wiki_links:
            title = link.get('title', '')
            
            # Look up article ID for this direct reference
            article_id = lookup_article_id(title)
            
            direct_ref = {
                'title': title,
                'article_path': link.get('article', ''),
                'href': link.get('href', ''),
                'source': 'direct_link',
                'relevance_score': 1.0  # Direct links are highly relevant
            }
            
            # Add article_id if found
            if article_id:
                direct_ref['article_id'] = article_id
            
            direct_references.append(direct_ref)
        
        # Deduplicate direct_references
        topic['direct_references'] = deduplicate_articles(direct_references)
        
        # Search for related articles (in addition to direct links)
        related_articles = find_related_articles(topic_text, max_articles)
        
        # Filter out articles that are already in direct_references (by title or article_id)
        direct_titles = {ref['title'].lower() for ref in topic['direct_references']}
        direct_ids = {ref.get('article_id') for ref in topic['direct_references'] if ref.get('article_id')}
        
        filtered_related = [
            article for article in related_articles
            if article['title'].lower() not in direct_titles
            and (not article.get('article_id') or article.get('article_id') not in direct_ids)
        ]
        
        # Deduplicate related_articles
        topic['related_articles'] = deduplicate_articles(filtered_related)
        
        # Remove wiki_links from output (duplicate of direct_references)
        if 'wiki_links' in topic:
            del topic['wiki_links']
        
        processed_topics.append(topic)
    
    # Create output structure
    year_data = {
        'year': year,
        'extracted_date': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'source': 'wikipedia_api',
        'total_topics': len(processed_topics),
        'topics': processed_topics
    }
    
    return year_data


def check_output_directory(output_dir: str) -> bool:
    """
    Check if the output directory exists and is writable.
    Creates the directory if it doesn't exist.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        True if directory is writable, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if directory is writable by attempting to create a temp file
        test_file = os.path.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        logger.info(f"Output directory is writable: {output_dir}")
        return True
        
    except PermissionError:
        logger.error(f"Permission denied: Cannot write to output directory: {output_dir}")
        return False
    except OSError as e:
        logger.error(f"Cannot access output directory {output_dir}: {e}")
        return False


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
    
    parser.add_argument(
        '--save-html',
        action='store_true',
        help='Save raw HTML to debug/ folder for analysis'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine output directory
    output_dir = args.output_dir or os.path.join(WIKI_DATA, 'topics')
    
    # Check if output directory is writable
    if not check_output_directory(output_dir):
        logger.error("Exiting: Output directory is not writable")
        return 1
    
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
            year_data = process_year(year, args.max_articles, output_dir=output_dir, save_debug=args.save_html)
            
            if year_data:
                save_year_data(year_data, output_dir)
                success_count += 1
            else:
                logger.error(f"Failed to process year {year}")
                
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}", exc_info=True)
    
    # Summary
    logger.info("Processing complete!")
    logger.info(f"Successfully processed: {success_count}/{len(list(years))} years")
    logger.info(f"Output location: {output_dir}")
    
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
