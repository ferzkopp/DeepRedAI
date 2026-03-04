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
    python3 extract_year_topics.py --start-year 2000 --end-year 2025 --resume
"""

import os
import sys
import json
import re
import argparse
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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

_DEEPRED_ROOT = os.environ.get('DEEPRED_ROOT', '/mnt/data')
WIKI_DATA = os.environ.get('WIKI_DATA', os.path.join(_DEEPRED_ROOT, 'wikipedia'))

_MCP_HOST = os.environ.get('MCP_HOST', 'localhost')
_MCP_PORT = os.environ.get('MCP_PORT', '7000')
MCP_SERVER_URL = os.environ.get('MCP_SERVER_URL', f'http://{_MCP_HOST}:{_MCP_PORT}')

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# User-Agent header required by Wikipedia API
# See: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "DeepRedAI/1.0 (Year Topics Extractor; contact@example.com)"

# Search parameters
DEFAULT_MAX_ARTICLES = 5
MAX_SEARCH_RESULTS = 10

# Rate limiting for Wikipedia API (seconds between requests)
WIKIPEDIA_API_DELAY = 1.0

# HTTP timeout (seconds)
HTTP_TIMEOUT = 30

# Parallel workers for local MCP lookups
DEFAULT_WORKERS = 8

# Month name → number lookup (single source of truth)
MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-local HTTP sessions for safe parallel connection pooling
_thread_local = threading.local()


def get_session() -> requests.Session:
    """Return a thread-local requests.Session for connection pooling."""
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({'Content-Type': 'application/json'})
    return _thread_local.session


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

    headers = {'User-Agent': USER_AGENT}

    try:
        logger.info(f"Fetching year page {year} from Wikipedia API...")
        response = requests.get(
            WIKIPEDIA_API_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        if 'error' in data:
            logger.error(f"Wikipedia API error: {data['error'].get('info', 'Unknown error')}")
            return None

        if 'parse' in data and 'text' in data['parse']:
            html_content = data['parse']['text']['*']
            logger.info(f"Successfully fetched {len(html_content):,} bytes of HTML for year {year}")

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
        Tuple of (full_date, month, day).
    """
    # Try "Month Day" (e.g., "January 1")
    match = re.match(r'([A-Za-z]+)\s+(\d+)', date_text, re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        day = int(match.group(2))
        if month_name in MONTHS:
            month = MONTHS[month_name]
            full_date = f"{year:04d}-{month:02d}-{day:02d}"
            return full_date, month, day

    # Try "Month" only
    match = re.match(r'^([A-Za-z]+)$', date_text.strip(), re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        if month_name in MONTHS:
            return None, MONTHS[month_name], None

    return None, None, None


def extract_wiki_links(element) -> List[Dict[str, str]]:
    """
    Extract Wikipedia article links from an HTML element.

    Filters out date links, citation links, and external links.

    Args:
        element: BeautifulSoup element to extract links from

    Returns:
        List of dictionaries with 'title', 'href', and 'article' keys.
    """
    links = []
    seen_titles = set()

    date_pattern = re.compile(
        r'^(January|February|March|April|May|June|July|August|September'
        r'|October|November|December)\s+\d+(?:\s*[-–—]\s*\d+)?$',
        re.IGNORECASE
    )

    for a in element.find_all('a', href=True):
        href = a.get('href', '')
        title = a.get('title', '') or a.get_text().strip()
        link_text = a.get_text().strip()

        # Skip non-wiki, citation, and meta links
        if not href.startswith('/wiki/'):
            continue
        if '#cite' in href or href.startswith('/wiki/Help:') or href.startswith('/wiki/Wikipedia:'):
            continue
        if date_pattern.match(link_text):
            continue

        title_lower = title.lower()
        if title_lower in seen_titles:
            continue
        seen_titles.add(title_lower)

        # Clean article path
        article_path = href[6:]  # Remove '/wiki/'
        if '#' in article_path:
            article_path = article_path.split('#')[0]

        links.append({
            'title': title,
            'href': href,
            'article': article_path
        })

    return links


def _clean_event_text(text: str) -> str:
    """Collapse whitespace and strip citation references like [12]."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[\d+\]', '', text).strip()
    return text


def extract_events_from_li(li, year: int, current_month: str, current_month_num: int) -> List[Dict[str, Any]]:
    """
    Extract events from a list item element.

    Handles both flat events and nested sub-event lists.  Also extracts
    Wikipedia article links from each event for direct reference.

    Args:
        li: BeautifulSoup li element
        year: The year being processed
        current_month: Current month name (e.g., "January")
        current_month_num: Current month number (1-12)

    Returns:
        List of topic dictionaries with 'wiki_links' field.
    """
    topics = []

    # Detect leading date link
    first_link = li.find('a', recursive=False) or li.find('a')

    date_text = None
    day = None
    full_date = None
    month_num = current_month_num

    if first_link:
        link_text = first_link.get_text().strip()
        date_match = re.match(r'^([A-Za-z]+)\s+(\d+)(?:\s*[-–—]\s*\d+)?$', link_text)
        if date_match:
            month_name = date_match.group(1).lower()
            if month_name in MONTHS:
                month_num = MONTHS[month_name]
                day = int(date_match.group(2))
                full_date = f"{year:04d}-{month_num:02d}-{day:02d}"
                date_text = link_text

    # Helper to build a topic dict
    def _make_topic(event_text: str, wiki_links: list) -> Dict[str, Any]:
        return {
            'year': year,
            'date': full_date,
            'month': month_num,
            'day': day,
            'date_text': date_text or '',
            'topic': event_text,
            'wiki_links': wiki_links
        }

    nested_ul = li.find('ul', recursive=False)

    if nested_ul:
        for sub_li in nested_ul.find_all('li', recursive=False):
            event_text = _clean_event_text(sub_li.get_text())
            if event_text:
                wiki_links = extract_wiki_links(sub_li)
                topics.append(_make_topic(event_text, wiki_links))
                logger.debug(f"  sub-event: {date_text} – {event_text[:50]}… ({len(wiki_links)} links)")
    else:
        full_text = li.get_text().strip()

        # "January 1 – Event description"
        match = re.match(
            r'^([A-Za-z]+\s+\d+(?:\s*[-–—]\s*\d+)?)\s*[–—]\s*(.+)$',
            full_text, re.DOTALL
        )
        if match:
            date_text = match.group(1).strip()
            event_text = match.group(2).strip()

            date_match = re.match(r'^([A-Za-z]+)\s+(\d+)', date_text)
            if date_match:
                month_name = date_match.group(1).lower()
                if month_name in MONTHS:
                    month_num = MONTHS[month_name]
                    day = int(date_match.group(2))
                    full_date = f"{year:04d}-{month_num:02d}-{day:02d}"
        else:
            event_text = full_text

        event_text = _clean_event_text(event_text)

        if event_text:
            wiki_links = extract_wiki_links(li)
            topics.append(_make_topic(event_text, wiki_links))
            logger.debug(f"  event: {date_text} – {event_text[:50]}… ({len(wiki_links)} links)")

    return topics


def extract_topics_from_html(html: str, year: int) -> List[Dict[str, Any]]:
    """
    Extract topics and dates from Wikipedia year page HTML.

    Args:
        html: HTML content of the year page
        year: The year being processed

    Returns:
        List of topic dictionaries with date and topic information.
    """
    soup = BeautifulSoup(html, 'html.parser')
    topics = []

    # Locate the "Events" heading
    events_h2 = soup.find('h2', id='Events')
    if not events_h2:
        for heading in soup.find_all('h2'):
            if heading.get_text().strip().lower() == 'events':
                events_h2 = heading
                break

    if not events_h2:
        logger.warning(f"No Events section found for year {year}")
        return topics

    logger.info(f"Found Events section for year {year}")

    # Navigate from the wrapper div (or the h2 itself)
    events_wrapper = events_h2.find_parent('div', class_='mw-heading')
    start_element = events_wrapper if events_wrapper else events_h2

    current_month = None
    current_month_num = None

    current = start_element.find_next_sibling()
    while current:
        # Stop at next h2 heading (end of Events section)
        if current.name == 'h2':
            break
        if current.name == 'div' and 'mw-heading2' in current.get('class', []):
            break

        # Detect month headings (h3 in wrapper div or direct h3)
        if current.name == 'div' and 'mw-heading3' in current.get('class', []):
            h3 = current.find('h3')
            if h3:
                month_text = re.sub(r'\[.*?\]', '', h3.get_text().strip()).lower()
                if month_text in MONTHS:
                    current_month = month_text.capitalize()
                    current_month_num = MONTHS[month_text]
                    logger.debug(f"Processing month: {current_month}")
        elif current.name == 'h3':
            month_text = re.sub(r'\[.*?\]', '', current.get_text().strip()).lower()
            if month_text in MONTHS:
                current_month = month_text.capitalize()
                current_month_num = MONTHS[month_text]
                logger.debug(f"Processing month: {current_month}")

        # Process list elements
        if current.name in ['ul', 'ol']:
            for li in current.find_all('li', recursive=False):
                extracted = extract_events_from_li(li, year, current_month, current_month_num)
                topics.extend(extracted)

        current = current.find_next_sibling()

    logger.info(f"Extracted {len(topics)} topics from year {year}")
    return topics


# -----------------------------------------------------------------------------
# MCP Server Search Functions
# -----------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def lookup_article_id(title: str) -> Optional[int]:
    """
    Look up the article ID for a Wikipedia article title via the MCP server.

    Results are cached (LRU, 4096 entries) so repeated titles across topics
    do not trigger redundant HTTP calls.

    Args:
        title: The article title to look up

    Returns:
        Article ID if found, None otherwise.
    """
    url = urljoin(MCP_SERVER_URL, '/mcp/search')
    payload = {
        'query': title,
        'mode': 'keyword',
        'limit': 5
    }

    try:
        response = get_session().post(url, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        results = response.json().get('results', [])

        # Prefer exact title match
        title_lower = title.lower().strip()
        for result in results:
            if result.get('title', '').lower().strip() == title_lower:
                article_id = result.get('article_id') or result.get('id')
                if article_id:
                    logger.debug(f"Exact match – article ID {article_id} for '{title}'")
                    return article_id

        # Fall back to first result
        if results:
            article_id = results[0].get('article_id') or results[0].get('id')
            if article_id:
                logger.debug(f"First-result ID {article_id} for '{title}'")
                return article_id

        logger.debug(f"No article ID found for '{title}'")
        return None

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logger.warning(f"Lookup failed for '{title}': {e}")
        return None


def _batch_lookup_article_ids(
    titles: List[str], max_workers: int = DEFAULT_WORKERS
) -> Dict[str, Optional[int]]:
    """
    Look up article IDs for many titles in parallel.

    Uses a :class:`~concurrent.futures.ThreadPoolExecutor` to issue
    concurrent HTTP requests to the MCP server.  Each call goes through
    the LRU-cached :func:`lookup_article_id`, so duplicate titles across
    topics are resolved from cache automatically.

    Args:
        titles: Unique article titles to look up.
        max_workers: Number of parallel threads.

    Returns:
        Dict mapping each title to its article ID (or *None*).
    """
    results: Dict[str, Optional[int]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {
            executor.submit(lookup_article_id, title): title
            for title in titles
        }

        iter_futures = as_completed(future_to_title)
        if HAS_TQDM:
            iter_futures = tqdm(
                iter_futures, total=len(titles),
                desc='    Article IDs', unit=' titles',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                leave=False,
            )

        for future in iter_futures:
            title = future_to_title[future]
            try:
                results[title] = future.result()
            except Exception as exc:
                logger.warning(f"Parallel lookup failed for '{title}': {exc}")
                results[title] = None

    return results


def search_mcp_server(query: str, limit: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
    """
    Search the Wikipedia MCP server for articles related to a query.

    Args:
        query: Search query text
        limit: Maximum number of results to return

    Returns:
        List of search result dictionaries.
    """
    url = urljoin(MCP_SERVER_URL, '/mcp/search')
    payload = {
        'query': query,
        'mode': 'hybrid',
        'limit': limit
    }

    try:
        response = get_session().post(url, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        results = response.json().get('results', [])
        logger.debug(f"Found {len(results)} results for: {query[:50]}…")
        return results

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logger.error(f"MCP search failed for '{query[:60]}': {e}")
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

    Weighted: 40% title similarity, 40% search score, 20% position.
    """
    title_similarity = fuzz.token_sort_ratio(topic.lower(), article_title.lower()) / 100.0
    position_penalty = 1.0 - (position / max(total_results, 1))
    normalized_search_score = min(search_score / 20.0, 1.0)

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
    Deduplicate a list of articles by title and article_id.
    Keeps the first occurrence of each unique entry.
    """
    seen_titles = set()
    seen_ids = set()
    deduplicated = []

    for article in articles:
        title = article.get(key_field, '').lower().strip()
        article_id = article.get('article_id')

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
    Find Wikipedia articles related to a topic via hybrid search,
    scored and sorted by relevance.
    """
    search_results = search_mcp_server(topic, limit=MAX_SEARCH_RESULTS)
    if not search_results:
        return []

    related_articles = []
    for idx, result in enumerate(search_results):
        title = result.get('title', '')
        article_id = result.get('article_id') or result.get('id')
        search_score = result.get('score', 0.0)

        metrics = calculate_relevance_score(
            topic, title, search_score, idx, len(search_results)
        )

        related_articles.append({
            'title': title,
            'article_id': article_id,
            'relevance_score': metrics['relevance_score'],
            'search_score': metrics['search_score'],
            'title_similarity': metrics['title_similarity']
        })

    related_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
    return related_articles[:max_articles]


# -----------------------------------------------------------------------------
# Temporal Validation Functions
# -----------------------------------------------------------------------------

# In-memory cache: article_id → earliest-date year (int) or None
_temporal_cache: Dict[int, Optional[int]] = {}


def fetch_temporal_info(article_ids: List[int]) -> Dict[int, Optional[int]]:
    """
    Batch-fetch temporal augmentation data from the MCP server.

    Queries ``POST /mcp/temporal`` for each uncached article ID and updates
    the module-level ``_temporal_cache``.

    Args:
        article_ids: List of article IDs to look up.

    Returns:
        Dict mapping each article_id to its earliest-date year (int) or
        *None* when no temporal information is available.
    """
    # Determine which IDs still need fetching
    uncached = [aid for aid in article_ids if aid not in _temporal_cache]

    if uncached:
        url = urljoin(MCP_SERVER_URL, '/mcp/temporal')
        # MCP endpoint accepts up to 500 IDs per call; chunk if needed
        for start in range(0, len(uncached), 500):
            chunk = uncached[start:start + 500]
            try:
                resp = get_session().post(
                    url, json={'article_ids': chunk}, timeout=HTTP_TIMEOUT
                )
                resp.raise_for_status()
                for item in resp.json().get('results', []):
                    aid = item['article_id']
                    if item.get('has_temporal_info') and item.get('earliest_date'):
                        try:
                            year = int(item['earliest_date'].split('-')[0])
                            _temporal_cache[aid] = year
                        except (ValueError, IndexError):
                            _temporal_cache[aid] = None
                    else:
                        _temporal_cache[aid] = None
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logger.warning(f"Temporal lookup failed for batch: {e}")
                # Mark as unknown so we don't retry endlessly
                for aid in chunk:
                    _temporal_cache.setdefault(aid, None)

    return {aid: _temporal_cache.get(aid) for aid in article_ids}


def filter_by_temporal_year(
    articles: List[Dict[str, Any]], year: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove articles whose earliest temporal year is after *year*.

    Articles without temporal information (or without an article_id) are
    kept – only those with a known earliest_date year > *year* are dropped.

    Args:
        articles: List of article dicts (must contain ``article_id``).
        year: The topic year; articles with earliest_date year > this are
              excluded.

    Returns:
        Tuple of (filtered list, number of articles removed).
    """
    ids = [a['article_id'] for a in articles if a.get('article_id')]
    if not ids:
        return articles, 0

    temporal_map = fetch_temporal_info(ids)

    kept: List[Dict[str, Any]] = []
    removed = 0
    for article in articles:
        aid = article.get('article_id')
        if aid and temporal_map.get(aid) is not None:
            if temporal_map[aid] > year:
                logger.debug(
                    f"Excluding '{article.get('title', '')}' (id={aid}): "
                    f"earliest year {temporal_map[aid]} > topic year {year}"
                )
                removed += 1
                continue
        kept.append(article)
    return kept, removed


# -----------------------------------------------------------------------------
# Main Processing Functions
# -----------------------------------------------------------------------------

def process_year(
    year: int,
    max_articles: int = DEFAULT_MAX_ARTICLES,
    output_dir: str = None,
    save_debug: bool = False,
    dry_run: bool = False,
    max_workers: int = DEFAULT_WORKERS
) -> Optional[Dict[str, Any]]:
    """
    Process a single year: fetch page, extract topics, find related articles.

    Article-ID lookups and related-article searches are parallelised using
    a :class:`~concurrent.futures.ThreadPoolExecutor` (``max_workers``
    threads) for significantly faster throughput against a local MCP server.

    Args:
        year: The year to process
        max_articles: Maximum number of related articles per topic
        output_dir: Output directory for saving debug files
        save_debug: If True, save raw HTML for debugging
        dry_run: If True, extract topics but skip MCP article lookups
        max_workers: Number of parallel threads for MCP lookups

    Returns:
        Dictionary with year data and topics, or None on failure.
    """
    logger.info(f"Processing year {year}…")

    html = fetch_year_page_html(year, save_debug=save_debug, output_dir=output_dir)
    if not html:
        logger.error(f"Failed to fetch year page for {year}")
        return None

    topics = extract_topics_from_html(html, year)
    if not topics:
        logger.warning(f"No topics extracted for year {year}")
        return None

    if dry_run:
        # Strip wiki_links and return without MCP lookups
        for topic in topics:
            topic['direct_references'] = []
            topic['related_articles'] = []
            topic.pop('wiki_links', None)

        return {
            'year': year,
            'extracted_date': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'source': 'wikipedia_api',
            'total_topics': len(topics),
            'topics': topics
        }

    # ── Phase 1: Pre-deduplicate wiki links & collect unique titles ──────
    all_titles = set()
    topics_dedup_links = []

    for topic in topics:
        wiki_links = topic.get('wiki_links', [])
        seen = set()
        dedup = []
        for link in wiki_links:
            t = link.get('title', '').lower().strip()
            if t and t not in seen:
                seen.add(t)
                dedup.append(link)
                all_titles.add(link['title'])
        topics_dedup_links.append(dedup)

    logger.info(
        f"Year {year}: {len(topics)} topics, "
        f"{len(all_titles)} unique wiki-link titles to resolve"
    )

    # ── Phase 2: Parallel article-ID lookups ─────────────────────────────
    if all_titles:
        title_to_id = _batch_lookup_article_ids(
            list(all_titles), max_workers=max_workers
        )
    else:
        title_to_id = {}

    # ── Phase 3: Parallel related-article searches ───────────────────────
    topic_related = [None] * len(topics)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(find_related_articles, topic['topic'], max_articles): idx
            for idx, topic in enumerate(topics)
        }

        iter_futures = as_completed(future_to_idx)
        if HAS_TQDM:
            iter_futures = tqdm(
                iter_futures, total=len(topics),
                desc=f'    Year {year} search', unit=' topics',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                leave=False,
            )

        for future in iter_futures:
            idx = future_to_idx[future]
            try:
                topic_related[idx] = future.result()
            except Exception as exc:
                logger.warning(f"Related-article search failed for topic {idx}: {exc}")
                topic_related[idx] = []

    # ── Phase 4: Assemble direct refs & related articles ─────────────────
    processed_topics = []
    all_article_ids = set()

    for idx, topic in enumerate(topics):
        dedup_links = topics_dedup_links[idx]

        # Build direct references (title_to_id is already populated)
        direct_references = []
        for link in dedup_links:
            title = link.get('title', '')
            article_id = title_to_id.get(title)

            direct_ref = {
                'title': title,
                'article_path': link.get('article', ''),
                'href': link.get('href', ''),
                'source': 'direct_link',
                'relevance_score': 1.0
            }
            if article_id:
                direct_ref['article_id'] = article_id
                all_article_ids.add(article_id)
            direct_references.append(direct_ref)

        topic['direct_references'] = deduplicate_articles(direct_references)

        # Apply related articles, filtering out duplicates of direct refs
        related_articles = topic_related[idx] or []
        direct_titles = {ref['title'].lower() for ref in topic['direct_references']}
        direct_ids = {
            ref.get('article_id') for ref in topic['direct_references']
            if ref.get('article_id')
        }

        filtered_related = [
            a for a in related_articles
            if a['title'].lower() not in direct_titles
            and (not a.get('article_id') or a.get('article_id') not in direct_ids)
        ]
        topic['related_articles'] = deduplicate_articles(filtered_related)

        for ref in topic['related_articles']:
            if ref.get('article_id'):
                all_article_ids.add(ref['article_id'])

        topic.pop('wiki_links', None)
        processed_topics.append(topic)

    # ── Phase 5: Batch temporal validation ────────────────────────────────
    temporal_filtered_total = 0
    if all_article_ids:
        logger.info(f"Temporal validation for {len(all_article_ids)} unique articles…")
        fetch_temporal_info(list(all_article_ids))  # pre-warm cache in one batch

    direct_ref_total = 0
    related_total = 0

    for topic in processed_topics:
        topic['direct_references'], dr_removed = filter_by_temporal_year(
            topic['direct_references'], year
        )
        topic['related_articles'], ra_removed = filter_by_temporal_year(
            topic['related_articles'], year
        )
        temporal_filtered_total += dr_removed + ra_removed
        direct_ref_total += len(topic['direct_references'])
        related_total += len(topic['related_articles'])

    logger.info(
        f"Year {year}: {len(processed_topics)} topics, "
        f"{direct_ref_total} direct refs, {related_total} related articles, "
        f"{temporal_filtered_total} excluded by temporal check"
    )

    return {
        'year': year,
        'extracted_date': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'source': 'wikipedia_api',
        'total_topics': len(processed_topics),
        'topics': processed_topics
    }


def check_output_directory(output_dir: str) -> bool:
    """Ensure output directory exists and is writable."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except (PermissionError, OSError) as e:
        logger.error(f"Cannot write to output directory {output_dir}: {e}")
        return False


def save_year_data(year_data: Dict[str, Any], output_dir: str) -> str:
    """
    Save year data to JSON file using atomic write (temp + rename).

    Returns:
        Path to the saved file.
    """
    year = year_data['year']
    filename = f"year_topics_{year}.json"
    filepath = os.path.join(output_dir, filename)
    temp_path = filepath + '.tmp'

    os.makedirs(output_dir, exist_ok=True)

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(year_data, f, indent=2, ensure_ascii=False)

    os.replace(temp_path, filepath)  # atomic on POSIX
    logger.info(f"Saved {filepath}")
    return filepath


def year_already_processed(year: int, output_dir: str) -> bool:
    """Check whether a year's output file already exists."""
    filepath = os.path.join(output_dir, f"year_topics_{year}.json")
    return os.path.isfile(filepath)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract year topics from Wikipedia and find related articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single year
  python3 extract_year_topics.py --year 2020

  # Process a range of years
  python3 extract_year_topics.py --start-year 1990 --end-year 2025

  # Resume an interrupted range (skip already-saved years)
  python3 extract_year_topics.py --start-year 1990 --end-year 2025 --resume

  # Dry-run: extract topics from HTML only, no MCP lookups
  python3 extract_year_topics.py --year 2020 --dry-run

  # Verbose logging with custom output directory
  python3 extract_year_topics.py --year 2020 --verbose --output-dir /tmp/topics

Environment variables (defaults from deepred-env.sh):
  DEEPRED_ROOT   Base data directory          (default: /mnt/data)
  WIKI_DATA      Wikipedia data directory     (default: $DEEPRED_ROOT/wikipedia)
  MCP_HOST       MCP server hostname          (default: localhost)
  MCP_PORT       MCP server port              (default: 7000)
  MCP_SERVER_URL Override full MCP URL         (default: http://$MCP_HOST:$MCP_PORT)
        """
    )

    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument('--year', type=int, help='Single year to process (e.g., 1990)')
    year_group.add_argument('--start-year', type=int, help='Start year for range processing')

    parser.add_argument('--end-year', type=int, help='End year for range (required with --start-year)')
    parser.add_argument(
        '--max-articles', type=int, default=DEFAULT_MAX_ARTICLES,
        help=f'Max related articles per topic (default: {DEFAULT_MAX_ARTICLES})'
    )
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: $WIKI_DATA/topics/)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose (DEBUG) logging')
    parser.add_argument('--save-html', action='store_true', help='Save raw HTML to debug/ folder for analysis')
    parser.add_argument('--dry-run', action='store_true', help='Extract topics from HTML only, skip MCP lookups')
    parser.add_argument('--resume', '-r', action='store_true', help='Skip years whose output file already exists')
    parser.add_argument(
        '--workers', '-w', type=int, default=DEFAULT_WORKERS,
        help=f'Parallel workers for MCP lookups (default: {DEFAULT_WORKERS})'
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Suppress noisy library loggers even in verbose mode
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

    output_dir = args.output_dir or os.path.join(WIKI_DATA, 'topics')

    if not check_output_directory(output_dir):
        logger.error("Exiting: output directory is not writable")
        return 1

    # Determine years to process
    if args.year:
        years = [args.year]
    else:
        if not args.end_year:
            logger.error("--end-year is required when using --start-year")
            return 1
        if args.end_year < args.start_year:
            logger.error("--end-year must be >= --start-year")
            return 1
        years = list(range(args.start_year, args.end_year + 1))

    # Resume support: filter out already-processed years
    if args.resume:
        before = len(years)
        years = [y for y in years if not year_already_processed(y, output_dir)]
        skipped = before - len(years)
        if skipped:
            logger.info(f"Resume: skipping {skipped} already-processed year(s)")
        if not years:
            logger.info("All years already processed – nothing to do")
            return 0

    logger.info(f"Processing {len(years)} year(s): {years[0]}–{years[-1] if len(years) > 1 else years[0]}")
    if args.dry_run:
        logger.info("Dry-run mode: MCP article lookups disabled")
    else:
        logger.info(f"Using {args.workers} parallel worker(s) for MCP lookups")

    success_count = 0
    fail_count = 0
    total_topics = 0

    try:
        for i, year in enumerate(years, 1):
            # Rate-limit Wikipedia API calls between years
            if i > 1:
                time.sleep(WIKIPEDIA_API_DELAY)

            try:
                year_data = process_year(
                    year, args.max_articles, output_dir=output_dir,
                    save_debug=args.save_html, dry_run=args.dry_run,
                    max_workers=args.workers
                )

                if year_data:
                    save_year_data(year_data, output_dir)
                    success_count += 1
                    total_topics += year_data['total_topics']
                else:
                    fail_count += 1
                    logger.error(f"Failed to process year {year}")

            except Exception as e:
                fail_count += 1
                logger.error(f"Error processing year {year}: {e}", exc_info=True)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        logger.info(f"Progress saved. Re-run with --resume to continue.")

    # Summary
    logger.info("─" * 50)
    logger.info(f"Processing complete: {success_count}/{success_count + fail_count} years succeeded")
    logger.info(f"Total topics extracted: {total_topics:,}")
    logger.info(f"Output: {output_dir}")
    if not args.dry_run:
        cache = lookup_article_id.cache_info()
        logger.info(f"Article ID cache: {cache.hits} hits / {cache.misses} misses ({cache.currsize} entries)")

    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
