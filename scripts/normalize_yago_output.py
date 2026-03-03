#!/usr/bin/env python3
"""
Normalize YAGO/Wikidata Parser Output

This script normalizes the output from yago_parser.py or wikidata_parser.py by:
1. Extracting Wikipedia article titles from URLs (any language)
2. Translating non-English titles to English via Wikipedia API
3. Looking up Wikipedia page IDs from the local PostgreSQL database
4. Validating that articles exist in the local database

The script queries:
- Wikipedia API (to translate non-English titles to English)
- Local PostgreSQL database (to get Wikipedia page IDs and validate existence)

Usage:
    # YAGO format (default): Entity,Wikipedia_URL,Earliest_Date,Latest_Date
    python normalize_yago_output.py input.csv --output normalized.csv
    python normalize_yago_output.py input.json --output normalized.json --format json
    
    # Wikidata format: Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date
    python normalize_yago_output.py input.csv --output normalized.csv --mode wikidata
    
    # Other options
    python normalize_yago_output.py input.csv --output normalized.csv --skip-missing
    python normalize_yago_output.py input.csv --output normalized.csv --api-delay 0.5
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

import psycopg2
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database configuration — honours deepred-env.sh environment variables
DB_CONFIG = {
    'host': os.environ.get('PG_HOST', 'localhost'),
    'port': int(os.environ.get('PG_PORT', 5432)),
    'database': os.environ.get('PG_DATABASE', 'wikidb'),
    'user': os.environ.get('PG_USER', 'wiki'),
    'password': os.environ.get('PG_PASSWORD', 'wikipass')
}

# Wikipedia API configuration
WIKIPEDIA_API_TIMEOUT = 10
WIKIPEDIA_USER_AGENT = 'DeepRedAI/1.0 (Educational; https://github.com/aschiffler/DeepRedAI) YagoNormalizer'

# ---------------------------------------------------------------------------
# Helpers – transparent zstd reading / compression / reclaim
# ---------------------------------------------------------------------------

def _sizeof_fmt(num: float) -> str:
    """Human-readable file size."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(num) < 1024.0:
            return f'{num:,.1f} {unit}'
        num /= 1024.0
    return f'{num:,.1f} PB'


def _open_input(path: str, mode: str = 'r'):
    """Return a text-mode file handle; transparently decompress ``.zst`` files."""
    if path.endswith('.zst'):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        raw = open(path, 'rb')
        return io.TextIOWrapper(
            dctx.stream_reader(raw, closefd=True),
            encoding='utf-8',
            newline='',
        )
    return open(path, mode, encoding='utf-8')


def _compress_file(src: Path, dst: Path, verbose: bool = False) -> None:
    """Compress *src* to *dst* using zstd (level 3, multi-threaded).

    Falls back to the ``zstd`` CLI (via sudo if needed) when the
    destination directory is not writable by the current user.
    """
    import tempfile

    src_size = src.stat().st_size
    logging.info(f'Compressing {src.name} → {dst.name} ({_sizeof_fmt(src_size)}) …')

    try:
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        part = dst.with_suffix(dst.suffix + '.part')
        with open(src, 'rb') as fin, open(part, 'wb') as fout:
            cctx.copy_stream(fin, fout)
        part.rename(dst)
    except PermissionError:
        logging.info('Permission denied writing directly; using sudo zstd fallback …')
        with tempfile.NamedTemporaryFile(suffix='.zst', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = ['zstd', '-3', '--threads=0', '-f', str(src), '-o', str(tmp_path)]
            subprocess.run(cmd, check=True)
            subprocess.run(['sudo', 'mv', str(tmp_path), str(dst)], check=True)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    compressed_size = dst.stat().st_size
    ratio = src_size / compressed_size if compressed_size else 0
    logging.info(f'Compressed: {_sizeof_fmt(compressed_size)} ({ratio:.1f}x ratio)')


def _reclaim_file(path: Path) -> int:
    """Delete *path* and create a ``.reclaim`` marker.  Returns bytes freed."""
    if not path.exists():
        return 0
    size = path.stat().st_size
    marker = Path(str(path) + '.reclaim')
    try:
        path.unlink()
    except PermissionError:
        try:
            subprocess.run(['sudo', 'rm', '-f', str(path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logging.warning(f'Failed to remove {path.name}: {exc}')
            return 0
    marker.write_text(
        f'reclaimed {datetime.now().isoformat()} '
        f'size={size} ({_sizeof_fmt(size)})\n'
    )
    logging.info(f'Reclaimed {path.name} ({_sizeof_fmt(size)})')
    return size


class ThrottlingError(Exception):
    """Raised when Wikipedia API returns throttling/rate limit error"""
    pass


class WikipediaNormalizer:
    """Normalize YAGO Wikipedia URLs to English Wikipedia with page IDs"""
    
    def __init__(self, db_config: Dict = None, api_delay: float = 0.1, verbose: bool = False):
        """
        Initialize the normalizer with database connection
        
        Args:
            db_config: Database configuration dict
            api_delay: Delay in seconds between API calls (default: 0.1)
            verbose: If True, show tqdm progress bars
        """
        self.db_config = db_config or DB_CONFIG
        self.api_delay = api_delay
        self.verbose = verbose
        self.conn = None
        self.cursor = None
        self.cache = {}  # Cache for API lookups
        self.api_call_count = 0
        self.api_success_count = 0
        self.api_notfound_count = 0
        
    def connect_db(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logging.info("Connected to PostgreSQL database")
            return True
        except psycopg2.Error as e:
            logging.error(f"Database connection failed: {e}")
            return False
    
    def close_db(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def count_output_lines(self, output_file: str) -> int:
        """Count existing lines in output file (excluding header)"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                # Count lines minus header
                return sum(1 for _ in f) - 1
        except FileNotFoundError:
            return 0
    
    def extract_wiki_info(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Extract language code and article title from Wikipedia URL
        
        Args:
            url: Wikipedia URL (e.g., https://ar.wikipedia.org/wiki/Article_Name)
            
        Returns:
            Tuple of (language_code, article_title) or None if invalid
        """
        if not url or 'wikipedia.org/wiki/' not in url:
            return None
        
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Extract language code from domain (e.g., 'ar' from 'ar.wikipedia.org')
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) >= 2 and domain_parts[-2] == 'wikipedia':
                lang_code = domain_parts[0]
            else:
                return None
            
            # Extract article title from path
            path_parts = parsed.path.split('/wiki/')
            if len(path_parts) < 2:
                return None
            
            article_title = unquote(path_parts[1])
            
            return (lang_code, article_title)
            
        except Exception as e:
            logging.warning(f"Failed to parse URL {url}: {e}")
            return None
    
    def get_english_title_from_api(self, lang_code: str, title: str) -> Optional[str]:
        """
        Get English Wikipedia title using Wikipedia API language links
        
        Args:
            lang_code: Source language code (e.g., 'ar', 'pnb', 'ca')
            title: Article title in source language
            
        Returns:
            English Wikipedia title or None if not found
            
        Raises:
            ThrottlingError: If API returns throttling/rate limit error
        """
        # Skip if already English
        if lang_code == 'en':
            return title
        
        # Check cache
        cache_key = f"{lang_code}:{title}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Query Wikipedia API for language links
            api_url = f"https://{lang_code}.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'langlinks',
                'lllang': 'en',
                'format': 'json',
                'formatversion': 2
            }
            
            headers = {
                'User-Agent': WIKIPEDIA_USER_AGENT
            }
            
            # Log API call
            self.api_call_count += 1
            logging.debug(f"API Call #{self.api_call_count}: Translating {lang_code}:{title}")
            
            # Apply rate limiting delay
            time.sleep(self.api_delay)
            
            response = requests.get(api_url, params=params, headers=headers, timeout=WIKIPEDIA_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Extract English title from response
            pages = data.get('query', {}).get('pages', [])
            if pages and len(pages) > 0:
                page = pages[0]
                langlinks = page.get('langlinks', [])
                
                for link in langlinks:
                    if link.get('lang') == 'en':
                        en_title = link.get('title')
                        self.cache[cache_key] = en_title
                        self.api_success_count += 1
                        logging.debug(f"API Result: Found English title '{en_title}' for {lang_code}:{title}")
                        return en_title
            
            # Not found
            self.cache[cache_key] = None
            self.api_notfound_count += 1
            logging.debug(f"API Result: No English equivalent found for {lang_code}:{title}")
            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.error(f"API THROTTLING ERROR (403 Forbidden) for {lang_code}:{title}")
                logging.error(f"Rate limiting detected. Stopping script. Use --resume with increased --api-delay")
                raise ThrottlingError(f"API returned 403 Forbidden for {lang_code}:{title}")
            elif e.response.status_code == 429:
                logging.error(f"API THROTTLING ERROR (429 Too Many Requests) for {lang_code}:{title}")
                logging.error(f"Rate limiting detected. Stopping script. Use --resume with increased --api-delay")
                raise ThrottlingError(f"API returned 429 Too Many Requests for {lang_code}:{title}")
            else:
                logging.warning(f"HTTP error {e.response.status_code} for {lang_code}:{title}: {e}")
            self.cache[cache_key] = None
            return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"API request failed for {lang_code}:{title}: {e}")
            self.cache[cache_key] = None
            return None

    def get_article_from_db(self, title: str) -> Optional[Tuple[int, str]]:
        """
        Get article from local database by title
        
        Args:
            title: Wikipedia article title
            
        Returns:
            Tuple of (page_id, url) or None if not found
        """
        try:
            # Try exact match first
            self.cursor.execute(
                "SELECT id, url FROM articles WHERE title = %s",
                (title,)
            )
            result = self.cursor.fetchone()
            
            if result:
                article_id, url = result
                # Extract page_id from URL (format: https://en.wikipedia.org/wiki?curid=12345)
                match = re.search(r'curid=(\d+)', url)
                if match:
                    page_id = int(match.group(1))
                    return (page_id, url)
            
            # Try with underscores replaced by spaces
            title_with_spaces = title.replace('_', ' ')
            self.cursor.execute(
                "SELECT id, url FROM articles WHERE title = %s",
                (title_with_spaces,)
            )
            result = self.cursor.fetchone()
            
            if result:
                article_id, url = result
                match = re.search(r'curid=(\d+)', url)
                if match:
                    page_id = int(match.group(1))
                    return (page_id, url)
            
            # Check redirects table
            self.cursor.execute(
                "SELECT target_title FROM redirects WHERE source_title = %s",
                (title,)
            )
            redirect = self.cursor.fetchone()
            
            if redirect:
                target_title = redirect[0]
                self.cursor.execute(
                    "SELECT id, url FROM articles WHERE title = %s",
                    (target_title,)
                )
                result = self.cursor.fetchone()
                
                if result:
                    article_id, url = result
                    match = re.search(r'curid=(\d+)', url)
                    if match:
                        page_id = int(match.group(1))
                        return (page_id, url)
            
            return None
            
        except Exception as e:
            logging.error(f"Database query failed for title '{title}': {e}")
            return None
    
    def normalize_entry(self, entity: str, wiki_url: str, earliest_date: str, latest_date: str) -> Optional[Dict]:
        """
        Normalize a single YAGO entry
        
        Strategy:
        1. Try looking up entity name in database first (fast, avoids API calls)
        2. If not found, use Wikipedia API to translate non-English URL to English
        3. Look up translated title in database
        
        Args:
            entity: Entity name from YAGO
            wiki_url: Wikipedia URL (may be non-English)
            earliest_date: Earliest date string
            latest_date: Latest date string
            
        Returns:
            Normalized entry dict or None if cannot be normalized
            
        Raises:
            ThrottlingError: If API throttling is detected
        """
        # STEP 1: Try entity name directly in database (most efficient)
        db_result = self.get_article_from_db(entity)
        
        if db_result:
            page_id, en_url = db_result
            logging.debug(f"Found entity '{entity}' directly in database")
            return {
                'entity': entity,
                'wikipedia_title': entity.replace('_', ' '),
                'wikipedia_id': page_id,
                'wikipedia_url': en_url,
                'earliest_date': earliest_date,
                'latest_date': latest_date,
                'original_url': wiki_url if wiki_url != en_url else None
            }
        
        # STEP 2: Entity not in database, try translating the Wikipedia URL
        logging.debug(f"Entity '{entity}' not found in database, attempting translation via URL")
        
        wiki_info = self.extract_wiki_info(wiki_url)
        
        if not wiki_info:
            logging.debug(f"Could not parse URL for {entity}: {wiki_url}")
            return None
        
        lang_code, title = wiki_info
        
        # Get English title via API if non-English (will raise ThrottlingError if throttled)
        if lang_code != 'en':
            en_title = self.get_english_title_from_api(lang_code, title)
            if not en_title:
                logging.debug(f"No English equivalent found for {entity}: {lang_code}:{title}")
                return None
            title = en_title
        
        # STEP 3: Look up translated title in database
        db_result = self.get_article_from_db(title)
        
        if not db_result:
            logging.debug(f"Translated title not found in database: {title}")
            return None
        
        page_id, en_url = db_result
        
        return {
            'entity': entity,
            'wikipedia_title': title,
            'wikipedia_id': page_id,
            'wikipedia_url': en_url,
            'earliest_date': earliest_date,
            'latest_date': latest_date,
            'original_url': wiki_url if wiki_url != en_url else None
        }
    
    def normalize_csv(self, input_file: str, output_file: str, skip_missing: bool = False, resume: bool = False, mode: str = 'yago') -> Tuple[int, int, int]:
        """
        Normalize CSV file from yago_parser.py or wikidata_parser.py
        
        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path
            skip_missing: If True, skip entries not found; if False, keep original URLs
            resume: If True, resume from existing output file
            mode: Input format mode - 'yago' or 'wikidata'
                  - yago: Entity,Wikipedia_URL,Earliest_Date,Latest_Date
                  - wikidata: Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date
            
        Returns:
            Tuple of (total_entries, normalized_entries, skipped_entries)
        """
        total = 0
        normalized = 0
        skipped = 0
        
        # Count total lines in input file
        logging.info("Scanning input file to count total entries...")
        with _open_input(input_file) as f:
            total_entries = sum(1 for _ in f)
        logging.info(f"Total entries in input file: {total_entries:,}")
        
        # Check for existing output if resuming
        skip_lines = 0
        if resume:
            skip_lines = self.count_output_lines(output_file)
            if skip_lines > 0:
                logging.info(f"Resuming: skipping first {skip_lines:,} already processed entries")
        
        # Track timing for ETA
        start_time = datetime.now()
        
        # Determine file mode
        file_mode = 'a' if (resume and skip_lines > 0) else 'w'
        
        with _open_input(input_file) as infile:
            with open(output_file, file_mode, newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                # Write header only if new file
                if file_mode == 'w':
                    writer.writerow(['Entity', 'Wikipedia_Title', 'Wikipedia_ID', 'Wikipedia_URL', 
                                   'Earliest_Date', 'Latest_Date', 'Original_URL'])
                
                # Determine expected columns based on mode
                if mode == 'wikidata':
                    min_cols = 5  # Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date
                    logging.info("Using Wikidata input format")
                else:
                    min_cols = 4  # Entity,Wikipedia_URL,Earliest_Date,Latest_Date
                    logging.info("Using YAGO input format")
                
                # Set up tqdm progress bar (always visible)
                pbar = tqdm(
                    total=total_entries,
                    initial=skip_lines,
                    unit=' entries',
                    desc='  Normalizing',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                )
                
                header_skipped = False
                for row in reader:
                    # Skip header row (first row with column names)
                    if not header_skipped:
                        header_skipped = True
                        # Check if this looks like a header row
                        if row and (row[0].startswith('Entity') or row[0] == 'Entity_ID'):
                            continue
                    
                    # Skip already processed lines
                    if total < skip_lines:
                        total += 1
                        continue
                    
                    total += 1
                    pbar.update(1)
                    
                    if len(row) < min_cols:
                        logging.warning(f"Skipping malformed row (expected {min_cols} columns, got {len(row)}): {row}")
                        skipped += 1
                        continue
                    
                    # Parse columns based on mode
                    if mode == 'wikidata':
                        # Wikidata format: Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date
                        entity_id = row[0]  # e.g., Q1044
                        entity = row[1]      # e.g., Sierra_Leone
                        wiki_url = row[2]    # e.g., https://en.wikipedia.org/wiki/Sierra_Leone
                        earliest_date = row[3]
                        latest_date = row[4]
                    else:
                        # YAGO format: Entity,Wikipedia_URL,Earliest_Date,Latest_Date
                        entity = row[0]
                        wiki_url = row[1]
                        earliest_date = row[2]
                        latest_date = row[3]
                    
                    # Normalize entry
                    normalized_entry = self.normalize_entry(entity, wiki_url, earliest_date, latest_date)
                    
                    if normalized_entry:
                        writer.writerow([
                            normalized_entry['entity'],
                            normalized_entry['wikipedia_title'],
                            normalized_entry['wikipedia_id'],
                            normalized_entry['wikipedia_url'],
                            normalized_entry['earliest_date'],
                            normalized_entry['latest_date'],
                            normalized_entry['original_url'] or ''
                        ])
                        normalized += 1
                    elif not skip_missing:
                        # Keep original entry with empty fields
                        writer.writerow([entity, '', '', wiki_url, earliest_date, latest_date, wiki_url])
                        skipped += 1
                    else:
                        skipped += 1
                    
                    # Update progress bar postfix with live counters
                    if total % 500 == 0:
                        pbar.set_postfix_str(
                            f'ok={normalized:,} skip={skipped:,} api={self.api_call_count:,}',
                            refresh=False,
                        )
                    
                    # Flush output periodically
                    if total % 1000 == 0:
                        outfile.flush()
                
                pbar.set_postfix_str(
                    f'ok={normalized:,} skip={skipped:,} api={self.api_call_count:,}',
                )
                pbar.close()
        
        return (total, normalized, skipped)
    
    def normalize_json(self, input_file: str, output_file: str, skip_missing: bool = False, resume: bool = False, mode: str = 'yago') -> Tuple[int, int, int]:
        """
        Normalize JSON file from yago_parser.py or wikidata_parser.py
        
        Args:
            input_file: Input JSON file path
            output_file: Output JSON file path
            skip_missing: If True, skip entries not found; if False, keep original URLs
            resume: If True, resume from existing output file
            mode: Input format mode - 'yago' or 'wikidata'
                  - yago: {entity, wikipedia_url, earliest_date, latest_date}
                  - wikidata: {entity_id, entity, wikipedia_url, earliest_date, latest_date}
            
        Returns:
            Tuple of (total_entries, normalized_entries, skipped_entries)
        """
        total = 0
        normalized = 0
        skipped = 0
        
        # Track timing for ETA
        start_time = datetime.now()
        
        # Read input JSON
        with _open_input(input_file) as infile:
            data = json.load(infile)
        
        total_entries = len(data)
        
        # Load existing data if resuming
        normalized_data = []
        skip_entries = 0
        if resume:
            try:
                with open(output_file, 'r', encoding='utf-8') as outfile:
                    normalized_data = json.load(outfile)
                    skip_entries = len(normalized_data)
                    if skip_entries > 0:
                        logging.info(f"Resuming: loaded {skip_entries} already processed entries")
            except FileNotFoundError:
                pass
        
        pbar = tqdm(
            total=total_entries,
            initial=skip_entries,
            unit=' entries',
            desc='  Normalizing',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        )
        
        for idx, entry in enumerate(data):
            # Skip already processed entries
            if idx < skip_entries:
                continue
            
            total += 1
            pbar.update(1)
            
            # Parse fields based on mode
            if mode == 'wikidata':
                # Wikidata format includes entity_id
                entity_id = entry.get('entity_id', entry.get('Entity_ID', ''))
                entity = entry.get('entity', entry.get('Entity', ''))
                wiki_url = entry.get('wikipedia_url', entry.get('Wikipedia_URL', ''))
                earliest_date = entry.get('earliest_date', entry.get('Earliest_Date', '0'))
                latest_date = entry.get('latest_date', entry.get('Latest_Date', '0'))
            else:
                # YAGO format
                entity = entry.get('entity', entry.get('Entity', ''))
                wiki_url = entry.get('wikipedia_url', entry.get('Wikipedia_URL', ''))
                earliest_date = entry.get('earliest_date', entry.get('Earliest_Date', '0'))
                latest_date = entry.get('latest_date', entry.get('Latest_Date', '0'))
            
            # Normalize entry
            normalized_entry = self.normalize_entry(entity, wiki_url, earliest_date, latest_date)
            
            if normalized_entry:
                normalized_data.append(normalized_entry)
                normalized += 1
            elif not skip_missing:
                # Keep original entry
                normalized_data.append({
                    'entity': entity,
                    'wikipedia_title': '',
                    'wikipedia_id': 0,
                    'wikipedia_url': wiki_url,
                    'earliest_date': earliest_date,
                    'latest_date': latest_date,
                    'original_url': wiki_url
                })
                skipped += 1
            else:
                skipped += 1
            
            # Update progress bar postfix
            if total % 500 == 0:
                pbar.set_postfix_str(
                    f'ok={normalized:,} skip={skipped:,} api={self.api_call_count:,}',
                    refresh=False,
                )
            
            # Periodically save progress
            if (total - skip_entries) % 1000 == 0:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    json.dump(normalized_data, outfile, indent=2, ensure_ascii=False)
        
        pbar.set_postfix_str(
            f'ok={normalized:,} skip={skipped:,} api={self.api_call_count:,}',
        )
        pbar.close()
        
        return (total, normalized, skipped)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Normalize YAGO parser output to English Wikipedia with page IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize CSV file
  python normalize_yago_output.py input.csv --output normalized.csv
  
  # Normalize JSON file
  python normalize_yago_output.py input.json --output normalized.json --format json
  
  # Skip entries not found in database
  python normalize_yago_output.py input.csv --output normalized.csv --skip-missing
  
  # Verbose logging
  python normalize_yago_output.py input.csv --output normalized.csv --verbose
        """
    )
    
    parser.add_argument('input_file', nargs='?', default=None,
                        help='Input file from yago_parser.py (CSV or JSON). '
                             'Default: $WIKI_DATA/yago/yago-facts.csv.zst (falls back to .csv)')
    parser.add_argument('--output', '-o', default=None, help='Output file path '
                        '(default: <input_dir>/<input_stem>-normalized.csv)')
    parser.add_argument('--format', '-f', choices=['csv', 'json'], 
                       help='Output format (auto-detected from file extension if not specified)')
    parser.add_argument('--skip-missing', action='store_true',
                       help='Skip entries not found in database (default: keep with original URLs)')
    parser.add_argument('--resume', '-r', action='store_true',
                       help='Resume from existing output file (skip already processed entries)')
    parser.add_argument('--mode', '-m', choices=['yago', 'wikidata'], default='yago',
                       help='Input format mode: "yago" for YAGO parser output (Entity,Wikipedia_URL,Earliest_Date,Latest_Date), '
                            '"wikidata" for Wikidata parser output (Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date). Default: yago')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--api-delay', type=float, default=0.1, 
                       help='Delay in seconds between Wikipedia API calls (default: 0.1). Increase if throttled.')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing output file without prompting (default: skip if output exists)')
    parser.add_argument('--no-compress', action='store_true',
                       help='Skip compression stage (keep plain CSV output, do not create .zst)')
    parser.add_argument('--no-reclaim', action='store_true',
                       help='Skip reclamation stage (keep plain CSV after compression)')
    parser.add_argument('--db-host', default=None, help='PostgreSQL host (default: $PG_HOST or localhost)')
    parser.add_argument('--db-name', default=None, help='Database name (default: $PG_DATABASE or wikidb)')
    parser.add_argument('--db-user', default=None, help='Database user (default: $PG_USER or wiki)')
    parser.add_argument('--db-password', default=None, help='Database password (default: $PG_PASSWORD or wikipass)')
    
    args = parser.parse_args()
    
    # Set logging level
    # Default is WARNING (quiet) — progress bar provides feedback.
    # --verbose enables DEBUG so INFO/DEBUG messages are shown alongside the bar.
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Suppress noisy urllib3/requests connection-level debug logs
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

    # Resolve default input file
    if args.input_file is None:
        wiki_data = os.environ.get('WIKI_DATA', '/mnt/data/wikipedia')
        zst_path = os.path.join(wiki_data, 'yago', 'yago-facts.csv.zst')
        csv_path = os.path.join(wiki_data, 'yago', 'yago-facts.csv')
        if os.path.exists(zst_path):
            args.input_file = zst_path
        elif os.path.exists(csv_path):
            args.input_file = csv_path
        else:
            logging.error(f'No input file specified and neither {zst_path} nor {csv_path} exists')
            sys.exit(1)
        logging.info(f'Using default input: {args.input_file}')

    # Resolve default output file
    if args.output is None:
        inp = Path(args.input_file)
        # Strip .zst suffix if present so stem is e.g. "yago-facts"
        stem = inp.stem if inp.suffix != '.zst' else Path(inp.stem).stem
        args.output = str(inp.parent / f'{stem}-normalized.csv')
        logging.info(f'Using default output: {args.output}')

    # Determine format
    output_format = args.format
    if not output_format:
        ext = Path(args.output).suffix.lower()
        if ext == '.json':
            output_format = 'json'
        elif ext == '.csv':
            output_format = 'csv'
        else:
            logging.error("Could not determine output format. Please specify --format")
            sys.exit(1)
    
    # Check if output file already exists (unless --force or --resume)
    if Path(args.output).exists() and not args.force and not args.resume:
        logging.error(f"Output file already exists: {args.output}")
        logging.error("Use --resume to continue from where it left off, or --force to overwrite")
        sys.exit(1)

    # Setup database config — CLI args override env vars which override defaults
    db_config = dict(DB_CONFIG)
    if args.db_host:
        db_config['host'] = args.db_host
    if args.db_name:
        db_config['database'] = args.db_name
    if args.db_user:
        db_config['user'] = args.db_user
    if args.db_password:
        db_config['password'] = args.db_password
    
    # Create normalizer
    normalizer = WikipediaNormalizer(db_config, api_delay=args.api_delay, verbose=args.verbose)
    
    # Connect to database
    if not normalizer.connect_db():
        logging.error("Failed to connect to database")
        sys.exit(1)
    
    try:
        logging.info(f"Normalizing {args.input_file} -> {args.output}")
        logging.info(f"API delay: {args.api_delay} seconds between calls")
        
        # Process file
        if output_format == 'csv':
            total, normalized, skipped = normalizer.normalize_csv(
                args.input_file, args.output, args.skip_missing, args.resume, args.mode
            )
        else:  # json
            total, normalized, skipped = normalizer.normalize_json(
                args.input_file, args.output, args.skip_missing, args.resume, args.mode
            )
        
        # Summary
        logging.info(f"\nNormalization complete!")
        logging.info(f"  Total entries: {total:,}")
        logging.info(f"  Normalized: {normalized:,} ({100*normalized/total:.1f}%)")
        logging.info(f"  Skipped/kept original: {skipped:,} ({100*skipped/total:.1f}%)")
        logging.info(f"  API calls made: {normalizer.api_call_count:,}")
        logging.info(f"  API translations successful: {normalizer.api_success_count:,}")
        logging.info(f"  API translations not found: {normalizer.api_notfound_count:,}")
        logging.info(f"  Output saved to: {args.output}")

        # --- Compression stage ------------------------------------------------
        output_path = Path(args.output)
        if not args.no_compress and output_format == 'csv' and output_path.exists():
            zst_path = output_path.with_suffix(output_path.suffix + '.zst')
            _compress_file(output_path, zst_path, verbose=args.verbose)
            logging.info(f"  Compressed output: {zst_path}")

            # --- Reclaim stage ------------------------------------------------
            if not args.no_reclaim:
                _reclaim_file(output_path)

    except ThrottlingError as e:
        logging.error(f"\nScript stopped due to API throttling: {e}")
        logging.error(f"To resume: python {sys.argv[0]} {args.input_file} --output {args.output} --resume --api-delay {args.api_delay * 2}")
        sys.exit(2)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during normalization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        normalizer.close_db()


if __name__ == '__main__':
    main()
