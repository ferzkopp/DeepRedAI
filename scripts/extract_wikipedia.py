#!/usr/bin/env python3
"""
Extract Wikipedia articles from XML dump to clean JSON format using parallel processing.

Features:
- Uses mediawiki-dump for XML parsing
- Uses mwparserfromhell for wikitext parsing
- Cleans up formatting artifacts and punctuation issues
- Parallelizes article cleaning using multiprocessing
- Pre-compiles regex patterns for performance

Prerequisites:
- mediawiki-dump: pip install mediawiki-dump
- mwparserfromhell: pip install mwparserfromhell
- Python 3.8+ recommended for multiprocessing stability

Usage:
    python extract_wikipedia.py
"""

import io
import bz2
import json
import logging
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from mediawiki_dump.dumps import IteratorDump
from mediawiki_dump.reader import DumpReaderArticles
import mwparserfromhell
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_bz2_content(file_name):
    """Buffered reader for bz2 file."""
    with bz2.open(file_name, mode="rb") as raw:
        with io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8") as fp:
            for line in fp:
                yield line.rstrip('\n')

# Precompiled regex patterns
STAR_SEPARATOR_PATTERN = re.compile(r'(\*\s*){3,}')
HEADING_PATTERN = re.compile(r'(={2,})([^=]+?)\1')
HEADING_THUMBNAIL_PATTERN = re.compile(r'(#{2,}\s*[^#\n]+?)thumbnail\s*')
BULLETED_LIST_PATTERN = re.compile(r'(?<!\w)\*([^*\n=]+?)(?=\*[^*]|=|\n|$)')
# URL pattern: matches http/https URLs that may be missing surrounding spaces
URL_PATTERN = re.compile(r'(https?://[^\s<>\[\]"\']+)', re.IGNORECASE)

# Precompiled regex substitutions as (pattern, replacement) tuples
REGEX_SUBSTITUTIONS = [
    (re.compile(r'\bthumb\|.*?\|'), ''),
    (re.compile(r'\bclass=.*?\|'), ''),
    (re.compile(r'\b(?:upright|left|right|center)=?.*?\|'), ''),
    (re.compile(r'\b\d+px\|'), ''),
    (re.compile(r'\|+'), ' '),
    (re.compile(r'\(\s*\)'), ''),               # Empty parentheses ()
    (re.compile(r'//'), ' '),                   # Double slashes //
    (re.compile(r'\*\*\s'), '* '),              # Double asterisks **
    (re.compile(r'\.\*\s'), '. * '),            # Period-asterisk .* to ". * "
    (re.compile(r'(, ?){2,}'), ', '),
    (re.compile(r'(; ?){2,}'), '; '),
    (re.compile(r'\.;'), '.'),
    (re.compile(r';\s*\.'), '.'),
    (re.compile(r'\s{2,}'), ' '),
    (re.compile(r'\n{3,}'), '\n\n'),
    (re.compile(r'\s+\.\s+'), '. '),
    (re.compile(r'\s+,(\s*)'), r',\1'),
    (re.compile(r'\(\s*,\s*\)'), ''),
]


def convert_headings_to_markdown(text):
    """
    Convert wiki-style headings (== Heading ==) to markdown format (## Heading).
    Handles multiple consecutive headings and removes 'thumbnail' artifacts.
    """
    def heading_replacer(match):
        equals_count = len(match.group(1))
        heading_text = match.group(2).strip()
        # Map wiki heading levels to markdown (== is h2, === is h3, etc.)
        markdown_level = min(equals_count, 6)  # Cap at h6
        hashes = '#' * markdown_level
        return f' {hashes} {heading_text} '
    
    # Convert wiki headings to markdown
    text = HEADING_PATTERN.sub(heading_replacer, text)
    # Remove 'thumbnail' artifacts from headings
    text = HEADING_THUMBNAIL_PATTERN.sub(r'\1 ', text)
    return text


def parse_table_to_text(table_content):
    """
    Parse a wiki-style table and convert to a cleaner text format.
    Extracts meaningful content while removing HTML tags and formatting.
    """
    # Remove HTML tags and attributes
    content = re.sub(r'<[^>]+>', '', table_content)
    # Remove style, class, width, and other common HTML/wiki attributes
    # Match attribute="value" or attribute='value' or attribute=value patterns
    content = re.sub(r'\b(?:style|class|width|height|border|cellpadding|cellspacing|bgcolor|valign|align|scope|colspan|rowspan|id|name)\s*=\s*"[^"]*"', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\b(?:style|class|width|height|border|cellpadding|cellspacing|bgcolor|valign|align|scope|colspan|rowspan|id|name)\s*=\s*\'[^\']*\'', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\b(?:style|class|width|height|border|cellpadding|cellspacing|bgcolor|valign|align|scope|colspan|rowspan|id|name)\s*=\s*\S+', '', content, flags=re.IGNORECASE)
    # Remove wikitable class references
    content = re.sub(r'\bwikitable\b', '', content, flags=re.IGNORECASE)
    
    # Split into rows (marked by -! or -)
    rows = re.split(r'\s*-!?\s*', content)
    
    parsed_rows = []
    for row in rows:
        if not row.strip():
            continue
        # Split cells by ! (header separator) or common delimiters
        # First try !! for header cells, then single !
        cells = re.split(r'\s*!!\s*|\s*!\s*|\s*\|\s*', row)
        # Clean each cell
        cleaned_cells = []
        for cell in cells:
            cell = cell.strip()
            # Remove remaining formatting artifacts
            cell = re.sub(r'^[!"#\-\|]+', '', cell)
            cell = re.sub(r'[!"#\-\|]+$', '', cell)
            cell = cell.strip()
            if cell and cell not in ['!', '-', '']:
                cleaned_cells.append(cell)
        if cleaned_cells:
            parsed_rows.append(', '.join(cleaned_cells))
    
    if parsed_rows:
        return ' [Table: ' + '; '.join(parsed_rows[:10]) + ('...' if len(parsed_rows) > 10 else '') + '] '
    return ' '


def convert_tables(text):
    """
    Find and convert all wiki-style tables in the text.
    
    Tables are detected by matching content between curly braces that contains
    table indicators like ! (cell separator), rowspan/colspan, or - (row separator).
    """
    # Pattern matches { ... } where content contains table indicators
    # Uses a function to validate the match contains table-like content
    def is_table_content(content):
        """Check if content between braces looks like a table."""
        # Must contain cell separators (!) or row separators with content
        has_cell_sep = '!' in content
        has_row_sep = ' - ' in content or content.count('-') >= 3
        has_html_attrs = 'rowspan' in content.lower() or 'colspan' in content.lower()
        has_table_markers = content.strip().startswith(('+', '-'))
        # Require multiple indicators to avoid false positives
        return has_table_markers or has_html_attrs or (has_cell_sep and has_row_sep)
    
    # Find all { ... } blocks and check if they're tables
    result = []
    i = 0
    while i < len(text):
        # Find opening brace
        brace_start = text.find('{', i)
        if brace_start == -1:
            result.append(text[i:])
            break
        
        # Add text before the brace
        result.append(text[i:brace_start])
        
        # Find matching closing brace
        brace_end = text.find('}', brace_start)
        if brace_end == -1:
            # No closing brace, keep rest of text as-is
            result.append(text[brace_start:])
            break
        
        # Extract content between braces
        content = text[brace_start + 1:brace_end]
        
        # Check if it's a table
        if is_table_content(content):
            result.append(parse_table_to_text(content))
        else:
            # Not a table, keep original (including braces)
            result.append(text[brace_start:brace_end + 1])
        
        i = brace_end + 1
    
    return ''.join(result)


def format_categories(text):
    """
    Convert Category:X Category:Y Category:Z format to 
    'Categories: X, Y, Z' format.
    Categories typically appear at the end of articles without spaces between them.
    """
    # Use iterative approach to find and collect consecutive Category: entries
    # This is more reliable than complex regex for this pattern
    
    def find_and_replace_categories(text):
        result = []
        i = 0
        text_lower = text.lower()
        
        while i < len(text):
            # Look for "Category:" (case-insensitive)
            cat_start = text_lower.find('category:', i)
            if cat_start == -1:
                result.append(text[i:])
                break
            
            # Add text before this category
            result.append(text[i:cat_start])
            
            # Collect all consecutive categories
            categories = []
            pos = cat_start
            
            while pos < len(text) and text_lower[pos:pos+9] == 'category:':
                # Find the end of this category (next Category: or end of string)
                cat_name_start = pos + 9
                next_cat = text_lower.find('category:', cat_name_start)
                
                if next_cat == -1:
                    # Last category - take until end or whitespace/newline
                    cat_name = text[cat_name_start:].strip()
                    categories.append(cat_name)
                    pos = len(text)
                else:
                    cat_name = text[cat_name_start:next_cat].strip()
                    categories.append(cat_name)
                    pos = next_cat
            
            # Format categories
            if len(categories) >= 2:
                result.append(' Categories: ' + ', '.join(categories) + ' ')
            else:
                # Single category, keep original
                for cat in categories:
                    result.append(f'Category:{cat}')
            
            i = pos
        
        return ''.join(result)
    
    return find_and_replace_categories(text)


def convert_bulleted_lists(text):
    """
    Convert wiki-style bulleted lists (*item1*item2*item3) to 
    comma-separated format (item1, item2, item3).
    """
    # Find sequences of *item patterns
    # Match patterns like *Name1*Name2*Name3
    bulleted_pattern = re.compile(r'(?:^|\s)\*([^*\n]{2,}?)(?=\*[^*\s]|\s*=|\s*\n|$)', re.MULTILINE)
    
    # Find all bulleted sequences and group them
    def process_bulleted_sequence(text):
        # Look for consecutive bulleted items
        sequence_pattern = re.compile(r'(\*[^*\n=]{2,}(?:\*[^*\n=]{2,}){2,})', re.MULTILINE)
        
        def sequence_replacer(match):
            sequence = match.group(1)
            # Split by * and clean
            items = [item.strip() for item in sequence.split('*') if item.strip()]
            if len(items) >= 2:
                return ' ' + ', '.join(items) + ' '
            return match.group(0)
        
        return sequence_pattern.sub(sequence_replacer, text)
    
    return process_bulleted_sequence(text)


def remove_star_separators(text):
    """Remove decorative star separator sequences like '* * * * *'."""
    return STAR_SEPARATOR_PATTERN.sub(' ', text)


def format_inline_urls(text):
    """
    Find inline URLs and convert them to markdown format with proper spacing.
    
    Example:
        'text1994http://example.com/page and more'
        becomes
        'text1994 [link](http://example.com/page) and more'
    """
    def url_replacer(match):
        url = match.group(1)
        # Clean up URL - remove trailing punctuation that's likely not part of URL
        while url and url[-1] in '.,;:!?)\'\"':
            url = url[:-1]
        return f' [link]({url}) '
    
    return URL_PATTERN.sub(url_replacer, text)


def clean_wikitext(wikitext):
    """
    Clean wikitext to plain text, removing templates, tags, and formatting artifacts.
    
    Processing steps:
    1. Parse wikitext and remove non-content sections (references, external links, etc.)
    2. Strip wiki markup code
    3. Apply structural transformations (tables, headings, categories, lists)
    4. Clean up punctuation and whitespace
    """
    try:
        # Parse and filter sections
        wikicode = mwparserfromhell.parse(wikitext)
        for section in wikicode.get_sections():
            heading = section.filter_headings()
            if heading:
                title = str(heading[0].title).strip().lower()
                if title in ['references', 'bibliography', 'external links', 
                             'see also', 'further reading', 'notes']:
                    wikicode.remove(section)
        
        # Strip wiki markup
        text = wikicode.strip_code()
        
        # Apply structural transformations
        text = remove_star_separators(text)
        text = convert_tables(text)
        text = convert_headings_to_markdown(text)
        text = format_categories(text)
        text = convert_bulleted_lists(text)
        text = format_inline_urls(text)
        
        # Apply regex substitutions for cleanup
        for pattern, repl in REGEX_SUBSTITUTIONS:
            text = pattern.sub(repl, text)
        
        return text.strip()
    except Exception as e:
        logging.warning(f"Error cleaning wikitext: {e}")
        return ""

def process_entry(entry):
    """Process a single article entry."""
    if entry.content.strip().lower().startswith('#redirect') or ':' in entry.title:
        return None
    clean_text = clean_wikitext(entry.content)
    if len(clean_text) < 100:
        return None
    return {
        'id': str(entry.page_id),
        'title': entry.title,
        'url': f'https://en.wikipedia.org/wiki?curid={entry.page_id}',
        'text': clean_text
    }

def backup_file_if_exists(output_file, backup_path):
    """Move existing file to backup directory before overwriting.
    
    Args:
        output_file: Path to the file that will be written
        backup_path: Directory to move existing file to
    
    Returns:
        Path to backup file if backup was created, None otherwise
    """
    if output_file.exists():
        backup_path.mkdir(parents=True, exist_ok=True)
        backup_file = backup_path / output_file.name
        shutil.move(str(output_file), str(backup_file))
        logging.info(f"Backed up existing file to: {backup_file}")
        return backup_file
    return None


def extract_articles(dump_file, output_dir, batch_size=1000, backup_dir=None):
    """Extract articles from Wikipedia dump to JSON files.
    
    Args:
        dump_file: Path to the Wikipedia XML dump (bz2 compressed)
        output_dir: Directory to write extracted JSON files
        batch_size: Number of articles per batch file
        backup_dir: If specified, existing files are moved here before writing new ones.
                    If None, defaults to output_dir + '_backup'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up backup directory
    if backup_dir is None:
        backup_path = Path(str(output_dir) + '_backup')
    else:
        backup_path = Path(backup_dir)

    logging.info(f"Reading bz2 compressed dump from: {dump_file}")
    dump = IteratorDump(iterator=get_bz2_content(dump_file))
    reader = DumpReaderArticles()

    batch = []
    batch_num = 0
    total_articles = 0

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = []
        for entry in reader.read(dump):
            futures.append(executor.submit(process_entry, entry))

            # Process completed futures as they accumulate
            while len(futures) >= 100:
                for future in as_completed(futures[:100]):
                    futures.remove(future)
                    result = future.result()
                    if result:
                        batch.append(result)

                    if len(batch) >= batch_size:
                        output_file = output_path / f'wikipedia_batch_{batch_num:05d}.json'
                        backup_file_if_exists(output_file, backup_path)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for article in batch:
                                f.write(json.dumps(article, ensure_ascii=False) + '\n')
                        logging.info(f"Wrote batch {batch_num} ({len(batch)} articles) - Total: {total_articles + len(batch)}")
                        total_articles += len(batch)
                        batch = []
                        batch_num += 1

        # Final cleanup
        for future in as_completed(futures):
            result = future.result()
            if result:
                batch.append(result)
            if len(batch) >= batch_size:
                output_file = output_path / f'wikipedia_batch_{batch_num:05d}.json'
                backup_file_if_exists(output_file, backup_path)
                with open(output_file, 'w', encoding='utf-8') as f:
                    for article in batch:
                        f.write(json.dumps(article, ensure_ascii=False) + '\n')
                logging.info(f"Wrote batch {batch_num} ({len(batch)} articles) - Total: {total_articles + len(batch)}")
                total_articles += len(batch)
                batch = []
                batch_num += 1

        if batch:
            output_file = output_path / f'wikipedia_batch_{batch_num:05d}.json'
            backup_file_if_exists(output_file, backup_path)
            with open(output_file, 'w', encoding='utf-8') as f:
                for article in batch:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
            logging.info(f"Wrote final batch {batch_num} ({len(batch)} articles)")
            total_articles += len(batch)

    logging.info(f"Extraction complete! Total articles: {total_articles}")

if __name__ == '__main__':
    extract_articles(
        dump_file='/srv/wikipedia/dumps/enwiki-latest-pages-articles.xml.bz2',
        output_dir='/srv/wikipedia/extracted',
        batch_size=1000
    )
