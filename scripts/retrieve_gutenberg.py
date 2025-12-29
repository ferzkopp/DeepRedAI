#!/usr/bin/env python3
"""
Retrieve and filter Project Gutenberg texts for theme fine-tuning.

This script downloads books from Project Gutenberg based on predefined
priority works and subject filters. It saves the corpus in JSONL format.

All works are filtered to ensure publication before 1969 (pre-moon landing)
to maintain temporal consistency with the Soviet utopia aesthetic.

Supports two retrieval modes:
1. gutenberg library (requires Berkeley DB) - for subject search
2. Direct HTTP download (primary method for individual works)

Environment Variables:
    GUTENBERG_DATA: Base directory for output (default: output/gutenberg_corpus)
    GUTENBERG_MIRROR: Mirror URL for gutenberg library (if using subject search)
                      See https://www.gutenberg.org/MIRRORS.ALL for options
"""

import os
import json
import requests
import re
import warnings
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote

# Suppress rdflib pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

# Temporal cutoff: All works must be published before 1969 (moon landing year)
TEMPORAL_CUTOFF_YEAR = 1969

# Set mirror from environment variable before importing gutenberg
# This avoids the default mirror which may be down
GUTENBERG_MIRROR = os.environ.get('GUTENBERG_MIRROR', 'https://www.gutenberg.org')
if 'GUTENBERG_MIRROR' not in os.environ:
    os.environ['GUTENBERG_MIRROR'] = GUTENBERG_MIRROR

# Try to import gutenberg library, but don't fail if unavailable
# Note: The library is only needed for subject-based searching
try:
    from gutenberg.acquire import load_etext
    from gutenberg.cleanup import strip_headers
    from gutenberg.query import get_metadata
    GUTENBERG_LIB_AVAILABLE = True
except ImportError:
    GUTENBERG_LIB_AVAILABLE = False
    print("Note: gutenberg library not available. Subject search disabled, but HTTP retrieval works fine.")


class GutenbergRetriever:
    """Retrieve and filter Project Gutenberg texts for theme fine-tuning."""
    
    # Target Gutenberg IDs for priority works
    PRIORITY_WORKS = {
        # Utopian Fiction
        624: "Looking Backward",      # Edward Bellamy
        3261: "A Modern Utopia",      # H.G. Wells
        3362: "News from Nowhere",    # William Morris
        1164: "The Iron Heel",        # Jack London
        32: "Herland",                # Charlotte Perkins Gilman
        61963: "We",                  # Yevgeny Zamyatin
        
        # Russian Literature (English translations)
        2554: "Crime and Punishment",  # Dostoevsky
        28054: "The Brothers Karamazov", # Dostoevsky
        600: "Notes from the Underground", # Dostoevsky
        2638: "The Idiot",             # Dostoevsky
        8117: "The Possessed",         # Dostoevsky
        1399: "Anna Karenina",         # Tolstoy
        2600: "War and Peace",         # Tolstoy
        3783: "Mother",                # Maxim Gorky
        47935: "Fathers and Sons",     # Ivan Turgenev
        1081: "Dead Souls",            # Nikolai Gogol
        7986: "The Cherry Orchard",    # Anton Chekhov
        1756: "Uncle Vanya",           # Anton Chekhov
        55351: "Three Sisters",        # Anton Chekhov
        1754: "The Seagull",           # Anton Chekhov
        
        # Early Science Fiction
        103: "From the Earth to the Moon",  # Jules Verne
        164: "20,000 Leagues Under the Sea", # Jules Verne
        165: "Around the Moon",        # Jules Verne
        19513: "Journey to the Center of the Earth", # Jules Verne
        1268: "The Mysterious Island", # Jules Verne
        35: "The Time Machine",        # H.G. Wells
        36: "The War of the Worlds",   # H.G. Wells
        1013: "The First Men in the Moon",  # H.G. Wells
        5230: "The Invisible Man",     # H.G. Wells
        159: "The Island of Doctor Moreau", # H.G. Wells
        62: "A Princess of Mars",      # Edgar Rice Burroughs
        139: "The Lost World",         # Arthur Conan Doyle
        59112: "R.U.R.",               # Karel Čapek
        84: "Frankenstein",            # Mary Shelley
        
        # Political Philosophy
        61: "The Communist Manifesto",  # Marx/Engels
        4341: "Mutual Aid",            # Peter Kropotkin
        
        # Chess
        33870: "Chess Fundamentals",   # Jose Raul Capablanca
    }
    
    # Subject filters for bulk retrieval
    SUBJECT_FILTERS = [
        "Science fiction",
        "Satire",
        "Political satire",
        "Utopias",
        "Dystopias",
        "Ideological extremism",
        "Soviet Union",
        "Russia",
        "Socialism",
        "Communism",
        "Capitalism",
        "Oligarchy",
        "Artificial intelligence",
        "AI governance",
        "AI ethics",
        "AI evolution",
        "Chess",
        "Space flight",
        "Mars (Planet)",
        "Colonies on Mars",
        "Secret societies",
        "Survival",
        "Human evolution",
        "Interplanetary voyages",
        "Space colonization",
        "Moon",
        "Astronauts",
        "Crash survival",
        "Political science",
        "Power struggles",
        "Revolutions",
        "Technological dystopias",
        "Posthumanism",
        "Transhumanism",
        "Future societies",
        "Totalitarianism",
        "Class struggle",
        "Wealth inequality",
        "Corporate power",
        "Terraforming",
        "Extraterrestrial environments",
    ]

    
    # Known pre-1969 author death dates for validation
    # Authors who died before 1969 guarantee pre-1969 works
    KNOWN_PRE1969_AUTHORS = {
        "Wells, H. G. (Herbert George), 1866-1946",
        "Verne, Jules, 1828-1905",
        "Dostoevsky, Fyodor, 1821-1881",
        "Tolstoy, Leo, graf, 1828-1910",
        "Chekhov, Anton Pavlovich, 1860-1904",
        "Bellamy, Edward, 1850-1898",
        "Morris, William, 1834-1896",
        "London, Jack, 1876-1916",
        "Gilman, Charlotte Perkins, 1860-1935",
        "Shelley, Mary Wollstonecraft, 1797-1851",
        "Marx, Karl, 1818-1883",
        "Engels, Friedrich, 1820-1895",
        "Kropotkin, Petr Alekseevich, kniaz, 1842-1921",
        "Burroughs, Edgar Rice, 1875-1950",
        "Doyle, Arthur Conan, Sir, 1859-1930",
        "Gorky, Maksim, 1868-1936",
        "Turgenev, Ivan Sergeevich, 1818-1883",
        "Gogol, Nikolai Vasilevich, 1809-1852",
        "Zamyatin, Evgeny Ivanovich, 1884-1937",
        "Čapek, Karel, 1890-1938",
        "Capablanca, José Raúl, 1888-1942",
    }
    
    def __init__(self, output_dir: str, max_year: int = TEMPORAL_CUTOFF_YEAR, prefer_http: bool = True):
        """Initialize the retriever.
        
        Args:
            output_dir: Directory to save retrieved works
            max_year: Maximum publication year (temporal cutoff)
            prefer_http: If True, use HTTP for individual works (faster, more reliable)
                        Library is still used for subject search if available
        """
        self.output_dir = output_dir
        self.max_year = max_year
        self.prefer_http = prefer_http
        os.makedirs(output_dir, exist_ok=True)
        # Library is available for subject search, but HTTP is preferred for downloads
        self.gutenberg_lib_available = GUTENBERG_LIB_AVAILABLE
        self.retrieved_ids = set()  # Track already retrieved IDs to avoid duplicates
        
        # Load existing IDs from corpus files to avoid re-downloading
        self._load_existing_corpus_ids()
    
    def _load_existing_corpus_ids(self):
        """Load IDs of works already in the corpus files to avoid duplicates."""
        corpus_files = [
            os.path.join(self.output_dir, 'gutenberg_corpus.jsonl'),
            os.path.join(self.output_dir, 'priority_works.jsonl'),
        ]
        
        for corpus_file in corpus_files:
            if os.path.exists(corpus_file):
                try:
                    with open(corpus_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                work = json.loads(line)
                                if 'id' in work:
                                    self.retrieved_ids.add(work['id'])
                    print(f"Loaded {len(self.retrieved_ids)} existing work IDs from {corpus_file}")
                except Exception as e:
                    print(f"Warning: Could not load existing corpus from {corpus_file}: {e}")
    
    def strip_gutenberg_headers(self, text: str) -> str:
        """Remove Project Gutenberg headers and footers from text."""
        # Start marker patterns
        start_patterns = [
            r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*',
            r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*',
            r'START OF THIS PROJECT GUTENBERG EBOOK',
            r'START OF THE PROJECT GUTENBERG EBOOK',
        ]
        
        # End marker patterns
        end_patterns = [
            r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*',
            r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*',
            r'END OF THIS PROJECT GUTENBERG EBOOK',
            r'END OF THE PROJECT GUTENBERG EBOOK',
        ]
        
        # Find start position
        start_pos = 0
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.end()
                break
        
        # Find end position
        end_pos = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                end_pos = match.start()
                break
        
        return text[start_pos:end_pos].strip()
    
    def _is_english_text(self, text: str, header: str = None) -> bool:
        """Check if text is in English.
        
        Uses multiple heuristics:
        1. Check Language field in Gutenberg header
        2. Check ratio of ASCII characters (English text is mostly ASCII)
        3. Check for common English words
        
        Args:
            text: The main text content
            header: Optional header text to check for Language field
            
        Returns:
            True if text appears to be English, False otherwise
        """
        # Check header for explicit language declaration
        if header:
            lang_match = re.search(r'Language:\s*(\w+)', header, re.IGNORECASE)
            if lang_match:
                language = lang_match.group(1).lower()
                if language != 'english':
                    return False
                return True
        
        # Sample the first 5000 characters for analysis
        sample = text[:5000] if len(text) > 5000 else text
        
        # Count ASCII vs non-ASCII characters (excluding whitespace)
        ascii_chars = sum(1 for c in sample if c.isalpha() and ord(c) < 128)
        non_ascii_chars = sum(1 for c in sample if c.isalpha() and ord(c) >= 128)
        total_alpha = ascii_chars + non_ascii_chars
        
        if total_alpha == 0:
            return False
        
        ascii_ratio = ascii_chars / total_alpha
        
        # English text should be >95% ASCII letters
        # Non-English (Chinese, Japanese, Finnish, etc.) will have much lower ratios
        if ascii_ratio < 0.90:
            return False
        
        # Additional check: look for common English words
        common_english = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'it', 'was']
        sample_lower = sample.lower()
        english_word_count = sum(1 for word in common_english if f' {word} ' in sample_lower)
        
        # Should find at least 3 common English words in the sample
        if english_word_count < 3:
            return False
        
        return True

    def retrieve_by_http(self, gutenberg_id: int, title: str) -> dict:
        """Retrieve a work directly via HTTP (fallback method)."""
        try:
            # First, fetch RDF metadata for better date detection
            rdf_metadata = self._fetch_gutenberg_rdf(gutenberg_id)
            
            # Try different URL patterns for text
            urls = [
                f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
                f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
                f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
            ]
            
            text = None
            for url in urls:
                print(f"  Trying {url}")
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    text = response.text
                    break
            
            if not text:
                print(f"  Could not retrieve from any URL")
                return None
            
            # Check if text is in English (skip non-English content)
            header = text[:3000]
            if not self._is_english_text(text, header):
                # Try to extract title for logging
                title_match = re.search(r'Title:\s*(.+)', header)
                display_title = title_match.group(1).strip() if title_match else title
                print(f"  Skipping non-English content: {display_title}")
                return None
            
            # Extract title - prefer RDF, then text header, then provided
            extracted_title = title
            if rdf_metadata and rdf_metadata.get('title'):
                extracted_title = rdf_metadata['title']
            elif title.startswith("Unknown_"):
                title_match = re.search(r'Title:\s*(.+)', text[:2000])
                if title_match:
                    extracted_title = title_match.group(1).strip()
            
            # Try to extract publication year - multiple sources (ordered by reliability)
            pub_year = None
            author_death_year = None
            
            # Get author death year from RDF (useful for validation)
            if rdf_metadata:
                author_death_year = rdf_metadata.get('author_death_year')
            
            # 1. Check text header for explicit publication patterns (most reliable)
            pub_year = self._extract_year_from_text_header(text)
            
            # 2. Check RDF description for time period
            if not pub_year and rdf_metadata:
                pub_year = self._extract_year_from_description(rdf_metadata.get('description', ''))
            
            # 3. Check title for year (lowest priority - only for reference/annual works)
            # This is last because titles like "1984" are misleading (written 1949)
            if not pub_year:
                pub_year = self._extract_year_from_title(extracted_title)
            
            # Strip headers/footers
            cleaned_text = self.strip_gutenberg_headers(text)
            
            # Extract author - prefer RDF, then text header
            author = "Unknown"
            if rdf_metadata and rdf_metadata.get('author'):
                author = rdf_metadata['author']
            else:
                author_match = re.search(r'Author:\s*(.+)', text[:2000])
                if author_match:
                    author = author_match.group(1).strip()
            
            result = {
                'id': gutenberg_id,
                'title': extracted_title,
                'author': author,
                'text': cleaned_text,
                'length': len(cleaned_text),
                'pub_year': pub_year,
                'method': 'http'
            }
            
            # Add author death year if found (useful for temporal validation)
            if author_death_year:
                result['author_death_year'] = author_death_year
            
            return result
        except Exception as e:
            print(f"  Error with HTTP retrieval: {e}")
            return None
    
    def _extract_year_from_title(self, title: str) -> int:
        """Extract publication year from title if present.
        
        Handles titles like:
        - "The 2002 CIA World Factbook"
        - "1984" (Orwell)
        - "The Year 1920"
        """
        if not title:
            return None
        
        # Look for 4-digit year in title
        # Match years that look like publication dates (1800-2100)
        year_matches = re.findall(r'\b(1[89]\d{2}|20\d{2}|21\d{2})\b', title)
        
        for year_str in year_matches:
            year = int(year_str)
            # Filter out years that are likely not publication dates
            # "1984" as a title is fine (it's the book name, published 1949)
            # But "The 2002 CIA World Factbook" means published in 2002
            if year >= 1900 and year <= 2100:
                # Check if this looks like a factual/reference work with year in title
                title_lower = title.lower()
                if any(keyword in title_lower for keyword in [
                    'factbook', 'almanac', 'yearbook', 'annual', 'report',
                    'edition', 'volume', 'survey', 'census', 'statistics'
                ]):
                    return year
                # Check for patterns like "The Year XXXX" or "XXXX Edition"
                if re.search(rf'\b(year|edition|vol\.?|volume)\s*{year}\b', title, re.IGNORECASE):
                    return year
                if re.search(rf'\b{year}\s*(edition|vol\.?|volume|annual|report)\b', title, re.IGNORECASE):
                    return year
        
        return None
    
    def _fetch_gutenberg_rdf(self, gutenberg_id: int) -> dict:
        """Fetch and parse Gutenberg RDF metadata.
        
        Returns dict with:
        - title: Book title
        - author: Author name
        - author_birth_year: Author birth year (if available)
        - author_death_year: Author death year (if available)  
        - issued_date: Gutenberg release date
        - subjects: List of subjects
        - description: Book description
        """
        rdf_url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}.rdf"
        
        try:
            response = requests.get(rdf_url, timeout=15)
            if response.status_code != 200:
                return None
            
            rdf_text = response.text
            metadata = {}
            
            # Extract title
            title_match = re.search(r'<dcterms:title>([^<]+)</dcterms:title>', rdf_text)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
            
            # Extract author name
            author_match = re.search(r'<pgterms:name>([^<]+)</pgterms:name>', rdf_text)
            if author_match:
                metadata['author'] = author_match.group(1).strip()
            
            # Extract author birth/death dates from dedicated RDF tags
            # Format: <pgterms:birthdate rdf:datatype="...">1866</pgterms:birthdate>
            birthdate_match = re.search(r'<pgterms:birthdate[^>]*>(\d{4})</pgterms:birthdate>', rdf_text)
            if birthdate_match:
                metadata['author_birth_year'] = int(birthdate_match.group(1))
            
            deathdate_match = re.search(r'<pgterms:deathdate[^>]*>(\d{4})</pgterms:deathdate>', rdf_text)
            if deathdate_match:
                metadata['author_death_year'] = int(deathdate_match.group(1))
            
            # Fallback: parse birth/death years from author name like "Author, 1866-1946"
            if 'author_death_year' not in metadata and metadata.get('author'):
                author_name = metadata['author']
                years_match = re.search(r'(\d{4})-(\d{4})', author_name)
                if years_match:
                    metadata['author_birth_year'] = int(years_match.group(1))
                    metadata['author_death_year'] = int(years_match.group(2))
            
            # Extract issued date (Gutenberg release, not publication)
            issued_match = re.search(r'<dcterms:issued[^>]*>([^<]+)</dcterms:issued>', rdf_text)
            if issued_match:
                metadata['issued_date'] = issued_match.group(1).strip()
            
            # Extract description (may contain publication info)
            desc_match = re.search(r'<pgterms:marc520>([^<]+)</pgterms:marc520>', rdf_text, re.DOTALL)
            if desc_match:
                metadata['description'] = desc_match.group(1).strip()
            
            # Extract subjects
            subjects = re.findall(r'<rdf:value>([^<]+)</rdf:value>', rdf_text)
            metadata['subjects'] = [s for s in subjects if not s.startswith('http')]
            
            return metadata
            
        except Exception as e:
            print(f"  Warning: Could not fetch RDF metadata: {e}")
            return None
    
    def _extract_year_from_description(self, description: str) -> int:
        """Extract publication year from RDF description/summary."""
        if not description:
            return None
        
        # Look for phrases indicating time period
        patterns = [
            r'produced in the (early |late |mid-)?(\d{4}|\d{2}(?:st|nd|rd|th) century)',
            r'written in (\d{4})',
            r'published in (\d{4})',
            r'from (\d{4})',
            r'(\d{4}) edition',
            r'early (\d{2})(?:st|nd|rd|th) century',
            r'late (\d{2})(?:st|nd|rd|th) century',
            r'mid-(\d{2})(?:st|nd|rd|th) century',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                # Get the last group (the year/century)
                year_str = match.group(match.lastindex)
                if year_str.isdigit():
                    if len(year_str) == 4:
                        return int(year_str)
                    elif len(year_str) == 2:
                        # Century reference - return end of century
                        century = int(year_str)
                        # "21st century" = 2000s, "20th century" = 1900s
                        return (century - 1) * 100 + 99
        
        return None
    
    def _extract_year_from_text_header(self, text: str) -> int:
        """Extract publication year from Gutenberg text header.
        
        Looks for explicit publication patterns in the header only.
        Does NOT check title (that's handled separately with lower priority).
        
        Args:
            text: The full text content
        """
        # Look for common year patterns in the first 3000 characters
        header = text[:3000]
        
        # Pattern: Published year in various formats (most reliable)
        pub_patterns = [
            r'(?:Published|First published|Originally published)[:\s]+(?:\w+\s+)?(?:\d{1,2},?\s+)?(\d{4})',
            r'(?:Written|Written in|Composed)[:\s]+(?:\w+\s+)?(?:\d{1,2},?\s+)?(\d{4})',
            r'Copyright[,\s]+(\d{4})',
            r'\((\d{4})\)\s*$',  # Year in parentheses at end of line
        ]
        
        for pattern in pub_patterns:
            match = re.search(pattern, header, re.IGNORECASE | re.MULTILINE)
            if match:
                year = int(match.group(1))
                if 1400 <= year <= 2100:  # Sanity check
                    return year
        
        return None
    
    def retrieve_by_id(self, gutenberg_id: int, title: str = None, skip_date_check: bool = False) -> dict:
        """Retrieve a specific work by Gutenberg ID.
        
        Uses direct HTTP download (faster and more reliable than gutenberg library mirrors).
        
        Args:
            gutenberg_id: The Project Gutenberg eBook ID
            title: Optional title override
            skip_date_check: If True, skip temporal validation (for known pre-1969 priority works)
        """
        # Check if already retrieved
        if gutenberg_id in self.retrieved_ids:
            print(f"  Skipping {gutenberg_id} (already retrieved)")
            return None
        
        # Use HTTP method (more reliable than library mirrors)
        work = self.retrieve_by_http(gutenberg_id, title or f"Unknown_{gutenberg_id}")
        
        if work and not skip_date_check:
            # Check author against known list
            is_known_author = any(known.lower() in work['author'].lower() 
                                  for known in ["Wells", "Verne", "Dostoevsky", "Tolstoy", 
                                               "Chekhov", "Bellamy", "Morris", "London",
                                               "Gilman", "Shelley", "Marx", "Engels",
                                               "Kropotkin", "Burroughs", "Doyle", "Gorky",
                                               "Turgenev", "Gogol", "Zamyatin", "Čapek",
                                               "Capek", "Capablanca"])
            work['is_known_pre1969_author'] = is_known_author
            
            if not is_known_author and not self._validate_temporal(work):
                print(f"  Skipping {gutenberg_id}: Could not verify pre-{self.max_year} publication")
                return None
        
        if work:
            self.retrieved_ids.add(gutenberg_id)
        return work
    
    def _validate_temporal(self, work: dict) -> bool:
        """Validate that a work was published before the temporal cutoff.
        
        Most Gutenberg works are pre-1969 due to copyright requirements,
        but we add additional validation for certainty.
        """
        title = work.get('title', 'Unknown')
        
        # If we have a publication year, check it
        if work.get('pub_year'):
            pub_year = work['pub_year']
            if pub_year >= self.max_year:
                print(f"  Rejected: '{title}' - publication year {pub_year} >= {self.max_year}")
                return False
            return True
        
        # If author is known to have died before cutoff, work is valid
        if work.get('is_known_pre1969_author'):
            return True
        
        # Check author death year from RDF metadata
        author_death_year = work.get('author_death_year')
        if author_death_year and author_death_year < self.max_year:
            print(f"  Accepted: '{title}' - author died {author_death_year} (before {self.max_year})")
            return True
        
        # For Gutenberg works, most are pre-1928 (US copyright threshold)
        # We can be reasonably confident they're pre-1969
        # But log a warning for manual review
        print(f"  Warning: Could not determine publication year for '{title}' - assuming pre-{self.max_year}")
        return True
    
    def _scrape_subject_ids_from_web(self, subject: str, max_results: int = 100) -> list:
        """Scrape book IDs from Gutenberg website by subject search.
        
        This is a fallback when the gutenberg library cache is not available.
        
        Args:
            subject: The subject to search for
            max_results: Maximum number of IDs to retrieve
            
        Returns:
            List of Gutenberg IDs found
        """
        ids = []
        start_index = 1
        
        while len(ids) < max_results:
            # Search URL with subject filter
            search_url = f"https://www.gutenberg.org/ebooks/search/?query={quote(subject)}&start_index={start_index}"
            
            try:
                response = requests.get(search_url, timeout=30)
                if response.status_code != 200:
                    print(f"  Failed to fetch search page: HTTP {response.status_code}")
                    break
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find book links - they're in format /ebooks/12345
                book_links = soup.find_all('a', href=re.compile(r'^/ebooks/\d+$'))
                
                if not book_links:
                    # No more results
                    break
                
                found_in_page = 0
                for link in book_links:
                    match = re.search(r'/ebooks/(\d+)$', link['href'])
                    if match:
                        book_id = int(match.group(1))
                        if book_id not in ids:
                            ids.append(book_id)
                            found_in_page += 1
                            if len(ids) >= max_results:
                                break
                
                if found_in_page == 0:
                    # No new books found, stop pagination
                    break
                    
                # Move to next page (25 results per page)
                start_index += 25
                
            except Exception as e:
                print(f"  Error scraping web: {e}")
                break
        
        return ids
    
    def retrieve_by_subject(self, subject: str, max_results: int = 50) -> list:
        """Retrieve works matching a subject filter.
        
        Uses the gutenberg library if available, otherwise falls back to web scraping.
        
        Args:
            subject: The Gutenberg subject to search for
            max_results: Maximum number of works to retrieve for this subject
            
        Returns:
            List of work dictionaries
        """
        ids = None
        
        # Try gutenberg library first if available
        if self.gutenberg_lib_available:
            try:
                from gutenberg.query import get_etexts
                
                print(f"\nSearching for subject: '{subject}'")
                # get_etexts returns a frozenset of IDs
                ids = list(get_etexts('subject', subject))
                print(f"  Found {len(ids)} potential works via library")
            except Exception as e:
                print(f"  Library search failed: {e}")
                print(f"  Falling back to web scraping...")
                ids = None
        
        # Fallback to web scraping if library failed or unavailable
        if ids is None:
            print(f"\nSearching for subject: '{subject}' (via web)")
            ids = self._scrape_subject_ids_from_web(subject, max_results=max_results * 2)  # Get extra to account for filtering
            print(f"  Found {len(ids)} potential works via web scraping")
        
        if not ids:
            print(f"  No works found for subject '{subject}'")
            return []
        
        works = []
        count = 0
        skipped = 0
        
        for gutenberg_id in sorted(ids):  # Sort for consistent ordering
            if count >= max_results:
                break
            
            # Skip if already retrieved (from any source: priority works or previous subjects)
            if gutenberg_id in self.retrieved_ids:
                continue
            
            try:
                work = self.retrieve_by_id(gutenberg_id)
                if work:
                    work['subject'] = subject  # Tag with source subject
                    works.append(work)
                    count += 1
                    print(f"  [{count}/{max_results}] {work['title']} by {work['author']} (ID: {gutenberg_id})")
                else:
                    skipped += 1
            except Exception as e:
                print(f"  Failed to retrieve ID {gutenberg_id}: {e}")
                skipped += 1
        
        if skipped > 0:
            print(f"  Skipped {skipped} works (already retrieved or failed temporal check)")
        
        return works
    
    def retrieve_extended_corpus(self, subjects: list = None, max_per_subject: int = 50) -> list:
        """Retrieve extended corpus by searching multiple subjects.
        
        Args:
            subjects: List of subjects to search (defaults to SUBJECT_FILTERS)
            max_per_subject: Maximum works to retrieve per subject
            
        Returns:
            List of all retrieved works
        """
        if subjects is None:
            subjects = self.SUBJECT_FILTERS
        
        all_works = []
        
        for subject in subjects:
            subject_works = self.retrieve_by_subject(subject, max_results=max_per_subject)
            all_works.extend(subject_works)
            print(f"  Subtotal: {len(all_works)} works retrieved so far")
        
        return all_works
    
    def filter_by_date(self, works: list, max_year: int = None) -> list:
        """Filter works published before cutoff year.
        
        Args:
            works: List of work dictionaries
            max_year: Cutoff year (defaults to self.max_year / 1969)
            
        Returns:
            Filtered list of works
        """
        if max_year is None:
            max_year = self.max_year
            
        filtered = []
        for work in works:
            # If we have a publication year, check it
            if work.get('pub_year'):
                if work['pub_year'] < max_year:
                    filtered.append(work)
                else:
                    print(f"  Filtered out: {work['title']} (published {work['pub_year']})")
            # If author is known pre-1969, include
            elif work.get('is_known_pre1969_author'):
                filtered.append(work)
            # Otherwise, assume valid (Gutenberg copyright rules)
            else:
                filtered.append(work)
        
        return filtered
    
    def save_corpus(self, works: list, filename: str, append: bool = True):
        """Save retrieved works to JSONL format.
        
        Args:
            works: List of work dictionaries to save
            filename: Output filename
            append: If True, append to existing file and skip duplicates. If False, overwrite.
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Load existing IDs if appending
        existing_ids = set()
        existing_works = []
        if append and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            work = json.loads(line)
                            if 'id' in work:
                                existing_ids.add(work['id'])
                                existing_works.append(work)
                print(f"Found {len(existing_ids)} existing works in {filename}")
            except Exception as e:
                print(f"Warning: Could not read existing corpus: {e}")
        
        # Filter out duplicates from new works
        new_works = [w for w in works if w and w.get('id') not in existing_ids]
        
        # Combine existing and new works
        all_works = existing_works + new_works
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for work in all_works:
                if work:
                    f.write(json.dumps(work, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(all_works)} total works to {output_path} ({len(new_works)} new, {len(existing_works)} existing)")


def main():
    """Main entry point for the script."""
    import argparse
    
    # Check for GUTENBERG_DATA environment variable
    gutenberg_data = os.environ.get('GUTENBERG_DATA')
    if gutenberg_data:
        default_output = os.path.join(gutenberg_data, 'corpus')
        print(f"Using GUTENBERG_DATA environment variable: {gutenberg_data}")
    else:
        default_output = 'output/gutenberg_corpus'
        print("Warning: GUTENBERG_DATA environment variable not set. Using default output directory.")
    
    parser = argparse.ArgumentParser(
        description="Retrieve Project Gutenberg texts for theme fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrieve only priority works (37 curated books)
  python retrieve_gutenberg.py --priority-only

  # Retrieve priority works + extended corpus by subject
  python retrieve_gutenberg.py

  # Retrieve with custom subjects
  python retrieve_gutenberg.py --subjects "Science fiction,Utopias,Chess"

  # Limit works per subject
  python retrieve_gutenberg.py --max-per-subject 20

  # Use custom temporal cutoff (default is 1969)
  python retrieve_gutenberg.py --max-year 1960

  # Start fresh, deleting any previously retrieved content
  python retrieve_gutenberg.py --reset
        """
    )
    parser.add_argument('--output-dir', default=default_output,
                        help=f'Output directory for retrieved texts (default: {default_output})')
    parser.add_argument('--priority-only', action='store_true',
                        help='Only retrieve priority works (skip subject search)')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated list of subjects to search (default: built-in list)')
    parser.add_argument('--max-per-subject', type=int, default=10,
                        help='Maximum works to retrieve per subject (default: 10)')
    parser.add_argument('--max-year', type=int, default=TEMPORAL_CUTOFF_YEAR,
                        help=f'Maximum publication year (default: {TEMPORAL_CUTOFF_YEAR})')
    parser.add_argument('--reset', action='store_true',
                        help='Delete existing corpus files and start fresh')
    
    args = parser.parse_args()
    
    # Handle reset flag - delete existing corpus files
    if args.reset:
        corpus_files = [
            os.path.join(args.output_dir, 'gutenberg_corpus.jsonl'),
            os.path.join(args.output_dir, 'priority_works.jsonl'),
        ]
        for corpus_file in corpus_files:
            if os.path.exists(corpus_file):
                os.remove(corpus_file)
                print(f"Deleted existing corpus file: {corpus_file}")
        print("Starting fresh corpus build...\n")
    
    # Parse subjects if provided
    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(',')]
    
    retriever = GutenbergRetriever(args.output_dir, max_year=args.max_year)
    
    print(f"\n{'='*60}")
    print(f"Project Gutenberg Retriever")
    print(f"Temporal cutoff: pre-{args.max_year}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Retrieve priority works
    print("Phase 1: Retrieving priority works...")
    print(f"Total priority works: {len(retriever.PRIORITY_WORKS)}\n")
    
    works = []
    success_count = 0
    for gutenberg_id, title in retriever.PRIORITY_WORKS.items():
        print(f"Retrieving: {title} (ID: {gutenberg_id})")
        # Priority works are known to be pre-1969, skip date check
        work = retriever.retrieve_by_id(gutenberg_id, title, skip_date_check=True)
        if work:
            works.append(work)
            success_count += 1
            print(f"  ✓ Retrieved {len(work['text']):,} characters via {work.get('method', 'unknown')}")
        else:
            print(f"  ✗ Failed to retrieve")
    
    print(f"\nPriority works: {success_count}/{len(retriever.PRIORITY_WORKS)} retrieved")
    
    # Retrieve subject-based works if not priority-only
    if not args.priority_only:
        print(f"\n{'='*60}")
        print(f"Phase 2: Retrieving extended corpus by subject...")
        print(f"Max per subject: {args.max_per_subject}")
        print(f"{'='*60}")
        
        extended_works = retriever.retrieve_extended_corpus(
            subjects=subjects,
            max_per_subject=args.max_per_subject
        )
        works.extend(extended_works)
        print(f"\nExtended corpus: {len(extended_works)} additional works retrieved")

    # Apply final date filter
    works = retriever.filter_by_date(works)
    
    # Save corpus
    filename = 'priority_works.jsonl' if args.priority_only else 'gutenberg_corpus.jsonl'
    retriever.save_corpus(works, filename)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RETRIEVAL COMPLETE")
    print(f"{'='*60}")
    print(f"Total works retrieved: {len(works)}")
    print(f"Total characters: {sum(w.get('length', 0) for w in works):,}")
    print(f"Output file: {os.path.join(args.output_dir, filename)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
