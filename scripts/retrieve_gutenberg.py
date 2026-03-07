#!/usr/bin/env python3
"""
Retrieve and filter Project Gutenberg texts for training-data preparation.

This script downloads books from Project Gutenberg based on predefined
priority works and subject filters.  It saves the corpus in JSONL format.

All works are filtered to ensure publication before 1969 (pre-moon landing)
to maintain temporal consistency with the Soviet utopia aesthetic.

The script is fully **idempotent** — re-running it will skip works that
already exist in the output files and append only new ones.  Use --reset
to wipe previous data and start fresh.

Retrieval is via direct HTTP download from gutenberg.org, with subject-based
search done by scraping the Gutenberg website.

Environment Variables:
    GUTENBERG_DATA: Base directory for output (default: output/gutenberg_corpus)
"""

import getpass
import grp
import json
import os
import pwd
import re
import requests
import subprocess
import sys
import tempfile
import time
from bs4 import BeautifulSoup
from urllib.parse import quote

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Temporal cutoff: All works must be published before 1969 (moon landing year)
TEMPORAL_CUTOFF_YEAR = 1969


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def _ensure_directory(path: str) -> None:
    """Create *path* (and parents) and verify the current user can write to it.

    The function is fully idempotent:
    - If the directory already exists and is writable, it is a no-op.
    - If it exists but is owned by another user, we attempt a recursive
      ``chown`` to the current user.  When that fails (no permission) we
      print the exact ``sudo`` command the operator should run and exit.
    - If it does not exist we create it; when that fails we again print
      the ``sudo`` command and exit.
    """
    uid = os.getuid()
    user = getpass.getuser()
    gid = os.getgid()
    group = grp.getgrgid(gid).gr_name

    # --- create if missing ------------------------------------------------
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError:
            print(f"\nError: cannot create {path} — permission denied.")
            print(f"Run the following command first, then retry:\n")
            print(f"  sudo mkdir -p {path} && sudo chown -R {user}:{group} {path}\n")
            sys.exit(1)

    # --- check / fix ownership --------------------------------------------
    dir_stat = os.stat(path)
    if dir_stat.st_uid != uid:
        # Attempt to take ownership
        try:
            subprocess.check_call(
                ['chown', '-R', f'{user}:{group}', path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Aligned ownership of {path} to {user}:{group}")
        except (subprocess.CalledProcessError, PermissionError):
            print(f"\nError: {path} is owned by "
                  f"uid {dir_stat.st_uid} ({pwd.getpwuid(dir_stat.st_uid).pw_name}), "
                  f"not {user}.")
            print(f"Run the following command first, then retry:\n")
            print(f"  sudo chown -R {user}:{group} {path}\n")
            sys.exit(1)

    # --- writability probe ------------------------------------------------
    probe = os.path.join(path, '.write_test')
    try:
        with open(probe, 'w') as f:
            f.write('ok')
        os.unlink(probe)
    except PermissionError:
        print(f"\nError: {path} exists but is not writable by {user}.")
        print(f"Run the following command first, then retry:\n")
        print(f"  sudo chown -R {user}:{group} {path} && sudo chmod -R u+rwX {path}\n")
        sys.exit(1)


class GutenbergRetriever:
    """Retrieve and filter Project Gutenberg texts for training-data preparation."""

    # Target Gutenberg IDs for priority works, organised by category
    PRIORITY_WORKS_BY_CATEGORY = {
        'Utopian/Dystopian': {
            624: "Looking Backward",          # Edward Bellamy
            6424: "A Modern Utopia",          # H.G. Wells
            3261: "News from Nowhere",        # William Morris
            1164: "The Iron Heel",            # Jack London — oligarchy, class struggle
            32: "Herland",                    # Charlotte Perkins Gilman
            61963: "We",                      # Yevgeny Zamyatin — THE Soviet dystopia
            12163: "The Sleeper Awakes",      # H.G. Wells — future society, class revolt
            1497: "The Republic",             # Plato — ideal society, philosopher-kings
            1998: "Thus Spake Zarathustra",   # Nietzsche — übermensch, will to power
        },
        'Russian Literature': {
            2554: "Crime and Punishment",     # Dostoevsky
            28054: "The Brothers Karamazov",  # Dostoevsky — free will, morality
            600: "Notes from the Underground", # Dostoevsky — alienation, consciousness
            2638: "The Idiot",                # Dostoevsky
            8117: "The Possessed",            # Dostoevsky — revolutionaries, nihilism
            1399: "Anna Karenina",            # Tolstoy
            2600: "War and Peace",            # Tolstoy
            3783: "Mother",                   # Maxim Gorky — revolutionary spirit
            47935: "Fathers and Sons",        # Turgenev — generational conflict, nihilism
            1081: "Dead Souls",               # Gogol
            7986: "Plays by Anton Chekhov, Second Series",  # Contains The Cherry Orchard, Three Sisters
            1756: "Uncle Vanya",              # Chekhov
            1754: "The Seagull",              # Chekhov
            2197: "The Gambler",              # Dostoevsky — obsession, fate
        },
        'Early Science Fiction': {
            83: "From the Earth to the Moon",     # Jules Verne (combined with Round the Moon)
            164: "20,000 Leagues Under the Sea",  # Jules Verne
            18857: "Journey to the Center of the Earth", # Jules Verne (full text version)
            1268: "The Mysterious Island",        # Jules Verne — survival, self-sufficiency
            35: "The Time Machine",               # H.G. Wells — class divide, evolution
            36: "The War of the Worlds",          # H.G. Wells — alien invasion
            1013: "The First Men in the Moon",    # H.G. Wells — lunar colony
            5230: "The Invisible Man",            # H.G. Wells
            159: "The Island of Doctor Moreau",   # H.G. Wells — playing god
            62: "A Princess of Mars",             # Burroughs — Mars adventure
            64: "The Gods of Mars",               # Burroughs — Mars sequel
            72: "Thuvia, Maid of Mars",           # Burroughs — Mars series
            1153: "The Chessmen of Mars",          # Burroughs — Mars + chess (Jetan), thematic overlap
            139: "The Lost World",                # Doyle — isolated civilization
            59112: "R.U.R.",                      # Čapek — robots, AI rebellion
            61213: "The 64-Square Madhouse",       # Fritz Leiber — sci-fi, computer plays chess (1962)
            84: "Frankenstein",                   # Shelley — created intelligence
            1059: "The World Set Free",           # H.G. Wells — atomic war, world government
            11696: "The Food of the Gods",        # H.G. Wells — scientific hubris
        },
        'Political Philosophy': {
            61: "The Communist Manifesto",        # Marx/Engels
            4341: "Mutual Aid",                   # Kropotkin — cooperation vs competition
            23428: "The Conquest of Bread",       # Kropotkin — anarcho-communism
            1232: "The Prince",                   # Machiavelli — power, statecraft
            815: "Democracy in America Vol 1",    # Tocqueville — political systems
            816: "Democracy in America Vol 2",    # Tocqueville
            3207: "Leviathan",                    # Hobbes — social contract, sovereignty
            46333: "The Social Contract",         # Rousseau — general will
        },
        'Isolation/Survival': {
            521: "Robinson Crusoe",               # Defoe — survival, self-reliance
            1184: "The Count of Monte Cristo",    # Dumas — imprisonment, revenge
            30197: "Farthest North Vol I",        # Nansen — polar exploration, survival
            34120: "Farthest North Vol II",       # Nansen — polar exploration, survival
        },
        'Chess & Strategy': {
            33870: "Chess Fundamentals",          # Capablanca — strategy, endgames, annotated games
            5614: "Chess Strategy",               # Edward Lasker — opening theory, middlegame, endgame
            4913: "Chess and Checkers: the Way to Mastership",  # Edward Lasker — rules, strategy
            16377: "The Blue Book of Chess",      # Howard Staunton — rules, openings, game annotations
            34180: "The Exploits and Triumphs of Paul Morphy",  # Frederick M. Edge — biography, annotated games
            4902: "Chess History and Reminiscences",  # H.E. Bird — historical survey, anecdotes
            55278: "Chess Generalship, Vol. I: Grand Reconnaissance",  # Franklin K. Young — strategic principles
            10672: "Game and Playe of the Chesse",  # William Caxton — earliest printed chess text in English (1474)
            4542: "Checkmates for Three Pieces",  # W.B. Fishburne — tactical patterns
            4656: "Checkmates for Four Pieces",   # W.B. Fishburne — tactical patterns
            39445: "Hoyle's Games Modernized",    # Prof. Hoffmann / Edmond Hoyle — rules and strategy
            36821: "Maxims and Hints on Angling, Chess, Shooting",  # Richard Penn — chess maxims
            60420: "Observations on the Automaton Chess Player",  # Oxford graduate (~1819) — Mechanical Turk, proto-AI
            61410: "An Attempt to Analyse the Automaton Chess Player",  # Robert Willis (1821) — Mechanical Turk analysis
            64061: "War-Chess, or the Game of Battle",  # Charles Richardson — kriegsspiel, chess-derived war strategy
            63660: "The Game of Chess: A Play in One Act",  # Kenneth Sawyer Goodman (1914) — chess-themed drama
        },
        'Satire': {
            1695: "The Man Who Was Thursday",     # Chesterton — anarchists, conspiracy
            829: "Gulliver's Travels",            # Swift — political satire
            1080: "A Modest Proposal",            # Swift — savage satire
            19942: "Candide",                     # Voltaire — satirical philosophy
        },
    }

    # Flat dict for backward compatibility: {id: title}
    PRIORITY_WORKS = {
        gid: title
        for category_works in PRIORITY_WORKS_BY_CATEGORY.values()
        for gid, title in category_works.items()
    }

    # Subject filters for bulk retrieval
    # Uses Library of Congress Subject Headings (LCSH) terminology
    # Aligned with Deep Red themes: Soviet Mars colony, AI chess master,
    # political satire, survival, ideological extremism
    SUBJECT_FILTERS = [
        # Fiction genres
        "Science fiction",
        "Satire",
        "Political fiction",
        "Allegories",
        "Utopias",
        "Dystopias",

        # Soviet/Russian themes
        "Soviet Union",
        "Russia",
        "Socialism",
        "Communism",
        "Propaganda",
        "Totalitarianism",
        "Collectivism",

        # Space and Mars
        "Space flight",
        "Mars (Planet)",
        "Interplanetary voyages",
        "Space colonies",
        "Life on other planets",
        "Outer space",
        "Astronautics",

        # AI/Machine/Chess themes
        "Chess",
        "Automata",
        "Automaton chess players",
        "Chess -- Early works to 1800",
        "War chess (Game)",
        "Machinery",
        "Robots",
        "Calculating machines",

        # Survival and isolation
        "Survival",
        "Wilderness survival",
        "Shipwrecks",    # Keep — analogous to crash survival
        "Castaways",
        "Prisoners",
        "Exiles",

        # Political/Social conflict
        "Revolutions",
        "Political science",
        "Secret societies",
        "Conspiracies",
        "Dictatorship",
        "Oligarchy",
        "Anarchism",
        "Radicalism",

        # Class and power
        "Capitalism",
        "Rich and poor",
        "Working class",
        "Labor",
        "Power (Social sciences)",

        # Human condition themes
        "Human evolution",
        "Evolution",
        "Civilization",
        "Future life",
        "End of the world",
        "Prophecies",

        # Colonisation/Exploration
        "Colonization",
        "Explorers",
        "Pioneers",
        "Frontier and pioneer life",

        # Psychology/Philosophy
        "Free will and determinism",
        "Man-machine systems",
        "Good and evil",
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
        "Plato, 428 BC-348 BC",
        "Nietzsche, Friedrich Wilhelm, 1844-1900",
        "Machiavelli, Niccolò, 1469-1527",
        "Tocqueville, Alexis de, 1805-1859",
        "Hobbes, Thomas, 1588-1679",
        "Rousseau, Jean-Jacques, 1712-1778",
        "Defoe, Daniel, 1661-1731",
        "Dumas, Alexandre, 1802-1870",
        "Nansen, Fridtjof, 1861-1930",
        "Franklin, Benjamin, 1706-1790",
        "Lasker, Edward, 1885-1981",  # Note: died 1981 but chess book is 1915
        "Staunton, Howard, 1810-1874",
        "Bird, H. E. (Henry Edward), 1830-1908",
        "Edge, Frederick Milnes, 1830-1895",
        "Caxton, William, approximately 1422-1491",
        "Leiber, Fritz, 1910-1992",
        "Goodman, Kenneth Sawyer, 1883-1918",
        "Willis, Robert, 1800-1875",
        "Richardson, Charles, -1913",  # War-Chess author
        "Chesterton, G. K. (Gilbert Keith), 1874-1936",
        "Swift, Jonathan, 1667-1745",
        "Voltaire, 1694-1778",
    }

    def __init__(self, output_dir: str, max_year: int = TEMPORAL_CUTOFF_YEAR, prefer_http: bool = True,
                 verbose: bool = False):
        """Initialise the retriever.

        Args:
            output_dir: Directory to save retrieved works
            max_year: Maximum publication year (temporal cutoff)
            prefer_http: If True, use HTTP for individual works (faster, more reliable).
                         Library is still used for subject search if available.
            verbose: If True, print detailed per-work output instead of progress bars.
        """
        self.output_dir = output_dir
        self.max_year = max_year
        self.prefer_http = prefer_http
        self.verbose = verbose

        # Create output directory tree and verify writability
        _ensure_directory(output_dir)

        self.retrieved_ids: set[int] = set()  # Track already retrieved IDs to avoid duplicates
        self.rejected_ids: dict[int, str] = {}    # Track previously rejected IDs {id: reason}

        # Load existing IDs from corpus files to avoid re-downloading
        self._load_existing_corpus_ids()
        self._load_rejected_ids()

    # ------------------------------------------------------------------
    # Idempotency helpers
    # ------------------------------------------------------------------

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
                    if self.verbose:
                        print(f"Loaded {len(self.retrieved_ids)} existing work IDs from {os.path.basename(corpus_file)}")
                except Exception as e:
                    print(f"Warning: Could not load existing corpus from {corpus_file}: {e}")

    def _load_rejected_ids(self):
        """Load IDs previously rejected (non-English, bad date, etc.) to skip on re-run."""
        path = os.path.join(self.output_dir, 'rejected_ids.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.rejected_ids = {int(k): v for k, v in data.items()}
                if self.verbose:
                    print(f"Loaded {len(self.rejected_ids)} previously rejected work IDs")
            except Exception as e:
                print(f"Warning: Could not load rejected IDs from {path}: {e}")

    def _save_rejected_ids(self):
        """Persist rejected IDs to disk (atomic write)."""
        path = os.path.join(self.output_dir, 'rejected_ids.json')
        fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=self.output_dir,
                                   prefix='.rejected_ids.')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in self.rejected_ids.items()}, f,
                          ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _reject(self, gutenberg_id: int, reason: str) -> None:
        """Record a rejected work and persist immediately."""
        self.rejected_ids[gutenberg_id] = reason
        self._save_rejected_ids()

    def status(self) -> dict:
        """Return a summary of existing corpus data in the output directory.

        Useful for checking state before a run and confirming idempotency.
        """
        info: dict = {
            'output_dir': self.output_dir,
            'existing_ids': len(self.retrieved_ids),
            'rejected_ids': len(self.rejected_ids),
            'files': {},
        }
        for name in ('gutenberg_corpus.jsonl', 'priority_works.jsonl'):
            path = os.path.join(self.output_dir, name)
            if os.path.exists(path):
                size = os.path.getsize(path)
                count = 0
                total_chars = 0
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                w = json.loads(line)
                                count += 1
                                total_chars += w.get('length', 0)
                            except json.JSONDecodeError:
                                pass
                info['files'][name] = {
                    'works': count,
                    'size_mb': round(size / (1024 * 1024), 1),
                    'total_chars': total_chars,
                }
        return info

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

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

    def _titles_match(self, expected: str, actual: str) -> bool:
        """Check if actual title matches expected title.

        Uses fuzzy matching to handle minor variations like:
        - Subtitle differences: "We" vs "We: A Novel"
        - Article variations: "The Time Machine" vs "Time Machine, The"
        - Punctuation differences

        Args:
            expected: The title we expect (from PRIORITY_WORKS)
            actual: The title extracted from the downloaded content

        Returns:
            True if titles match sufficiently
        """
        if not expected or not actual:
            return False

        # Normalise both titles for comparison
        def normalize(title: str) -> str:
            # Lowercase
            t = title.lower()
            # Remove leading articles
            for article in ['the ', 'a ', 'an ']:
                if t.startswith(article):
                    t = t[len(article):]
            # Remove trailing articles (e.g., ", The")
            for article in [', the', ', a', ', an']:
                if t.endswith(article):
                    t = t[:-len(article)]
            # Remove subtitles (after colon or dash)
            t = re.split(r'[:\-—–]', t)[0]
            # Remove punctuation and extra whitespace
            t = re.sub(r'[^\w\s]', '', t)
            t = ' '.join(t.split())
            return t.strip()

        norm_expected = normalize(expected)
        norm_actual = normalize(actual)

        # Exact match after normalisation
        if norm_expected == norm_actual:
            return True

        # Check if one contains the other (for subtitle variations)
        if norm_expected in norm_actual or norm_actual in norm_expected:
            return True

        # Check word overlap (at least 80% of expected words present)
        expected_words = set(norm_expected.split())
        actual_words = set(norm_actual.split())
        if expected_words and len(expected_words & actual_words) / len(expected_words) >= 0.8:
            return True

        return False

    # ------------------------------------------------------------------
    # HTTP retrieval
    # ------------------------------------------------------------------

    def retrieve_by_http(self, gutenberg_id: int, title: str) -> dict:
        """Retrieve a work directly via HTTP (fallback method).

        Tries multiple strategies in order:
        1. Standard plain-text URL patterns
        2. Additional text URLs discovered from RDF metadata
        3. Additional static text URL patterns (-8.txt encoding variant)
        4. HTML-to-text conversion from RDF-listed HTML files
        5. HTML-to-text conversion from common HTML URL patterns
        6. Scrape the ebook download page for any text/HTML links
        """
        try:
            # First, fetch RDF metadata for better date detection
            rdf_metadata = self._fetch_gutenberg_rdf(gutenberg_id)

            # ---- Strategy 1: Standard plain-text URL patterns ----
            urls = [
                f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
                f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
                f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
            ]

            # ---- Strategy 2: Text URLs from RDF metadata ----
            if rdf_metadata:
                for rdf_url in rdf_metadata.get('text_urls', []):
                    if rdf_url not in urls:
                        urls.append(rdf_url)

            # ---- Strategy 3: Additional static text URL patterns ----
            extra_txt = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-8.txt"
            if extra_txt not in urls:
                urls.append(extra_txt)

            text = None
            retrieval_method = 'http'
            for url in urls:
                print(f"  Trying {url}")
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        candidate = response.text
                        # Sanity: reject tiny responses (< 500 chars) that are
                        # likely index pages or readme stubs, not actual books
                        if len(candidate) >= 500:
                            text = candidate
                            break
                except requests.RequestException:
                    continue

            # ---- Strategy 4–6: HTML fallback ----
            if not text:
                text = self._try_html_fallback(gutenberg_id, rdf_metadata)
                if text:
                    retrieval_method = 'http_html'

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
                self._reject(gutenberg_id, f"non-English: {display_title}")
                return None

            # Extract title — prefer RDF, then text header, then provided
            extracted_title = title
            if rdf_metadata and rdf_metadata.get('title'):
                extracted_title = rdf_metadata['title']
            elif title.startswith("Unknown_"):
                title_match = re.search(r'Title:\s*(.+)', text[:2000])
                if title_match:
                    extracted_title = title_match.group(1).strip()

            # Validate title matches expected (for priority works)
            if not title.startswith("Unknown_") and not self._titles_match(title, extracted_title):
                print(f"  Title mismatch! Expected: '{title}', Got: '{extracted_title}'")
                self._reject(gutenberg_id, f"title-mismatch: expected '{title}', got '{extracted_title}'")
                return None

            # Try to extract publication year — multiple sources (ordered by reliability)
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

            # 3. Check title for year (lowest priority — only for reference/annual works)
            # This is last because titles like "1984" are misleading (written 1949)
            if not pub_year:
                pub_year = self._extract_year_from_title(extracted_title)

            # Strip headers/footers
            cleaned_text = self.strip_gutenberg_headers(text)

            # Extract author — prefer RDF, then text header
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
                'method': retrieval_method
            }

            # Add author death year if found (useful for temporal validation)
            if author_death_year:
                result['author_death_year'] = author_death_year

            return result
        except Exception as e:
            print(f"  Error with HTTP retrieval: {e}")
            return None

    # ------------------------------------------------------------------
    # HTML-to-text fallback helpers
    # ------------------------------------------------------------------

    def _html_to_text(self, html: str) -> str:
        """Convert HTML content to plain text using BeautifulSoup.

        Strips all tags, collapses whitespace, and removes navigation /
        boilerplate elements commonly found in Gutenberg HTML files.

        Args:
            html: Raw HTML string

        Returns:
            Extracted plain text, or None if extraction fails or result
            is too short to be a real book.
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script, style, nav, and header/footer elements
            for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()

            # Extract text; use newlines as separators for block elements
            text = soup.get_text(separator='\n')

            # Collapse multiple blank lines
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            # Must be substantial enough to be actual book content
            # (10K chars ≈ 2000 words — filters out index/navigation pages)
            if len(text) < 10000:
                return None
            return text
        except Exception:
            return None

    def _try_html_fallback(self, gutenberg_id: int, rdf_metadata: dict) -> str:
        """Attempt to retrieve book text via HTML download + tag stripping.

        Tries (in order):
        4. HTML URLs listed in RDF metadata
        5. Common Gutenberg HTML URL patterns
        6. Scraping the ebook download page for text/HTML links

        Args:
            gutenberg_id: The Project Gutenberg ID
            rdf_metadata: Previously fetched RDF metadata dict (may be None)

        Returns:
            Plain text extracted from HTML, or None on failure.
        """
        html_urls = []

        # ---- Strategy 4: HTML URLs from RDF metadata ----
        if rdf_metadata:
            html_urls.extend(rdf_metadata.get('html_urls', []))

        # ---- Strategy 5: Common HTML URL patterns ----
        common_html = [
            f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}-images.html",
            f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.html",
            f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-h/{gutenberg_id}-h.htm",
            f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-h/{gutenberg_id}-h.html",
        ]
        for url in common_html:
            if url not in html_urls:
                html_urls.append(url)

        # De-duplicate while preserving order
        seen = set()
        deduped = []
        for url in html_urls:
            if url not in seen:
                seen.add(url)
                deduped.append(url)
        html_urls = deduped

        for url in html_urls:
            print(f"  Trying HTML fallback: {url}")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.text) >= 500:
                    text = self._html_to_text(response.text)
                    if text:
                        print(f"  Retrieved via HTML conversion")
                        return text
            except requests.RequestException:
                continue

        # ---- Strategy 6: Scrape ebook download page ----
        return self._try_ebook_page_fallback(gutenberg_id)

    def _try_ebook_page_fallback(self, gutenberg_id: int) -> str:
        """Last-resort: scrape the Gutenberg ebook page for download links.

        Parses ``https://www.gutenberg.org/ebooks/{id}`` looking for any
        text/plain or text/html download links not yet tried.

        Args:
            gutenberg_id: The Project Gutenberg ID

        Returns:
            Plain text (possibly extracted from HTML), or None.
        """
        page_url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}"
        print(f"  Trying ebook page fallback: {page_url}")
        try:
            response = requests.get(page_url, timeout=30)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for download links with content-type info
            # Gutenberg marks them in <a> tags with a type= attribute
            text_links = []
            html_links = []

            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href', '')
                link_type = a_tag.get('type', '').lower()
                link_text = a_tag.get_text(strip=True).lower()

                # Build absolute URL
                if href.startswith('/'):
                    href = f"https://www.gutenberg.org{href}"
                elif not href.startswith('http'):
                    continue

                # Skip audio, images, readme, zip, and rdf files
                if any(ext in href.lower() for ext in [
                    '.ogg', '.mp3', '.wav', '.jpg', '.png', '.zip',
                    'readme', '.rdf', 'cover',
                ]):
                    continue

                if 'text/plain' in link_type:
                    text_links.append(href)
                elif 'text/html' in link_type:
                    html_links.append(href)
                elif href.endswith('.txt') and 'readme' not in link_text:
                    text_links.append(href)
                elif href.endswith(('.html', '.htm')) and 'index' not in link_text:
                    html_links.append(href)

            # Try text links first
            for url in text_links:
                print(f"  Trying scraped text link: {url}")
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200 and len(resp.text) >= 500:
                        return resp.text
                except requests.RequestException:
                    continue

            # Then HTML links
            for url in html_links:
                print(f"  Trying scraped HTML link: {url}")
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200 and len(resp.text) >= 500:
                        text = self._html_to_text(resp.text)
                        if text:
                            print(f"  Retrieved via ebook-page HTML conversion")
                            return text
                except requests.RequestException:
                    continue

        except Exception as e:
            if self.verbose:
                print(f"  Ebook page fallback failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Year extraction helpers
    # ------------------------------------------------------------------

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
        - text_urls: List of text/plain file URLs from hasFormat
        - html_urls: List of text/html file URLs from hasFormat
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

            # ----- Extract file download URLs by MIME type -----
            # Parse <dcterms:hasFormat> blocks to find text/plain and text/html URLs.
            # Each block looks like:
            #   <pgterms:file rdf:about="https://...url...">
            #     ...
            #     <rdf:value ...>text/plain; charset=utf-8</rdf:value>
            #     ...
            #   </pgterms:file>
            text_urls = []
            html_urls = []

            for url, block in re.findall(
                r'<pgterms:file\s+rdf:about="([^"]+)"[^>]*>(.*?)</pgterms:file>',
                rdf_text, re.DOTALL,
            ):
                mime_match = re.search(
                    r'<rdf:value[^>]*>([^<]+)</rdf:value>', block,
                )
                if not mime_match:
                    continue
                mime = mime_match.group(1).strip().lower()

                # Skip readme files, zip archives, and index pages
                if any(skip in url.lower() for skip in ['readme', 'index']):
                    continue
                if url.lower().endswith('.zip'):
                    continue

                if mime.startswith('text/plain'):
                    text_urls.append(url)
                elif mime.startswith('text/html'):
                    html_urls.append(url)

            metadata['text_urls'] = text_urls
            metadata['html_urls'] = html_urls

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
                        # Century reference — return end of century
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

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve_by_id(self, gutenberg_id: int, title: str = None, skip_date_check: bool = False) -> dict:
        """Retrieve a specific work by Gutenberg ID.

        Uses direct HTTP download.

        Args:
            gutenberg_id: The Project Gutenberg eBook ID
            title: Optional title override
            skip_date_check: If True, skip temporal validation (for known pre-1969 priority works)
        """
        # Check if already retrieved (idempotent — skip without error)
        if gutenberg_id in self.retrieved_ids:
            if self.verbose:
                print(f"  Skipping {gutenberg_id} (already in corpus)")
            return None

        # Check if previously rejected (non-English, bad date, etc.)
        if gutenberg_id in self.rejected_ids:
            if self.verbose:
                print(f"  Skipping {gutenberg_id} (previously rejected: {self.rejected_ids[gutenberg_id]})")
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
                if self.verbose:
                    print(f"  Skipping {gutenberg_id}: Could not verify pre-{self.max_year} publication")
                self._reject(gutenberg_id, f"temporal: could not verify pre-{self.max_year}")
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
                if self.verbose:
                    print(f"  Rejected: '{title}' — publication year {pub_year} >= {self.max_year}")
                gid = work.get('id')
                if gid:
                    self._reject(gid, f"pub_year {pub_year} >= {self.max_year}: {title}")
                return False
            return True

        # If author is known to have died before cutoff, work is valid
        if work.get('is_known_pre1969_author'):
            return True

        # Check author death year from RDF metadata
        author_death_year = work.get('author_death_year')
        if author_death_year and author_death_year < self.max_year:
            if self.verbose:
                print(f"  Accepted: '{title}' — author died {author_death_year} (before {self.max_year})")
            return True

        # For Gutenberg works, most are pre-1928 (US copyright threshold)
        # We can be reasonably confident they're pre-1969
        # But log a warning for manual review
        if self.verbose:
            print(f"  Warning: Could not determine publication year for '{title}' — assuming pre-{self.max_year}")
        return True

    # ------------------------------------------------------------------
    # Subject-based retrieval
    # ------------------------------------------------------------------

    def _scrape_subject_ids_from_web(self, subject: str, max_results: int = 100) -> list:
        """Scrape book IDs from Gutenberg website by subject search.

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
                    if self.verbose:
                        print(f"  Failed to fetch search page: HTTP {response.status_code}")
                    break

                soup = BeautifulSoup(response.text, 'html.parser')

                # Find book links — they're in format /ebooks/12345
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
                if self.verbose:
                    print(f"  Error scraping web: {e}")
                break

        return ids

    def retrieve_by_subject(self, subject: str, max_results: int = 50) -> list:
        """Retrieve works matching a subject filter.

        Searches the Gutenberg website by subject and downloads matching works.

        Args:
            subject: The Gutenberg subject to search for
            max_results: Maximum number of works to retrieve for this subject

        Returns:
            List of work dictionaries
        """
        if self.verbose:
            print(f"\nSearching for subject: '{subject}'")
        ids = self._scrape_subject_ids_from_web(subject, max_results=max_results * 2)
        if self.verbose:
            print(f"  Found {len(ids)} potential works")

        if not ids:
            if self.verbose:
                print(f"  No works found for subject '{subject}'")
            return []

        works = []
        count = 0
        skipped = 0

        # Build candidate list (excluding already-retrieved IDs)
        candidates = [gid for gid in sorted(ids)
                      if gid not in self.retrieved_ids
                      and gid not in self.rejected_ids][:max_results * 2]

        if not self.verbose and tqdm is not None:
            iter_ids = tqdm(candidates, desc=f"  {subject[:30]}", unit="work", leave=False)
        else:
            iter_ids = candidates

        for gutenberg_id in iter_ids:
            if count >= max_results:
                break

            try:
                work = self.retrieve_by_id(gutenberg_id)
                if work:
                    work['subject'] = subject  # Tag with source subject
                    works.append(work)
                    count += 1
                    if self.verbose:
                        print(f"  [{count}/{max_results}] {work['title']} by {work['author']} (ID: {gutenberg_id})")
                else:
                    skipped += 1
            except Exception as e:
                if self.verbose:
                    print(f"  Failed to retrieve ID {gutenberg_id}: {e}")
                skipped += 1

        # Close tqdm iterator if used
        if hasattr(iter_ids, 'close'):
            iter_ids.close()

        if skipped > 0 and self.verbose:
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

        if not self.verbose and tqdm is not None:
            subject_iter = tqdm(subjects, desc="Subjects", unit="subj")
        else:
            subject_iter = subjects

        for subject in subject_iter:
            subject_works = self.retrieve_by_subject(subject, max_results=max_per_subject)
            all_works.extend(subject_works)
            if self.verbose:
                print(f"  Subtotal: {len(all_works)} works retrieved so far")
            elif tqdm is not None and hasattr(subject_iter, 'set_postfix'):
                subject_iter.set_postfix(works=len(all_works))

        if hasattr(subject_iter, 'close'):
            subject_iter.close()

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
                    if self.verbose:
                        print(f"  Filtered out: {work['title']} (published {work['pub_year']})")
            # If author is known pre-1969, include
            elif work.get('is_known_pre1969_author'):
                filtered.append(work)
            # Otherwise, assume valid (Gutenberg copyright rules)
            else:
                filtered.append(work)

        return filtered

    # ------------------------------------------------------------------
    # Atomic save (idempotent)
    # ------------------------------------------------------------------

    def save_corpus(self, works: list, filename: str, append: bool = True):
        """Save retrieved works to JSONL format.

        Uses atomic write (write to temp file, then rename) so that a
        crash mid-write never corrupts the corpus.

        Args:
            works: List of work dictionaries to save
            filename: Output filename
            append: If True, append to existing file and skip duplicates.
                    If False, overwrite.
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
                if self.verbose:
                    print(f"Found {len(existing_ids)} existing works in {filename}")
            except Exception as e:
                print(f"Warning: Could not read existing corpus: {e}")

        # Filter out duplicates from new works
        new_works = [w for w in works if w and w.get('id') not in existing_ids]

        if not new_works and existing_works:
            print(f"No new works to add — {filename} already up-to-date ({len(existing_works)} works)")
            return

        # Combine existing and new works
        all_works = existing_works + new_works

        # Atomic write: temp file in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(
            suffix='.jsonl.tmp',
            dir=self.output_dir,
            prefix=f'.{filename}.',
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                for work in all_works:
                    if work:
                        f.write(json.dumps(work, ensure_ascii=False) + '\n')
            os.replace(tmp_path, output_path)     # atomic on POSIX
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        print(f"Saved {len(all_works)} total works to {output_path} ({len(new_works)} new, {len(existing_works)} existing)")


def main():
    """Main entry point for the script."""
    import argparse
    start_time = time.time()

    # Check for GUTENBERG_DATA environment variable
    gutenberg_data = os.environ.get('GUTENBERG_DATA')
    if gutenberg_data:
        default_output = os.path.join(gutenberg_data, 'corpus')
        print(f"Using GUTENBERG_DATA environment variable: {gutenberg_data}")
    else:
        default_output = 'corpus'
        print("Warning: GUTENBERG_DATA environment variable not set. Output will go to ./corpus")

    parser = argparse.ArgumentParser(
        description="Retrieve Project Gutenberg texts for training-data preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output defaults to $GUTENBERG_DATA/corpus (or ./corpus when the env var is unset).

Examples:
  python retrieve_gutenberg.py                  # Full retrieval (priority + subjects)
  python retrieve_gutenberg.py --priority-only   # Priority works only
  python retrieve_gutenberg.py --status           # Show existing corpus stats
  python retrieve_gutenberg.py --info             # List priority works & subjects
  python retrieve_gutenberg.py --reset            # Wipe corpus and start fresh
  python retrieve_gutenberg.py --subjects "Science fiction,Chess" --max-per-subject 20
        """
    )
    parser.add_argument('--info', action='store_true',
                        help='Display priority works and subject filters, then exit (no download)')
    parser.add_argument('--status', action='store_true',
                        help='Show status of existing corpus files and exit')
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
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-work output instead of progress bars')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # --info: display priority works and subjects, then exit
    # ------------------------------------------------------------------
    if args.info:
        print(f"\n{'='*70}")
        print(f"PROJECT GUTENBERG SOURCE MATERIALS")
        print(f"Temporal cutoff: pre-{args.max_year}")
        print(f"{'='*70}\n")

        print("PRIORITY WORKS ({} books)".format(len(GutenbergRetriever.PRIORITY_WORKS)))
        print("-" * 70)
        for category, works in GutenbergRetriever.PRIORITY_WORKS_BY_CATEGORY.items():
            print(f"\n{category} ({len(works)} works):")
            for gid, title in works.items():
                print(f"  {gid:>6}: {title}")

        print(f"\n{'='*70}")
        print("SUBJECT FILTERS ({} subjects)".format(len(GutenbergRetriever.SUBJECT_FILTERS)))
        print("-" * 70)
        for i, subject in enumerate(GutenbergRetriever.SUBJECT_FILTERS, 1):
            print(f"  {i:>2}. {subject}")

        print(f"\n{'='*70}\n")
        return

    # ------------------------------------------------------------------
    # --status: show existing corpus state, then exit
    # ------------------------------------------------------------------
    if args.status:
        retriever = GutenbergRetriever(args.output_dir, max_year=args.max_year)
        info = retriever.status()

        print(f"\n{'='*60}")
        print(f"CORPUS STATUS")
        print(f"{'='*60}")
        print(f"Output directory : {info['output_dir']}")
        print(f"Known work IDs   : {info['existing_ids']}")
        print(f"Rejected IDs     : {info['rejected_ids']}")

        if info['files']:
            for name, finfo in info['files'].items():
                print(f"\n  {name}:")
                print(f"    Works       : {finfo['works']}")
                print(f"    Size        : {finfo['size_mb']} MB")
                print(f"    Total chars : {finfo['total_chars']:,}")
        else:
            print("\n  (no corpus files found)")

        print(f"\n{'='*60}\n")
        return

    # ------------------------------------------------------------------
    # --reset: delete existing corpus files
    # ------------------------------------------------------------------
    if args.reset:
        corpus_files = [
            os.path.join(args.output_dir, 'gutenberg_corpus.jsonl'),
            os.path.join(args.output_dir, 'priority_works.jsonl'),
            os.path.join(args.output_dir, 'rejected_ids.json'),
        ]
        for corpus_file in corpus_files:
            if os.path.exists(corpus_file):
                os.remove(corpus_file)
                print(f"Deleted: {corpus_file}")
        print("Starting fresh corpus build...\n")

    # Parse subjects if provided
    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(',')]

    verbose = args.verbose
    retriever = GutenbergRetriever(args.output_dir, max_year=args.max_year, verbose=verbose)

    # Warn if tqdm is missing in default (non-verbose) mode
    if not verbose and tqdm is None:
        print("Note: install 'tqdm' for progress bars (pip install tqdm). Falling back to verbose output.")
        verbose = True
        retriever.verbose = True

    print(f"\n{'='*60}")
    print(f"Project Gutenberg Retriever")
    print(f"Temporal cutoff: pre-{args.max_year}")
    print(f"Output directory: {args.output_dir}")
    if retriever.retrieved_ids:
        print(f"Existing works: {len(retriever.retrieved_ids)} (will be skipped)")
    if retriever.rejected_ids:
        print(f"Rejected works: {len(retriever.rejected_ids)} (will be skipped)")
    print(f"{'='*60}\n")

    # Retrieve priority works
    print("Phase 1: Retrieving priority works...")
    if verbose:
        print(f"Total priority works: {len(retriever.PRIORITY_WORKS)}\n")

    works = []
    success_count = 0
    fail_count = 0
    skipped_count = 0

    # Build list of works to download (not already in corpus)
    to_download = [(gid, title) for gid, title in retriever.PRIORITY_WORKS.items()
                   if gid not in retriever.retrieved_ids]
    skipped_count = len(retriever.PRIORITY_WORKS) - len(to_download)

    if not verbose and tqdm is not None:
        pbar = tqdm(to_download, desc="Priority works", unit="book")
    else:
        pbar = to_download

    for gutenberg_id, title in pbar:
        if verbose:
            print(f"Retrieving: {title} (ID: {gutenberg_id})")
        # Priority works are known to be pre-1969, skip date check
        work = retriever.retrieve_by_id(gutenberg_id, title, skip_date_check=True)
        if work:
            works.append(work)
            success_count += 1
            if verbose:
                print(f"  ✓ Retrieved {len(work['text']):,} characters via {work.get('method', 'unknown')}")
        else:
            fail_count += 1
            if verbose:
                print(f"  ✗ Failed to retrieve")

    if hasattr(pbar, 'close'):
        pbar.close()

    print(f"\nPriority works: {success_count} new, {skipped_count} already present"
          f"{f', {fail_count} failed' if fail_count else ''}"
          f" (of {len(retriever.PRIORITY_WORKS)} total)")

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
        print(f"Extended corpus: {len(extended_works)} additional works retrieved")

    # Apply final date filter
    works = retriever.filter_by_date(works)

    # Save corpus (atomic write)
    filename = 'priority_works.jsonl' if args.priority_only else 'gutenberg_corpus.jsonl'
    retriever.save_corpus(works, filename)

    # Summary
    elapsed_time = time.time() - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = elapsed_time % 60

    print(f"\n{'='*60}")
    print(f"RETRIEVAL COMPLETE")
    print(f"{'='*60}")
    print(f"New works retrieved : {len(works)}")
    print(f"New characters      : {sum(w.get('length', 0) for w in works):,}")
    print(f"Total in corpus     : {len(retriever.retrieved_ids)} works")
    print(f"Output file         : {os.path.join(args.output_dir, filename)}")
    print(f"Runtime             : {elapsed_minutes}m {elapsed_seconds:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
