#!/usr/bin/env python3
"""
Retrieve and convert chess content for training-data preparation.

This script downloads PGN game databases and Internet Archive chess books,
then converts PGN games into natural-language prose suitable for LLM
training.  It is organised into independent **phases** that can be run
individually for easy testing and incremental builds.

    Phase 1  — Download PGN databases (PGN Mentor player/event files,
               Lumbras Gigabase)
    Phase 2  — Convert downloaded PGN to narrative text (pre-1969 filter)
    Phase 3  — Download public-domain chess books from Internet Archive

Phase 2 safety mechanisms:
    - Per-game parse timeout (GAME_PARSE_TIMEOUT, default 30s)
    - Per-file wall-clock timeout (dynamic: FILE_PARSE_TIMEOUT base + per-MB scaling)
    - Consecutive-error detection (MAX_CONSECUTIVE_ERRORS, default 10)
    - Automatic skip-ahead to the next game on parse failures
    - Files sorted largest-first for better parallel utilisation
    - Optional parallel mode via --workers N

Post-processed output is written to the *corpus* subdirectory inside
$CHESS_DATA in JSONL format, ready for tokenisation.

Gutenberg chess books are handled separately by retrieve_gutenberg.py
(Chess & Strategy category) — see Chess-Setup.md for details.

Environment Variables:
    CHESS_DATA: Base directory for chess data (default: /mnt/data/chess)
"""

import getpass
import glob
import grp
import io
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import urllib3
import warnings

import requests
from bs4 import BeautifulSoup

# Suppress InsecureRequestWarning for sites with expired SSL certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import chess
    import chess.pgn
    HAS_PYTHON_CHESS = True
except ImportError:
    HAS_PYTHON_CHESS = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPORAL_CUTOFF_YEAR = 1969
TEMPORAL_CUTOFF_DATE = "1969.07.20"

# Phase 2 — parsing safety limits
GAME_PARSE_TIMEOUT = 30     # seconds: max time for a single read_game() call
FILE_PARSE_TIMEOUT = 300    # seconds: base wall-clock time per PGN file
FILE_TIMEOUT_PER_MB = 30    # seconds: additional time per MB of file size
MAX_CONSECUTIVE_ERRORS = 10 # skip file after N consecutive parse failures

# Requests session defaults
USER_AGENT = (
    "Mozilla/5.0 (compatible; DeepRedAI-ChessRetriever/1.0; "
    "+https://github.com/DeepRedAI)"
)
REQUEST_TIMEOUT = 60
RETRY_DELAY = 2          # seconds between retries
MAX_RETRIES = 3

# Domains with expired/broken SSL certificates — bypass verification
SSL_BYPASS_DOMAINS = {"pgnmentor.com", "www.pgnmentor.com"}

# ---------------------------------------------------------------------------
# Internet Archive — public-domain chess books (pre-1929 US copyright)
# ---------------------------------------------------------------------------

# Curated list: (archive_org_identifier, title, author, year)
# Only pre-1929 works (safely US public domain)
ARCHIVE_CHESS_BOOKS = [
    ("my-system-2020",
     "My System", "Aron Nimzowitsch", 1925),
    ("praxisofmysystem00nimz",
     "The Praxis of My System", "Aron Nimzowitsch", 1929),
    ("gameofchess00sieg",
     "The Game of Chess", "Siegbert Tarrasch", 1931),   # NOTE: verify PD status
    ("chessplayerscom01staugoog",
     "The Chess-Player's Companion", "Howard Staunton", 1849),
    ("chessplayershan00greegoog",
     "The Chess-Player's Handbook", "Howard Staunton", 1847),
    ("mybestgamesofche0000alek",
     "My Best Games of Chess 1908-1923", "Alexander Alekhine", 1924),
    ("mybestgamesofche00alek",
     "My Best Games of Chess 1908-1937", "Alexander Alekhine", 1937),
    ("principleschess00masogoog",
     "The Principles of Chess", "James Mason", 1894),
    ("artofchesscombin00euge",
     "The Art of Chess Combination", "Eugène Znosko-Borovsky", 1936),  # verify PD
    ("howtothinkaheadi0000iaho_p6v5",
     "How to Think Ahead in Chess", "I.A. Horowitz", 1951),  # verify PD
]


# ---------------------------------------------------------------------------
# Directory bootstrap (reused from retrieve_gutenberg.py pattern)
# ---------------------------------------------------------------------------

def _ensure_directory(path: str) -> None:
    """Create *path* (and parents) and verify the current user can write."""
    uid = os.getuid()
    user = getpass.getuser()
    gid = os.getgid()
    group = grp.getgrgid(gid).gr_name

    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError:
            print(f"\nError: cannot create {path} — permission denied.")
            print(f"Run the following command first, then retry:\n")
            print(f"  sudo mkdir -p {path} && sudo chown -R {user}:{group} {path}\n")
            sys.exit(1)

    dir_stat = os.stat(path)
    if dir_stat.st_uid != uid:
        try:
            subprocess.check_call(
                ['chown', '-R', f'{user}:{group}', path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print(f"Aligned ownership of {path} to {user}:{group}")
        except (subprocess.CalledProcessError, PermissionError):
            owner = pwd.getpwuid(dir_stat.st_uid).pw_name
            print(f"\nError: {path} is owned by uid {dir_stat.st_uid} ({owner}), not {user}.")
            print(f"Run the following command first, then retry:\n")
            print(f"  sudo chown -R {user}:{group} {path}\n")
            sys.exit(1)

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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """Return a requests session with standard headers."""
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _needs_ssl_bypass(url: str) -> bool:
    """Return True if the URL's host is in the SSL-bypass list."""
    host = urlparse(url).hostname or ""
    return host in SSL_BYPASS_DOMAINS


def _download(session: requests.Session, url: str, dest: str,
              label: str = "", verbose: bool = False) -> bool:
    """Download *url* to *dest* with retries.  Returns True on success."""
    if os.path.exists(dest):
        if verbose:
            print(f"  [skip] {label or url} — already downloaded")
        return True

    verify = not _needs_ssl_bypass(url)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                print(f"  [{attempt}/{MAX_RETRIES}] {label or url}")
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True,
                               verify=verify)
            if resp.status_code == 200:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp = dest + '.tmp'
                with open(tmp, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                os.replace(tmp, dest)
                size_kb = os.path.getsize(dest) / 1024
                if verbose:
                    print(f"    ✓ {size_kb:,.0f} KB")
                return True
            elif resp.status_code == 404:
                if verbose:
                    print(f"    ✗ 404 Not Found")
                return False
            else:
                if verbose:
                    print(f"    ✗ HTTP {resp.status_code}")
                # 503/429 = rate-limited; use longer back-off
                if resp.status_code in (429, 503):
                    time.sleep(RETRY_DELAY * attempt * 3)
        except requests.RequestException as exc:
            if verbose:
                print(f"    ✗ {exc}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return False


# ===================================================================
# PHASE 1 — Download PGN databases
# ===================================================================

def phase1_download_pgn(chess_dir: str, verbose: bool = False) -> dict:
    """Download PGN game databases from free online sources.

    Downloads are saved into $CHESS_DATA/pgn/ with subdirectories:
        pgn/pgnmentor/players/   — individual player collections
        pgn/pgnmentor/events/    — tournament/event collections
        pgn/lumbras/             — Lumbras Gigabase OTB games

    Returns a summary dict with counts.
    """
    print(f"\n{'='*60}")
    print("PHASE 1: Download PGN Databases")
    print(f"{'='*60}\n")

    pgn_dir = os.path.join(chess_dir, 'pgn')
    session = _get_session()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    # --- 1a. PGN Mentor — Player collections ---
    # Scrape files.html for all available player ZIP links, then download
    # every player listed on the page.  Temporal filtering of individual
    # games happens later in Phase 2 using the PGN Date header.
    print("── PGN Mentor: Player collections ──")
    players_dir = os.path.join(pgn_dir, 'pgnmentor', 'players')
    _ensure_directory(players_dir)

    # Build player list from files.html — discover all available ZIPs
    player_stems: list[tuple[str, str]] = []   # (stem, display_name)
    files_html_text: str | None = None
    try:
        verify = not _needs_ssl_bypass("https://www.pgnmentor.com/files.html")
        resp = session.get("https://www.pgnmentor.com/files.html",
                           timeout=REQUEST_TIMEOUT, verify=verify)
        if resp.status_code == 200:
            files_html_text = resp.text
            # Parse all player ZIP links:  href="players/Morphy.zip"
            seen_stems: set[str] = set()
            for m in re.finditer(
                r'href=["\']?players/([A-Za-z][A-Za-z0-9_-]+)\.zip',
                files_html_text,
            ):
                stem = m.group(1)
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                # Try to grab display name from adjacent table cell
                display = stem   # fallback
                player_stems.append((stem, display))
            if verbose:
                print(f"  Scraped {len(player_stems)} player ZIPs from files.html")
        else:
            if verbose:
                print(f"  ✗ Could not fetch files.html (HTTP {resp.status_code})")
    except requests.RequestException as exc:
        if verbose:
            print(f"  ✗ Could not fetch files.html: {exc}")

    if not player_stems:
        print("  ✗ No player ZIPs discovered — cannot proceed with players")
        print("    Check network connectivity to pgnmentor.com")

    items = player_stems
    if not verbose and tqdm:
        items = tqdm(items, desc="Players", unit="file")

    for stem, name in items:
        # PGN Mentor distributes player files as ZIP archives
        dest_pgn = os.path.join(players_dir, f"{stem}.pgn")
        if os.path.exists(dest_pgn):
            stats["skipped"] += 1
            continue

        dest_zip = os.path.join(players_dir, f"{stem}.zip")
        url_zip = f"https://www.pgnmentor.com/players/{stem}.zip"

        downloaded = False
        if _download(session, url_zip, dest_zip, label=f"{name} (zip)",
                     verbose=verbose):
            try:
                with zipfile.ZipFile(dest_zip, 'r') as zf:
                    pgn_members = [m for m in zf.namelist()
                                   if m.lower().endswith('.pgn')]
                    if pgn_members:
                        # Extract PGN file(s) into players_dir
                        zf.extractall(players_dir, members=pgn_members)
                        # Rename first PGN to canonical name if needed
                        extracted = os.path.join(players_dir, pgn_members[0])
                        if extracted != dest_pgn and os.path.exists(extracted):
                            os.replace(extracted, dest_pgn)
                        downloaded = True
                        if verbose:
                            print(f"    Extracted {len(pgn_members)} PGN from ZIP")
                    else:
                        if verbose:
                            print(f"    ✗ ZIP contains no PGN files")
            except zipfile.BadZipFile:
                if verbose:
                    print(f"    ✗ ZIP is corrupt")
                if os.path.exists(dest_zip):
                    os.remove(dest_zip)

        if downloaded:
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

    if hasattr(items, 'close'):
        items.close()
    print(f"  Players: {stats['downloaded']} downloaded, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")

    # --- 1b. PGN Mentor — Event / tournament collections ---
    # Scrape ALL event PGN links from files.html and apply temporal
    # filter — every pre-1969 tournament is in scope.
    print("\n── PGN Mentor: Event collections ──")
    events_dir = os.path.join(pgn_dir, 'pgnmentor', 'events')
    _ensure_directory(events_dir)

    dl, sk, fl = 0, 0, 0

    event_urls: list[tuple[str, str, str]] = []   # (url, dest, label)

    # Reuse page text from player scrape if available, otherwise fetch
    if files_html_text is None:
        try:
            verify = not _needs_ssl_bypass("https://www.pgnmentor.com/files.html")
            resp = session.get("https://www.pgnmentor.com/files.html",
                               timeout=REQUEST_TIMEOUT, verify=verify)
            if resp.status_code == 200:
                files_html_text = resp.text
            else:
                if verbose:
                    print(f"  ✗ Could not fetch files.html (HTTP {resp.status_code})")
        except requests.RequestException as exc:
            if verbose:
                print(f"  ✗ Could not fetch files.html: {exc}")

    if files_html_text:
        # Match every  href="events/<Name><Year><suffix>.pgn"  link
        seen: set[str] = set()
        for m in re.finditer(
            r'href=["\']?(events/([A-Za-z][A-Za-z0-9_-]*?)(\d{4})\w*\.pgn)',
            files_html_text,
        ):
            rel_path = m.group(1)   # e.g. events/MardelPlata1934.pgn
            name_part = m.group(2)  # e.g. MardelPlata
            year_str  = m.group(3)  # e.g. 1934
            if rel_path in seen:
                continue
            seen.add(rel_path)
            try:
                year = int(year_str)
            except ValueError:
                continue
            # Apply temporal filter
            if year > TEMPORAL_CUTOFF_YEAR:
                continue
            fname = os.path.basename(rel_path)
            url = f"https://www.pgnmentor.com/{rel_path}"
            dest = os.path.join(events_dir, fname)
            # Readable label: insert space before year
            label = f"{name_part} {year_str}"
            event_urls.append((url, dest, label))

        if verbose:
            print(f"  Scraped {len(event_urls)} pre-{TEMPORAL_CUTOFF_YEAR} "
                  f"event PGN URLs from files.html")
    else:
        if verbose:
            print("  ✗ files.html unavailable — no events will be downloaded")

    if not event_urls:
        print("  ✗ No event PGN URLs discovered — cannot proceed with events")
        print("    Check network connectivity to pgnmentor.com")

    ev_items = event_urls
    if not verbose and tqdm:
        ev_items = tqdm(ev_items, desc="Events", unit="file")

    for url, dest, label in ev_items:
        if os.path.exists(dest):
            sk += 1
        elif _download(session, url, dest, label=label, verbose=verbose):
            dl += 1
        else:
            fl += 1

    if hasattr(ev_items, 'close'):
        ev_items.close()
    stats["downloaded"] += dl
    stats["skipped"] += sk
    stats["failed"] += fl
    print(f"  Events: {dl} downloaded, {sk} skipped, {fl} failed")

    # --- 1c. Lumbras Gigabase (manual download from MEGA) ---
    # The download page at lumbrasgigabase.com uses a WordPress download
    # manager that redirects to MEGA.  Automated download is not feasible,
    # so we check for manually placed 7z archives and extract them.
    print("\n── Lumbras Gigabase ──")
    lg_dir = os.path.join(pgn_dir, 'lumbras')
    _ensure_directory(lg_dir)

    # Expected 7z archives (manually downloaded from MEGA links on the page)
    LUMBRAS_ARCHIVES = [
        ("LumbrasGigaBase_OTB_0001-1899.7z", "OTB games from year 1 to 1899"),
        ("LumbrasGigaBase_OTB_1900-1949.7z", "OTB games from 1900 to 1949"),
        ("LumbrasGigaBase_OTB_1950-1969.7z", "OTB games from 1950 to 1969"),
    ]

    lg_pgn_glob = os.path.join(lg_dir, '*.pgn')
    if glob.glob(lg_pgn_glob):
        print("  [skip] Lumbras PGN already extracted")
        stats["skipped"] += 1
    else:
        any_found = False
        any_missing = False
        for arc_name, desc in LUMBRAS_ARCHIVES:
            arc_path = os.path.join(lg_dir, arc_name)
            if not os.path.exists(arc_path):
                any_missing = True
                continue
            any_found = True
            try:
                result = subprocess.run(
                    ['7z', 'x', '-y', f'-o{lg_dir}', arc_path],
                    capture_output=True, text=True, timeout=600,
                )
                # Count extracted PGN files
                extracted = [
                    line.split('- ')[-1].strip()
                    for line in result.stdout.splitlines()
                    if line.strip().lower().endswith('.pgn')
                ]
                pgn_count = len(glob.glob(os.path.join(lg_dir, '*.pgn')))
                if result.returncode == 0:
                    print(f"  Extracted PGN from {arc_name} "
                          f"({pgn_count} PGN file(s) in directory)")
                    stats["downloaded"] += 1
                else:
                    print(f"  ✗ 7z extraction failed for {arc_name}")
                    if result.stderr:
                        print(f"    {result.stderr.strip()[:200]}")
                    stats["failed"] += 1
            except FileNotFoundError:
                print("  ✗ '7z' command not found — install p7zip-full:")
                print("    sudo dnf install p7zip-plugins   # Fedora")
                stats["failed"] += 1
                break
            except subprocess.TimeoutExpired:
                print(f"  ✗ Extraction of {arc_name} timed out")
                stats["failed"] += 1

        if any_missing:
            missing = [a for a, _ in LUMBRAS_ARCHIVES
                       if not os.path.exists(os.path.join(lg_dir, a))]
            if not any_found:
                print("  ✗ No Lumbras Gigabase archives found.")
            else:
                print(f"  ✗ Missing {len(missing)} archive(s).")
            print("")
            print("    Manual download required (files are hosted on MEGA):")
            print("    1. Visit: https://lumbrasgigabase.com/en/"
                  "download-in-pgn-format-en/")
            print("    2. Under the 'Downloads OTB' tab, download these files:")
            for a in missing:
                print(f"       • {a}")
            print(f"    3. Place the .7z file(s) in: {lg_dir}/")
            print("    4. Re-run this script (Phase 1) to extract them.")
            print("")
            stats["failed"] += len(missing)

    print(f"\nPhase 1 complete: {stats['downloaded']} downloaded, "
          f"{stats['skipped']} skipped, {stats['failed']} failed")
    return stats


# ===================================================================
# PHASE 2 — Convert PGN to natural-language narrative
# ===================================================================

def _parse_pgn_date(date_str: str) -> int | None:
    """Extract year from a PGN Date header.  Returns None if unparseable."""
    if not date_str or date_str == '????.??.??' or date_str == '?':
        return None
    m = re.match(r'(\d{4})', date_str)
    return int(m.group(1)) if m else None


def _format_result(result: str) -> str:
    """Human-readable game result."""
    mapping = {
        "1-0": "White wins",
        "0-1": "Black wins",
        "1/2-1/2": "Draw",
    }
    return mapping.get(result, result)


def _classify_opening(eco: str, opening: str) -> str:
    """Return a short description of the opening from ECO + name."""
    if opening:
        return opening
    if not eco:
        return ""
    # Broad ECO ranges
    if eco.startswith('A'):
        return "Flank opening"
    if eco.startswith('B'):
        return "Semi-open game"
    if eco.startswith('C'):
        return "Open game"
    if eco.startswith('D'):
        return "Closed game"
    if eco.startswith('E'):
        return "Indian defense"
    return ""


def _game_to_summary(game) -> str | None:
    """Convert a python-chess game to an Approach-A structured summary.

    Returns None if the game lacks sufficient metadata.
    """
    h = game.headers
    white = h.get("White", "Unknown")
    black = h.get("Black", "Unknown")
    event = h.get("Event", "?")
    date = h.get("Date", "?")
    result = h.get("Result", "*")
    eco = h.get("ECO", "")
    opening = h.get("Opening", "")

    # Skip games with no player names
    if white in ("?", "Unknown", "") and black in ("?", "Unknown", ""):
        return None

    lines = []
    lines.append(f"Game: {white} vs. {black}")

    if event and event != '?':
        date_display = date.replace('.??', '').replace('??', '').rstrip('.')
        lines.append(f"Event: {event}, {date_display}")
    elif date and date != '????.??.??':
        lines.append(f"Date: {date}")

    opening_desc = _classify_opening(eco, opening)
    if opening_desc:
        eco_tag = f" ({eco})" if eco else ""
        lines.append(f"Opening: {opening_desc}{eco_tag}")

    lines.append(f"Result: {_format_result(result)}")
    lines.append("")

    # Collect moves
    board = game.board()
    move_strs = []
    move_num = 1
    for node in game.mainline():
        move = node.move
        san = board.san(move)
        if board.turn == chess.WHITE:
            move_strs.append(f"{move_num}.{san}")
        else:
            move_strs.append(san)
            move_num += 1
        board.push(move)

    if not move_strs:
        return None

    # Format moves in lines of ~80 chars
    move_text = ""
    line = ""
    for ms in move_strs:
        if len(line) + len(ms) + 1 > 78:
            move_text += line.rstrip() + "\n"
            line = ""
        line += ms + " "
    if line.strip():
        move_text += line.rstrip()

    lines.append(move_text)

    total_moves = (len(list(game.mainline())) + 1) // 2
    lines.append("")
    lines.append(f"The game lasted {total_moves} moves and ended in "
                 f"{_format_result(result).lower()}.")

    return "\n".join(lines)


def _game_to_annotated(game) -> str | None:
    """Convert a python-chess game to an Approach-C annotated narrative.

    Provides richer description for famous or well-annotated games.
    Returns None if game is unsuitable.
    """
    h = game.headers
    white = h.get("White", "Unknown")
    black = h.get("Black", "Unknown")
    event = h.get("Event", "?")
    date = h.get("Date", "?")
    result = h.get("Result", "*")
    eco = h.get("ECO", "")
    opening = h.get("Opening", "")

    if white in ("?", "Unknown", "") and black in ("?", "Unknown", ""):
        return None

    lines = []
    date_display = date.replace('.??', '').replace('??', '').rstrip('.')
    event_str = f" at {event}" if event and event != '?' else ""
    lines.append(f"{white} vs. {black}{event_str} ({date_display})")
    lines.append("")

    opening_desc = _classify_opening(eco, opening)
    if opening_desc:
        eco_tag = f" ({eco})" if eco else ""
        lines.append(f"Opening: {opening_desc}{eco_tag}")
        lines.append("")

    # Walk the game with commentary at key moments
    board = game.board()
    move_num = 1
    move_buffer = []
    commentary_lines = []
    material_changes = 0

    for node in game.mainline():
        move = node.move
        san = board.san(move)

        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)
        is_castling = board.is_castling(move)

        if board.turn == chess.WHITE:
            move_str = f"{move_num}.{san}"
        else:
            move_str = san
            move_num += 1

        move_buffer.append(move_str)

        # Add inline commentary for significant moves
        if is_capture:
            material_changes += 1
        if gives_check:
            move_buffer[-1] += "  — check!"
        if is_castling:
            side = "kingside" if board.is_kingside_castling(move) else "queenside"
            move_buffer[-1] += f"  — {side} castling"

        board.push(move)

        # Flush move buffer every 6-8 full moves
        if move_num % 7 == 0 and board.turn == chess.WHITE and move_buffer:
            commentary_lines.append(" ".join(move_buffer))
            move_buffer = []

    # Flush remaining moves
    if move_buffer:
        commentary_lines.append(" ".join(move_buffer))

    lines.append("\n".join(commentary_lines))
    lines.append("")

    total_moves = (len(list(game.mainline())) + 1) // 2
    result_desc = _format_result(result).lower()
    lines.append(f"After {total_moves} moves the game ended in {result_desc}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2 — parsing helpers (timeout, stuck detection, skip-ahead)
# ---------------------------------------------------------------------------

class _GameParseTimeout(Exception):
    """Raised when a single chess.pgn.read_game() exceeds the time limit."""
    pass


def _alarm_handler(signum, frame):
    raise _GameParseTimeout()


def _skip_to_next_game(pgn_f) -> bool:
    """Advance *pgn_f* past a malformed game to the next ``[Event`` header.

    Returns True if a new game header was found, False on EOF.
    """
    while True:
        pos = pgn_f.tell()
        line = pgn_f.readline()
        if not line:
            return False
        if line.startswith('[Event '):
            pgn_f.seek(pos)  # back up so read_game sees the header
            return True


def _compute_file_timeout(pgn_path: str) -> int:
    """Return a dynamic file-parse timeout (seconds) scaled by file size.

    Small files get the base FILE_PARSE_TIMEOUT.  Larger files receive
    an additional FILE_TIMEOUT_PER_MB seconds per megabyte so that
    multi-hundred-MB PGN archives are not prematurely abandoned.
    """
    try:
        size_mb = os.path.getsize(pgn_path) / (1024 * 1024)
    except OSError:
        size_mb = 0
    return max(FILE_PARSE_TIMEOUT,
               int(FILE_PARSE_TIMEOUT + size_mb * FILE_TIMEOUT_PER_MB))


def _process_pgn_file(
    pgn_path: str,
    pgn_dir: str,
    cutoff_year: int,
    known_keys: set | frozenset,
    annotated_budget: int = 0,
    game_timeout: int = GAME_PARSE_TIMEOUT,
    file_timeout: int | None = None,
) -> tuple[list[dict], dict]:
    """Process one PGN file with safety timeouts and stuck detection.

    Parameters
    ----------
    pgn_path : path to the PGN file
    pgn_dir  : base PGN directory (for relative-path calculation)
    cutoff_year : skip games with a Date header after this year
    known_keys : keys already in the corpus (for dedup)
    annotated_budget : how many games to convert in annotated mode
    game_timeout : max seconds per ``read_game()`` call
    file_timeout : max wall-clock seconds for the entire file

    Returns
    -------
    (records, stats) — *records* is a list of dicts ready for JSONL output,
    *stats* is a dict of per-file counters.
    """
    if file_timeout is None:
        file_timeout = _compute_file_timeout(pgn_path)

    rel_path = os.path.relpath(pgn_path, pgn_dir)
    records: list[dict] = []
    stats = {
        "games_total": 0,
        "games_pre_cutoff": 0,
        "games_post_cutoff": 0,
        "games_skipped_dup": 0,
        "games_skipped_nodata": 0,
        "games_timed_out": 0,
        "games_parse_error": 0,
        "annotated_count": 0,
        "file_timed_out": False,
        "file_timeout_used": file_timeout,
        "consec_error_skip": False,
        "error": None,
    }

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)

    file_start = time.monotonic()
    consecutive_errors = 0
    annotated_left = annotated_budget
    local_keys: set[str] = set()   # intra-file dedup

    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='replace') as pgn_f:
            while True:
                # Per-file wall-clock check
                if time.monotonic() - file_start > file_timeout:
                    stats["file_timed_out"] = True
                    break

                # Read one game with alarm-based timeout
                game = None
                file_pos = pgn_f.tell()
                signal.alarm(game_timeout)
                try:
                    game = chess.pgn.read_game(pgn_f)
                except _GameParseTimeout:
                    stats["games_timed_out"] += 1
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        stats["consec_error_skip"] = True
                        break
                    _skip_to_next_game(pgn_f)
                    continue
                except Exception:
                    stats["games_parse_error"] += 1
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        stats["consec_error_skip"] = True
                        break
                    # Ensure file pointer advanced; if stuck, skip ahead
                    if pgn_f.tell() == file_pos:
                        if not _skip_to_next_game(pgn_f):
                            break
                    continue
                finally:
                    signal.alarm(0)

                if game is None:
                    break

                consecutive_errors = 0
                stats["games_total"] += 1
                h = game.headers

                # --- Temporal filter ---
                year = _parse_pgn_date(h.get("Date", ""))
                if year is not None and year > cutoff_year:
                    stats["games_post_cutoff"] += 1
                    continue
                # Games with no date: include (many historical games lack dates)

                stats["games_pre_cutoff"] += 1

                # --- Dedup key ---
                key = (f"{h.get('White','')}-{h.get('Black','')}-"
                       f"{h.get('Date','')}-{h.get('Event','')}-"
                       f"{h.get('Round','')}")
                if key in known_keys or key in local_keys:
                    stats["games_skipped_dup"] += 1
                    continue

                # --- Convert ---
                if annotated_left > 0:
                    text = _game_to_annotated(game)
                    mode = "annotated"
                else:
                    text = _game_to_summary(game)
                    mode = "summary"

                if text is None:
                    stats["games_skipped_nodata"] += 1
                    continue

                record = {
                    "key": key,
                    "white": h.get("White", ""),
                    "black": h.get("Black", ""),
                    "date": h.get("Date", ""),
                    "event": h.get("Event", ""),
                    "eco": h.get("ECO", ""),
                    "opening": h.get("Opening", ""),
                    "result": h.get("Result", ""),
                    "source_file": rel_path,
                    "mode": mode,
                    "text": text,
                    "length": len(text),
                }
                records.append(record)
                local_keys.add(key)
                if mode == "annotated":
                    stats["annotated_count"] += 1
                    annotated_left -= 1

    except Exception as exc:
        stats["error"] = str(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    return records, stats


def phase2_convert_pgn(chess_dir: str, verbose: bool = False,
                       annotated_limit: int = 1000,
                       workers: int = 1) -> dict:
    """Parse all downloaded PGN files, filter to pre-1969, convert to text.

    Writes JSONL output to $CHESS_DATA/corpus/chess_games.jsonl.
    Uses Approach C (annotated) for the first *annotated_limit* games
    and Approach A (summary) for the rest.

    When *workers* > 1, files are processed in parallel using a process
    pool.  Annotated mode is disabled in parallel (all summary).

    Requires the ``python-chess`` library.

    Returns a summary dict.
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Convert PGN to Narrative Text")
    print(f"{'='*60}\n")

    if not HAS_PYTHON_CHESS:
        print("ERROR: python-chess is required for Phase 2.")
        print("Install it with:  pip install python-chess")
        return {"error": "python-chess not installed"}

    pgn_dir = os.path.join(chess_dir, 'pgn')
    corpus_dir = os.path.join(chess_dir, 'corpus')
    _ensure_directory(corpus_dir)
    output_file = os.path.join(corpus_dir, 'chess_games.jsonl')
    manifest_file = os.path.join(corpus_dir, 'chess_games.manifest.json')

    # Collect all PGN files, sorted largest-first for better parallel
    # scheduling and early detection of problematic files.
    pgn_files_raw = glob.glob(os.path.join(pgn_dir, '**', '*.pgn'),
                              recursive=True)
    if not pgn_files_raw:
        print("No PGN files found. Run Phase 1 first.")
        return {"error": "no PGN files", "games": 0}

    pgn_files_all = sorted(pgn_files_raw,
                           key=lambda p: os.path.getsize(p), reverse=True)
    total_size = sum(os.path.getsize(f) for f in pgn_files_all)
    print(f"Found {len(pgn_files_all)} PGN file(s), "
          f"{total_size / (1024*1024):.1f} MB total")
    for f in pgn_files_all[:5]:
        sz = os.path.getsize(f)
        print(f"  {os.path.relpath(f, pgn_dir):40s} {sz / (1024*1024):6.1f} MB")
    if len(pgn_files_all) > 5:
        print(f"  ... and {len(pgn_files_all) - 5} more files")

    # Load existing game keys to support idempotent re-runs
    existing_keys: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_keys.add(rec.get('key', ''))
                    except json.JSONDecodeError:
                        pass
        print(f"Existing corpus has {len(existing_keys)} games "
              f"— will skip duplicates")

    # Load manifest of previously-processed PGN files.  A file is
    # considered "done" if it was fully processed (no timeout / error)
    # and its size + mtime haven't changed on disk.
    manifest: dict[str, dict] = {}
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}

    def _is_already_processed(pgn_path: str) -> bool:
        """Return True if *pgn_path* is in the manifest and unchanged."""
        rel = os.path.relpath(pgn_path, pgn_dir)
        entry = manifest.get(rel)
        if entry is None or not entry.get("completed"):
            return False
        try:
            st = os.stat(pgn_path)
            return (st.st_size == entry.get("size")
                    and st.st_mtime == entry.get("mtime"))
        except OSError:
            return False

    # Filter out files that are already fully processed
    pgn_files = [p for p in pgn_files_all if not _is_already_processed(p)]
    skipped_manifest = len(pgn_files_all) - len(pgn_files)
    if skipped_manifest:
        print(f"Skipping {skipped_manifest} already-processed file(s) "
              f"(manifest match)")
    if not pgn_files and skipped_manifest:
        print("All PGN files already processed — nothing to do.")
        return {"files_processed": 0, "games_written": 0,
                "skipped_manifest": skipped_manifest}

    def _manifest_entry(pgn_path: str, fstats: dict) -> dict:
        """Build a manifest entry for a successfully-processed file."""
        st = os.stat(pgn_path)
        return {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "games_total": fstats["games_total"],
            "completed": (not fstats["file_timed_out"]
                          and not fstats["consec_error_skip"]
                          and fstats["error"] is None),
        }

    def _save_manifest() -> None:
        """Persist the manifest to disk."""
        try:
            tmp = manifest_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=1)
            os.replace(tmp, manifest_file)
        except OSError as exc:
            print(f"  ⚠ Could not save manifest: {exc}")

    stats = {
        "files_processed": 0,
        "games_total": 0,
        "games_pre1969": 0,
        "games_written": 0,
        "games_skipped_dup": 0,
        "games_skipped_nodata": 0,
        "games_post_cutoff": 0,
        "annotated_count": 0,
        "games_timed_out": 0,
        "games_parse_error": 0,
        "files_timed_out": 0,
        "files_consec_error": 0,
    }

    use_parallel = workers > 1 and len(pgn_files) > 1

    if use_parallel:
        print(f"\nParallel mode: {workers} worker(s), "
              f"game timeout {GAME_PARSE_TIMEOUT}s, "
              f"file timeout {FILE_PARSE_TIMEOUT}s base "
              f"+ {FILE_TIMEOUT_PER_MB}s/MB")
        if annotated_limit > 0:
            print("  Note: annotated mode disabled in parallel "
                  "(all games use summary mode)")
    else:
        print(f"\nSequential mode, game timeout {GAME_PARSE_TIMEOUT}s, "
              f"file timeout {FILE_PARSE_TIMEOUT}s base "
              f"+ {FILE_TIMEOUT_PER_MB}s/MB")

    # --- Helper to merge per-file stats into the aggregate dict ---
    def _merge_stats(fstats: dict, rel: str) -> None:
        stats["games_total"] += fstats["games_total"]
        stats["games_pre1969"] += fstats["games_pre_cutoff"]
        stats["games_post_cutoff"] += fstats["games_post_cutoff"]
        stats["games_skipped_dup"] += fstats["games_skipped_dup"]
        stats["games_skipped_nodata"] += fstats["games_skipped_nodata"]
        stats["games_timed_out"] += fstats["games_timed_out"]
        stats["games_parse_error"] += fstats["games_parse_error"]
        if fstats["file_timed_out"]:
            stats["files_timed_out"] += 1
            ft_used = fstats.get("file_timeout_used", FILE_PARSE_TIMEOUT)
            print(f"  ⚠ {rel}: file timeout after {ft_used}s")
        if fstats["consec_error_skip"]:
            stats["files_consec_error"] += 1
            print(f"  ⚠ {rel}: skipped — {MAX_CONSECUTIVE_ERRORS} "
                  f"consecutive parse errors")
        if fstats["error"]:
            print(f"  ✗ {rel}: {fstats['error']}")

    # ---------------------------------------------------------------
    # PARALLEL path
    # ---------------------------------------------------------------
    if use_parallel:
        frozen_keys = frozenset(existing_keys)
        completed = 0

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_pgn_file,
                    p, pgn_dir, TEMPORAL_CUTOFF_YEAR,
                    frozen_keys, 0,              # no annotated in parallel
                    GAME_PARSE_TIMEOUT,          # per-game timeout
                    _compute_file_timeout(p),    # dynamic per-file timeout
                ): p
                for p in pgn_files
            }

            pbar = None
            if not verbose and tqdm:
                pbar = tqdm(total=len(pgn_files),
                            desc="PGN files", unit="file")

            with open(output_file, 'a', encoding='utf-8') as out:
                for future in as_completed(futures):
                    pgn_path = futures[future]
                    rel = os.path.relpath(pgn_path, pgn_dir)
                    try:
                        records, fstats = future.result()
                    except Exception as exc:
                        print(f"  ✗ Worker error on {rel}: {exc}")
                        if pbar:
                            pbar.update(1)
                        continue

                    _merge_stats(fstats, rel)

                    # Write records, dedup against cross-file duplicates
                    for rec in records:
                        if rec["key"] not in existing_keys:
                            out.write(
                                json.dumps(rec, ensure_ascii=False) + '\n')
                            existing_keys.add(rec["key"])
                            stats["games_written"] += 1

                    # Update manifest for this file
                    entry = _manifest_entry(pgn_path, fstats)
                    manifest[rel] = entry

                    stats["files_processed"] += 1
                    completed += 1

                    if verbose:
                        sz = os.path.getsize(pgn_path) / (1024 * 1024)
                        elapsed = fstats.get("games_total", 0)
                        print(f"  [{completed}/{len(pgn_files)}] {rel} "
                              f"({sz:.1f} MB): {elapsed} games, "
                              f"{len(records)} new records")
                    if pbar:
                        pbar.update(1)

            if pbar:
                pbar.close()

    # ---------------------------------------------------------------
    # SEQUENTIAL path (default)
    # ---------------------------------------------------------------
    else:
        annotated_remaining = annotated_limit

        with open(output_file, 'a', encoding='utf-8') as out:
            pbar = pgn_files
            if not verbose and tqdm:
                pbar = tqdm(pgn_files, desc="PGN files", unit="file")

            for pgn_path in pbar:
                rel_path = os.path.relpath(pgn_path, pgn_dir)
                file_size_mb = os.path.getsize(pgn_path) / (1024 * 1024)

                if verbose:
                    print(f"\nProcessing: {rel_path} ({file_size_mb:.1f} MB)")

                records, fstats = _process_pgn_file(
                    pgn_path, pgn_dir, TEMPORAL_CUTOFF_YEAR,
                    existing_keys, annotated_remaining,
                    GAME_PARSE_TIMEOUT,
                    _compute_file_timeout(pgn_path),
                )

                _merge_stats(fstats, rel_path)

                # Write records
                for rec in records:
                    if rec["key"] not in existing_keys:
                        out.write(
                            json.dumps(rec, ensure_ascii=False) + '\n')
                        existing_keys.add(rec["key"])
                        stats["games_written"] += 1
                        if rec["mode"] == "annotated":
                            stats["annotated_count"] += 1

                annotated_remaining = (annotated_limit
                                       - stats["annotated_count"])

                # Update manifest for this file
                entry = _manifest_entry(pgn_path, fstats)
                manifest[os.path.relpath(pgn_path, pgn_dir)] = entry

                stats["files_processed"] += 1

                if verbose and records:
                    print(f"  → {len(records)} new records "
                          f"({fstats['games_total']} games parsed)")

            if hasattr(pbar, 'close'):
                pbar.close()

    # Persist the manifest so subsequent runs can skip processed files
    _save_manifest()

    print(f"\nPhase 2 complete:")
    if skipped_manifest:
        print(f"  PGN files skipped    : {skipped_manifest} (already processed)")
    print(f"  PGN files processed  : {stats['files_processed']}")
    print(f"  Total games parsed   : {stats['games_total']:,}")
    print(f"  Pre-1969 games       : {stats['games_pre1969']:,}")
    print(f"  Games written        : {stats['games_written']:,}")
    print(f"    (annotated mode)   : {stats['annotated_count']:,}")
    print(f"    (summary mode)     : "
          f"{stats['games_written'] - stats['annotated_count']:,}")
    print(f"  Skipped (duplicate)  : {stats['games_skipped_dup']:,}")
    print(f"  Skipped (no data)    : {stats['games_skipped_nodata']:,}")
    print(f"  Post-1969 filtered   : {stats['games_post_cutoff']:,}")
    if stats["games_timed_out"]:
        print(f"  Games timed out      : {stats['games_timed_out']:,}")
    if stats["games_parse_error"]:
        print(f"  Games parse errors   : {stats['games_parse_error']:,}")
    if stats["files_timed_out"]:
        print(f"  Files timed out      : {stats['files_timed_out']}")
    if stats["files_consec_error"]:
        print(f"  Files skipped (errs) : {stats['files_consec_error']}")
    print(f"  Output: {output_file}")
    return stats


# ===================================================================
# PHASE 3 — Internet Archive chess books
# ===================================================================

def _download_ia_text(session: requests.Session, identifier: str,
                      dest: str, verbose: bool = False) -> bool:
    """Download the plain-text version of an Internet Archive item.

    Tries multiple strategies:
    1. Metadata API to find the _djvu.txt file name, then download it
    2. Direct {identifier}_djvu.txt URL (common convention)
    3. Stream endpoint fallback
    """
    base = "https://archive.org"

    # Strategy 1: metadata API to find the actual text file name
    meta_url = f"{base}/metadata/{identifier}"
    try:
        resp = session.get(meta_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get('files', [])
            # Prefer _djvu.txt files (OCR text)
            for f in files:
                name = f.get('name', '')
                if name.endswith('_djvu.txt'):
                    from urllib.parse import quote
                    txt_url = f"{base}/download/{identifier}/{quote(name)}"
                    if _download(session, txt_url, dest,
                                 label=f"{identifier}/{name}", verbose=verbose):
                        return True
            # Fall back to any .txt file
            for f in files:
                name = f.get('name', '')
                if name.endswith('.txt') and not name.startswith('__'):
                    from urllib.parse import quote
                    txt_url = f"{base}/download/{identifier}/{quote(name)}"
                    if _download(session, txt_url, dest,
                                 label=f"{identifier}/{name}", verbose=verbose):
                        return True
    except Exception:
        pass

    # Strategy 2: direct DjVu text URL (common naming convention)
    djvu_url = f"{base}/download/{identifier}/{identifier}_djvu.txt"
    if _download(session, djvu_url, dest, label=f"{identifier} (DjVu text)",
                 verbose=verbose):
        return True

    # Strategy 3: Stream endpoint (full text)
    stream_url = f"{base}/stream/{identifier}/{identifier}_djvu.txt"
    if _download(session, stream_url, dest, label=f"{identifier} (stream)",
                 verbose=verbose):
        return True

    if verbose:
        print(f"  ✗ Could not find text for {identifier}")
    return False


def phase3_internet_archive(chess_dir: str, verbose: bool = False) -> dict:
    """Download public-domain chess books from Internet Archive.

    Saves raw text into $CHESS_DATA/archive/ and converts to JSONL
    in $CHESS_DATA/corpus/chess_archive_books.jsonl.

    Returns a summary dict.
    """
    print(f"\n{'='*60}")
    print("PHASE 3: Internet Archive Chess Books")
    print(f"{'='*60}\n")

    archive_dir = os.path.join(chess_dir, 'archive')
    corpus_dir = os.path.join(chess_dir, 'corpus')
    _ensure_directory(archive_dir)
    _ensure_directory(corpus_dir)
    output_file = os.path.join(corpus_dir, 'chess_archive_books.jsonl')

    # Load existing to support idempotency
    existing_ids: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_ids.add(rec.get('identifier', ''))
                    except json.JSONDecodeError:
                        pass
        print(f"Existing corpus has {len(existing_ids)} books — will skip duplicates")

    session = _get_session()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "written": 0}

    items = ARCHIVE_CHESS_BOOKS
    if not verbose and tqdm:
        items = tqdm(items, desc="IA books", unit="book")

    with open(output_file, 'a', encoding='utf-8') as out:
        for idx, (identifier, title, author, year) in enumerate(items):
            if verbose:
                print(f"\n{title} by {author} ({year})")

            # Rate-limit: pause between IA requests to avoid 503/429
            if idx > 0:
                time.sleep(3)

            if identifier in existing_ids:
                stats["skipped"] += 1
                if verbose:
                    print("  [skip] already in corpus")
                continue

            txt_path = os.path.join(archive_dir, f"{identifier}.txt")
            if _download_ia_text(session, identifier, txt_path, verbose=verbose):
                stats["downloaded"] += 1

                # Read and clean text
                try:
                    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
                        raw_text = f.read()

                    # Basic OCR cleanup
                    text = raw_text
                    # Collapse excessive whitespace but keep paragraph breaks
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = text.strip()

                    if len(text) < 500:
                        if verbose:
                            print(f"  ✗ Text too short ({len(text)} chars) — skipping")
                        stats["failed"] += 1
                        continue

                    record = {
                        "identifier": identifier,
                        "title": title,
                        "author": author,
                        "pub_year": year,
                        "text": text,
                        "length": len(text),
                        "source": "internet_archive",
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats["written"] += 1
                    existing_ids.add(identifier)

                    if verbose:
                        print(f"  ✓ {len(text):,} characters")
                except Exception as exc:
                    print(f"  ✗ Error processing {identifier}: {exc}")
                    stats["failed"] += 1
            else:
                stats["failed"] += 1

    if hasattr(items, 'close'):
        items.close()

    print(f"\nPhase 3 complete:")
    print(f"  Downloaded : {stats['downloaded']}")
    print(f"  Written    : {stats['written']}")
    print(f"  Skipped    : {stats['skipped']}")
    print(f"  Failed     : {stats['failed']}")
    print(f"  Output: {output_file}")
    return stats


# ===================================================================
# Status / Info
# ===================================================================

def show_status(chess_dir: str) -> None:
    """Print a summary of the current chess data directory."""
    print(f"\n{'='*60}")
    print("CHESS CONTENT STATUS")
    print(f"{'='*60}")
    print(f"Data directory: {chess_dir}")

    # PGN files
    pgn_dir = os.path.join(chess_dir, 'pgn')
    if os.path.isdir(pgn_dir):
        pgn_files = glob.glob(os.path.join(pgn_dir, '**', '*.pgn'),
                              recursive=True)
        total_size = sum(os.path.getsize(f) for f in pgn_files)
        print(f"\nPGN files: {len(pgn_files)} files, "
              f"{total_size / (1024*1024):.1f} MB")

        for subdir in ['pgnmentor/players', 'pgnmentor/events',
                       'lumbras']:
            sd = os.path.join(pgn_dir, subdir)
            if os.path.isdir(sd):
                files = glob.glob(os.path.join(sd, '*.pgn'))
                if files:
                    sz = sum(os.path.getsize(f) for f in files)
                    print(f"  {subdir}: {len(files)} files, "
                          f"{sz / (1024*1024):.1f} MB")
    else:
        print("\nPGN files: (none — run Phase 1)")

    # Archive texts
    archive_dir = os.path.join(chess_dir, 'archive')
    if os.path.isdir(archive_dir):
        txt_files = glob.glob(os.path.join(archive_dir, '*.txt'))
        if txt_files:
            total_size = sum(os.path.getsize(f) for f in txt_files)
            print(f"\nArchive texts: {len(txt_files)} files, "
                  f"{total_size / (1024*1024):.1f} MB")
    else:
        print("\nArchive texts: (none — run Phase 3)")

    # Corpus output
    corpus_dir = os.path.join(chess_dir, 'corpus')
    if os.path.isdir(corpus_dir):
        print(f"\nCorpus (post-processed):")
        for name in sorted(os.listdir(corpus_dir)):
            if name.endswith('.jsonl'):
                path = os.path.join(corpus_dir, name)
                size = os.path.getsize(path)
                count = 0
                total_chars = 0
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                r = json.loads(line)
                                count += 1
                                total_chars += r.get('length', 0)
                            except json.JSONDecodeError:
                                pass
                print(f"  {name}: {count:,} records, "
                      f"{size / (1024*1024):.1f} MB, "
                      f"{total_chars:,} chars")
    else:
        print("\nCorpus: (none — run Phase 2/3)")

    print(f"\n{'='*60}")


def show_info() -> None:
    """Display source lists and configuration."""
    print(f"\n{'='*60}")
    print("CHESS CONTENT RETRIEVER — SOURCE INFORMATION")
    print(f"Temporal cutoff: pre-{TEMPORAL_CUTOFF_YEAR}")
    print(f"{'='*60}\n")

    print("PHASE 1: PGN SOURCES")
    print("-" * 60)
    print(f"\nPGN Mentor — Players (all discovered from files.html)")
    print(f"  All player ZIP archives are downloaded; temporal filtering")
    print(f"  of individual games is applied in Phase 2 using the PGN")
    print(f"  Date header (cutoff: {TEMPORAL_CUTOFF_YEAR}).")
    print(f"\nPGN Mentor — Events (all pre-{TEMPORAL_CUTOFF_YEAR} from files.html)")
    print(f"  All event PGN links with year ≤ {TEMPORAL_CUTOFF_YEAR} in the URL")
    print(f"  are downloaded automatically.")
    print(f"\nLumbras Gigabase — OTB games (pre-{TEMPORAL_CUTOFF_YEAR})")
    print(f"  Three ZIP sets covering years 0001–1899, 1900–1949, 1950–1969.")
    print(f"  Manual download from MEGA required — see Phase 1 output.")
    print(f"  Source: https://lumbrasgigabase.com/en/download-in-pgn-format-en/")

    print(f"\n{'='*60}")
    print("PHASE 3: INTERNET ARCHIVE BOOKS")
    print("-" * 60)
    for ident, title, author, year in ARCHIVE_CHESS_BOOKS:
        print(f"  {year}  {title}")
        print(f"        {author} — archive.org/details/{ident}")

    print(f"\n{'='*60}")
    print("GUTENBERG CHESS BOOKS (handled by retrieve_gutenberg.py)")
    print("-" * 60)
    print("  12 chess books in the 'Chess & Strategy' category")
    print("  Run: python scripts/retrieve_gutenberg.py --priority-only")

    print(f"\n{'='*60}\n")


# ===================================================================
# Main entry point
# ===================================================================

def main():
    """Main entry point with phase-based CLI."""
    import argparse

    chess_data = os.environ.get('CHESS_DATA')
    if chess_data:
        default_dir = chess_data
        print(f"Using CHESS_DATA environment variable: {chess_data}")
    else:
        default_dir = '/mnt/data/chess'
        print("Warning: CHESS_DATA environment variable not set. "
              f"Using default: {default_dir}")

    parser = argparse.ArgumentParser(
        description="Retrieve and convert chess content for training-data preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase-based operation — run phases individually or all at once:

  python retrieve_chess_content.py                # Run all phases (1, 2, 3)
  python retrieve_chess_content.py --phase 1      # Download PGN databases only
  python retrieve_chess_content.py --phase 2      # Convert PGN → text only
  python retrieve_chess_content.py --phase 3      # Download Internet Archive books only
  python retrieve_chess_content.py --status        # Show current data status
  python retrieve_chess_content.py --info          # List all sources

Phase 2 parallel mode (useful for large PGN collections):
  python retrieve_chess_content.py --phase 2 --workers 4

Phase 2 includes automatic stuck detection: per-game timeout (30s),
per-file timeout (300s), and consecutive-error limits (10). Files are
processed largest-first so the biggest files start early in parallel mode.

Gutenberg chess books are handled separately:
  python retrieve_gutenberg.py --priority-only    # Downloads 12 chess books
        """
    )
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run only the specified phase (default: all)')
    parser.add_argument('--chess-dir', default=default_dir,
                        help=f'Base directory for chess data (default: {default_dir})')
    parser.add_argument('--status', action='store_true',
                        help='Show status of existing chess data and exit')
    parser.add_argument('--info', action='store_true',
                        help='Display source lists and configuration, then exit')
    parser.add_argument('--reset', action='store_true',
                        help='Delete corpus output files and start fresh')
    parser.add_argument('--annotated-limit', type=int, default=1000,
                        help='Number of games to convert in annotated mode (default: 1000)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for Phase 2 (default: 1 = sequential)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-item output')

    args = parser.parse_args()

    # --info
    if args.info:
        show_info()
        return

    # --status
    if args.status:
        show_status(args.chess_dir)
        return

    # Ensure base directory exists
    _ensure_directory(args.chess_dir)

    # --reset
    if args.reset:
        corpus_dir = os.path.join(args.chess_dir, 'corpus')
        if os.path.isdir(corpus_dir):
            for f in glob.glob(os.path.join(corpus_dir, '*.jsonl')):
                os.remove(f)
                print(f"Deleted: {f}")
            manifest = os.path.join(corpus_dir,
                                    'chess_games.manifest.json')
            if os.path.exists(manifest):
                os.remove(manifest)
                print(f"Deleted: {manifest}")
        print("Corpus reset complete.\n")

    verbose = args.verbose
    if not verbose and tqdm is None:
        print("Note: install 'tqdm' for progress bars (pip install tqdm). "
              "Falling back to verbose output.")
        verbose = True

    start_time = time.time()

    print(f"\n{'='*60}")
    print("Chess Content Retriever")
    print(f"Temporal cutoff: pre-{TEMPORAL_CUTOFF_YEAR}")
    print(f"Chess directory: {args.chess_dir}")
    print(f"{'='*60}")

    phases_to_run = [args.phase] if args.phase else [1, 2, 3]
    results = {}

    if 1 in phases_to_run:
        results['phase1'] = phase1_download_pgn(args.chess_dir, verbose=verbose)

    if 2 in phases_to_run:
        results['phase2'] = phase2_convert_pgn(
            args.chess_dir, verbose=verbose,
            annotated_limit=args.annotated_limit,
            workers=args.workers)

    if 3 in phases_to_run:
        results['phase3'] = phase3_internet_archive(args.chess_dir, verbose=verbose)

    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = elapsed % 60

    print(f"\n{'='*60}")
    print("RETRIEVAL COMPLETE")
    print(f"{'='*60}")
    print(f"Phases run : {', '.join(str(p) for p in phases_to_run)}")
    print(f"Runtime    : {mins}m {secs:.1f}s")
    print(f"Data dir   : {args.chess_dir}")
    print(f"Corpus dir : {os.path.join(args.chess_dir, 'corpus')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
