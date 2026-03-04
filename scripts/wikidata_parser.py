#!/usr/bin/env python3
"""
Wikidata TTL Parser - Download, extract, parse, compress, and reclaim

This script manages the full Wikidata temporal-data pipeline through five
explicit stages:

  1. Download  - fetch the Wikidata TTL bz2 dump  (with resume support)
  2. Extract   - decompress the bz2 to a plain TTL file
  3. Parse     - extract temporal properties and write plain CSV
  4. Compress  - zstd-compress the CSV for downstream streaming
  5. Reclaim   - delete intermediate files, leaving .reclaim markers

Every stage checks whether its output file already exists *or* whether
a ``.reclaim`` marker file indicates the output was previously produced
and then cleaned up.  This means the pipeline can be re-run at any time
and will pick up where it left off.

After a full run the working directory contains:
  - ``wikidata-temporal.csv.zst``                      (final compressed output)
  - ``wikidata-YYYYMMDD-all-BETA.ttl.bz2.reclaim``    (marker: bz2 was downloaded)
  - ``wikidata-YYYYMMDD-all-BETA.ttl.reclaim``         (marker: TTL was extracted)
  - ``wikidata-temporal.csv.reclaim``                   (marker: CSV was produced)

Pass ``--force`` to clear all ``.reclaim`` markers and redo every stage.

Full dataset: https://dumps.wikimedia.org/wikidatawiki/entities/

Environment variables (from deepred-env.sh):
  WIKI_DATA   - root data directory (default: /mnt/data/wikipedia)
"""

import argparse
import bz2
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKIDATA_URL = (
    "https://dumps.wikimedia.org/wikidatawiki/entities/"
    "20251215/wikidata-20251215-all-BETA.ttl.bz2"
)

# Default output filenames (relative to <wikidata_dir>)
DEFAULT_CSV_NAME = "wikidata-temporal.csv"

DOWNLOAD_CHUNK = 1 << 20  # 1 MiB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sizeof_fmt(num: float) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:,.1f} {unit}"
        num /= 1024.0
    return f"{num:,.1f} PB"


def _compress_existing(src: Path, dst: Path, verbose: bool = False) -> None:
    """Compress an existing plain file to zstd without re-parsing.

    Falls back to the ``zstd`` CLI (via sudo if needed) when the
    destination directory is not writable by the current user.
    """
    src_size = src.stat().st_size
    print(
        f"  Compressing {src.name} \u2192 {dst.name} "
        f"({_sizeof_fmt(src_size)}) \u2026"
    )

    # --- fast-path: try writing directly via the zstandard library --------
    try:
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        part = dst.with_suffix(dst.suffix + ".part")
        with open(src, "rb") as fin, open(part, "wb") as fout:
            if verbose:
                with tqdm(
                    total=src_size,
                    unit="B",
                    unit_scale=True,
                    desc="  zstd",
                ) as pbar:
                    writer = cctx.stream_writer(fout)
                    while True:
                        chunk = fin.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        writer.write(chunk)
                        pbar.update(len(chunk))
                    writer.close()
            else:
                cctx.copy_stream(fin, fout)
        part.rename(dst)
    except PermissionError:
        # --- fallback: compress to a temp file then sudo-move into place --
        import tempfile

        print(
            "  Permission denied writing directly; "
            "using sudo zstd fallback \u2026"
        )
        with tempfile.NamedTemporaryFile(
            suffix=".zst", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = ["zstd", "-3", "--threads=0", "-f", str(src), "-o", str(tmp_path)]
            subprocess.run(cmd, check=True)
            subprocess.run(
                ["sudo", "mv", str(tmp_path), str(dst)], check=True
            )
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    compressed_size = dst.stat().st_size
    ratio = src_size / compressed_size if compressed_size else 0
    print(
        f"  Compressed: {_sizeof_fmt(compressed_size)} "
        f"({ratio:.1f}x ratio)"
    )


# ---------------------------------------------------------------------------
# Stage tracking  (.reclaim markers)
# ---------------------------------------------------------------------------

STAGE_TOTAL = 5


def _reclaim_marker(path: Path) -> Path:
    """Return the ``.reclaim`` marker path for *path*."""
    return Path(str(path) + ".reclaim")


def _stage_complete(output: Path) -> bool:
    """A stage is complete when its output exists **or** has been reclaimed."""
    return output.exists() or _reclaim_marker(output).exists()


def _reclaim_file(path: Path) -> int:
    """Delete *path* and create a ``.reclaim`` marker.  Returns bytes freed."""
    if not path.exists():
        return 0
    size = path.stat().st_size
    marker = _reclaim_marker(path)
    try:
        path.unlink()
    except PermissionError:
        try:
            subprocess.run(["sudo", "rm", "-f", str(path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"  Failed to remove {path.name}: {exc}")
            return 0
    marker.write_text(
        f"reclaimed {datetime.now().isoformat()} "
        f"size={size} ({_sizeof_fmt(size)})\n"
    )
    print(f"  Reclaimed {path.name} ({_sizeof_fmt(size)})")
    return size


def _clean_reclaim_markers(directory: Path) -> None:
    """Remove all ``.reclaim`` markers in *directory* for a fresh start."""
    for marker in sorted(directory.glob("*.reclaim")):
        marker.unlink()
        print(f"  Cleared marker: {marker.name}")


def _print_skip(stage: int, name: str, path: Path) -> None:
    """Print a formatted skip message for a completed stage."""
    marker = _reclaim_marker(path)
    if path.exists():
        detail = f"{path.name} present ({_sizeof_fmt(path.stat().st_size)})"
    elif marker.exists():
        detail = f"{path.name} previously completed (reclaimed)"
    else:
        detail = "already complete"
    print(f"[{stage}/{STAGE_TOTAL}] {name} \u2014 skipped ({detail})")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_wikidata(
    wikidata_dir: Path,
    url: str = WIKIDATA_URL,
    force: bool = False,
) -> Path:
    """
    Download the Wikidata bz2 dump with progress and resume support.

    Returns the path to the downloaded bz2 file.
    """
    wikidata_dir.mkdir(parents=True, exist_ok=True)
    bz2_name = Path(urlparse(url).path).name
    dest = wikidata_dir / bz2_name
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not force:
        print(
            f"  Download: {dest} already present "
            f"({_sizeof_fmt(dest.stat().st_size)}), skipping"
        )
        return dest

    # Resume from partial download if possible
    downloaded = part.stat().st_size if part.exists() else 0
    headers = {
        "User-Agent": "DeepRedAI/1.0 (temporal augmentation pipeline)",
    }
    if downloaded > 0 and not force:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"  Resuming download from {_sizeof_fmt(downloaded)}")
    elif force and part.exists():
        part.unlink()
        downloaded = 0

    resp = requests.get(url, headers=headers, stream=True, timeout=60)

    if resp.status_code == 416:
        # Range not satisfiable - file is already complete on server side
        if part.exists():
            part.rename(dest)
            print(
                f"  Download complete (already fully downloaded): "
                f"{_sizeof_fmt(dest.stat().st_size)}"
            )
            return dest

    resp.raise_for_status()

    total = None
    if "content-length" in resp.headers:
        total = int(resp.headers["content-length"]) + downloaded
    elif "content-range" in resp.headers:
        m = re.search(r"/(\d+)", resp.headers["content-range"])
        if m:
            total = int(m.group(1))

    mode = "ab" if downloaded > 0 and resp.status_code == 206 else "wb"
    if mode == "wb":
        downloaded = 0

    with open(part, mode) as f, tqdm(
        total=total,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="  Downloading",
        ncols=100,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    part.rename(dest)
    print(f"  Download complete: {_sizeof_fmt(dest.stat().st_size)}")
    return dest


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _find_bz2_tool() -> Optional[str]:
    """Find the best available bz2 decompression tool.

    Preference order: lbzip2 (parallel) > pbzip2 (parallel) > bunzip2.
    Returns ``None`` when none are installed.
    """
    for tool in ("lbzip2", "pbzip2", "bunzip2"):
        if shutil.which(tool):
            return tool
    return None


def _extract_bz2_native(
    tool: str, bz2_path: Path, output_path: Path, verbose: bool
) -> None:
    """Decompress *bz2_path* using a system tool, writing to *output_path*.

    Progress (when *verbose*) is shown by polling the output file size.
    """
    cmd = [tool, "-d", "-c"]
    if tool == "lbzip2":
        import multiprocessing

        cmd.append(f"-n{multiprocessing.cpu_count()}")
    cmd.append(str(bz2_path))

    print(f"  Using {tool} for decompression")

    with open(output_path, "wb") as fout:
        proc = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.PIPE)

        last_size = 0
        bar = None
        if verbose:
            bar = tqdm(
                total=None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="  Extracting",
                ncols=100,
            )
        try:
            while proc.poll() is None:
                time.sleep(2)
                if bar:
                    try:
                        sz = output_path.stat().st_size
                        bar.update(sz - last_size)
                        last_size = sz
                    except OSError:
                        pass
        finally:
            if bar:
                try:
                    sz = output_path.stat().st_size
                    bar.update(sz - last_size)
                except OSError:
                    pass
                bar.close()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"{tool} failed (exit {proc.returncode}): {stderr}")


def _extract_bz2_python(
    bz2_path: Path, output_path: Path, verbose: bool
) -> None:
    """Decompress *bz2_path* using Python's bz2 module (fallback).

    Progress tracks compressed bytes read.  This is significantly slower
    than native tools for large files; install ``lbzip2`` for faster
    extraction.
    """
    print(
        "  Using Python bz2 fallback (slower; "
        "install lbzip2 for faster extraction)"
    )

    bz2_size = bz2_path.stat().st_size

    with (
        open(bz2_path, "rb") as fin,
        open(output_path, "wb") as fout,
        tqdm(
            total=bz2_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  Extracting",
            ncols=100,
            disable=not verbose,
        ) as bar,
    ):
        decompressor = bz2.BZ2Decompressor()
        while True:
            raw = fin.read(DOWNLOAD_CHUNK)
            if not raw:
                break
            bar.update(len(raw))

            # Handle multi-stream bz2 (each stream ends with eof flag)
            data = raw
            while data:
                try:
                    decompressed = decompressor.decompress(data)
                    fout.write(decompressed)
                except EOFError:
                    break
                if decompressor.eof:
                    data = decompressor.unused_data
                    if data:
                        decompressor = bz2.BZ2Decompressor()
                    else:
                        break
                else:
                    break


def extract_wikidata(
    bz2_path: Path,
    wikidata_dir: Path,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Decompress a ``.ttl.bz2`` archive with progress.

    Uses native tools (lbzip2 > pbzip2 > bunzip2) when available, falling
    back to Python's ``bz2`` module.

    Returns the path to the extracted TTL file.
    """
    # Derive TTL name: strip trailing .bz2
    bz2_name = bz2_path.name
    ttl_name = bz2_name[:-4] if bz2_name.endswith(".bz2") else bz2_name
    ttl_path = wikidata_dir / ttl_name

    if ttl_path.exists() and not force:
        print(
            f"  Extract: {ttl_path} already present "
            f"({_sizeof_fmt(ttl_path.stat().st_size)}), skipping"
        )
        return ttl_path

    part = ttl_path.with_suffix(ttl_path.suffix + ".part")

    tool = _find_bz2_tool()
    if tool:
        _extract_bz2_native(tool, bz2_path, part, verbose)
    else:
        _extract_bz2_python(bz2_path, part, verbose)

    part.rename(ttl_path)
    print(f"  Extraction complete: {_sizeof_fmt(ttl_path.stat().st_size)}")
    return ttl_path


# ---------------------------------------------------------------------------
# Wikidata TTL Parser
# ---------------------------------------------------------------------------


class WikidataTimeExtractor:
    """Extract time-related metadata from Wikidata TTL files."""

    # Wikidata time-related properties
    # P569: date of birth
    # P570: date of death
    # P571: inception (founding, establishment)
    # P576: dissolved, abolished or demolished date
    TIME_PROPERTIES = {
        "wdt:P569",  # birth date
        "wdt:P570",  # death date
        "wdt:P571",  # inception date
        "wdt:P576",  # dissolution date
    }

    def __init__(
        self,
        ttl_file_path: str,
        csv_output_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        checkpoint_interval: int = 1_000_000,
    ):
        """Initialize the parser with the path to a TTL file.

        Args:
            ttl_file_path: Path to the TTL file to parse.
            csv_output_file: Path to CSV output file (for incremental writing).
            checkpoint_file: Path to checkpoint file for resume capability.
            checkpoint_interval: Number of lines between checkpoints and
                incremental saves.
        """
        self.ttl_file_path = Path(ttl_file_path)
        if not self.ttl_file_path.exists():
            raise FileNotFoundError(f"TTL file not found: {ttl_file_path}")

        # Store entity dates: entity_id -> list of dates
        self.entity_dates: Dict[str, List[datetime]] = defaultdict(list)

        # Store entity to Wikipedia info mapping: entity_id -> info dict
        self.entity_info: Dict[str, Dict] = {}

        # Store current entity being processed (for multi-line statements)
        self.current_entity: Optional[str] = None

        # Track entities we've seen Wikipedia links for
        self.entities_with_wiki: Set[str] = set()

        # Incremental save configuration
        self.csv_output_file = csv_output_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval
        self.csv_file_handle = None
        self.csv_writer = None

        # Track entities already written to CSV
        self.written_entities: Set[str] = set()

        # Current line position (for checkpointing)
        self.current_line = 0

        # Total lines in file (for progress reporting)
        self.total_lines = 0

        # Timing information
        self.start_time = None
        self.last_progress_time = None

        # Pre-compiled regex patterns for performance
        self._date_pattern = re.compile(r'"([^"]+)"')
        self._entity_pattern = re.compile(r"/entity/(Q\d+)")
        self._qid_pattern = re.compile(r"^Q\d+$")
        self._wiki_url_pattern = re.compile(
            r"<(https?://en\.wikipedia\.org/wiki/[^>]+)>"
        )
        self._schema_about_pattern = re.compile(r"schema:about\s+wd:(Q\d+)")
        self._title_pattern = re.compile(r"/wiki/(.+)$")

    # -------------------------------------------------------------- date parsing

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse a date string from Wikidata TTL format."""
        match = self._date_pattern.search(date_str)
        if not match:
            return None

        date_value = match.group(1)

        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m",
            "%Y",
        ):
            try:
                return datetime.strptime(date_value, fmt)
            except ValueError:
                continue

        return None

    # -------------------------------------------------------------- CSV output

    def initialize_csv_output(self, wikipedia_only: bool = True) -> None:
        """Initialize CSV output file and write header.

        Checks file permissions early before processing starts.
        """
        if not self.csv_output_file:
            return

        try:
            file_exists = os.path.exists(self.csv_output_file)

            if file_exists:
                self.csv_file_handle = open(
                    self.csv_output_file, "a", newline="", encoding="utf-8"
                )
                self.csv_writer = csv.writer(self.csv_file_handle)
                print(f"  Resuming: appending to existing {self.csv_output_file}")
            else:
                self.csv_file_handle = open(
                    self.csv_output_file, "w", newline="", encoding="utf-8"
                )
                self.csv_writer = csv.writer(self.csv_file_handle)
                self.csv_writer.writerow(
                    ["Entity_ID", "Entity", "Wikipedia_URL",
                     "Earliest_Date", "Latest_Date"]
                )
                self.csv_file_handle.flush()
                print(f"  Created output file: {self.csv_output_file}")
        except PermissionError as e:
            raise PermissionError(
                f"Cannot write to output file {self.csv_output_file}: {e}"
            )

    def close_csv_output(self) -> None:
        """Close the CSV output file."""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None

    # -------------------------------------------------------------- checkpoint

    def load_checkpoint(self) -> dict:
        """Load checkpoint data to resume from last position."""
        if not self.checkpoint_file or not os.path.exists(self.checkpoint_file):
            return {}

        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                line_num = data.get("line_number", 0)
                self.written_entities = set(data.get("written_entities", []))
                self.total_lines = data.get("total_lines", 0)
                print(f"  Loaded checkpoint: Resuming from line {line_num:,}")
                print(
                    f"  Already written {len(self.written_entities):,} entities"
                )
                if self.total_lines > 0:
                    print(f"  Total lines (cached): {self.total_lines:,}")
                entities_found = data.get("entities_found", 0)
                dates_found = data.get("dates_found", 0)
                wikipedia_links_found = data.get("wikipedia_links_found", 0)
                total_written = data.get(
                    "total_written", len(self.written_entities)
                )
                if entities_found or dates_found or wikipedia_links_found:
                    print(
                        f"  Entities: {entities_found:,} | "
                        f"Dates: {dates_found:,} | "
                        f"Wikipedia: {wikipedia_links_found:,}"
                    )
                return {
                    "line_number": line_num,
                    "entities_found": entities_found,
                    "dates_found": dates_found,
                    "wikipedia_links_found": wikipedia_links_found,
                    "total_written": total_written,
                }
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            return {}

    def save_checkpoint(
        self,
        line_number: int,
        entities_found: int = 0,
        dates_found: int = 0,
        wikipedia_links_found: int = 0,
        total_written: int = 0,
    ) -> None:
        """Save checkpoint data for resume capability."""
        if not self.checkpoint_file:
            return

        try:
            temp_file = self.checkpoint_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(
                    {
                        "line_number": line_number,
                        "written_entities": list(self.written_entities),
                        "total_lines": self.total_lines,
                        "entities_found": entities_found,
                        "dates_found": dates_found,
                        "wikipedia_links_found": wikipedia_links_found,
                        "total_written": total_written,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                )
            os.replace(temp_file, self.checkpoint_file)
        except Exception as e:
            print(f"  Warning: Could not save checkpoint: {e}")

    # -------------------------------------------------------------- line counting

    def count_file_lines(self, verbose: bool = False) -> int:
        """Count total lines using fast buffered binary reading."""
        if verbose:
            print(f"  Counting lines in {self.ttl_file_path} \u2026")
            start = time.time()

        try:
            def _make_gen(reader):
                buf_size = 2 ** 16  # 64 KB
                b = reader(buf_size)
                while b:
                    yield b
                    b = reader(buf_size)

            with open(self.ttl_file_path, "rb") as f:
                line_count = sum(
                    buf.count(b"\n") for buf in _make_gen(f.raw.read)
                )

            if verbose:
                elapsed = time.time() - start
                print(f"  Total lines: {line_count:,} (counted in {elapsed:.1f}s)")

            return line_count
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not count lines: {e}")
                print(
                    "  Will proceed without total line count "
                    "(progress % not available)"
                )
            return 0

    def fast_skip_lines(
        self, file_handle, target_line: int, verbose: bool = False
    ) -> int:
        """Skip to *target_line* using fast binary newline counting.

        Much faster than ``for line in f`` when content is not needed.
        """
        if target_line <= 0:
            return 0

        if verbose:
            print(f"  Fast-skipping to line {target_line:,} \u2026")
            start = time.time()
            last_report = start

        lines_counted = 0
        buf_size = 2 ** 20  # 1 MB

        while lines_counted < target_line:
            buf = file_handle.read(buf_size)
            if not buf:
                break

            newlines_in_buf = buf.count(b"\n")

            if lines_counted + newlines_in_buf >= target_line:
                remaining = target_line - lines_counted
                pos = 0
                for _ in range(remaining):
                    next_nl = buf.find(b"\n", pos)
                    if next_nl == -1:
                        break
                    pos = next_nl + 1
                bytes_back = len(buf) - pos
                file_handle.seek(-bytes_back, 1)
                lines_counted = target_line
                break

            lines_counted += newlines_in_buf

            if verbose and time.time() - last_report > 10:
                elapsed = time.time() - start
                pct = (lines_counted / target_line) * 100
                rate = lines_counted / elapsed if elapsed > 0 else 0
                eta = (target_line - lines_counted) / rate if rate > 0 else 0
                print(
                    f"  Skipped {lines_counted:,} / {target_line:,} lines "
                    f"({pct:.1f}%) | "
                    f"ETA: {self._format_time(eta)}"
                )
                last_report = time.time()

        if verbose:
            elapsed = time.time() - start
            print(f"  Skipped {lines_counted:,} lines in {elapsed:.1f}s")

        return lines_counted

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format *seconds* into human-readable duration."""
        if seconds < 0:
            return "calculating\u2026"
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        if td.days > 0:
            return f"{td.days}d {hours}h {minutes}m"
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    # -------------------------------------------------------------- incremental write

    def write_incremental_results(self, wikipedia_only: bool = True) -> int:
        """Write new results to CSV and free memory.

        Returns the number of entities written.
        """
        if not self.csv_writer:
            return 0

        written_count = 0
        entities_to_cleanup = []

        for entity_id in list(self.entity_dates.keys()):
            if entity_id in self.written_entities:
                entities_to_cleanup.append(entity_id)
                continue

            dates = self.entity_dates[entity_id]
            info = self.entity_info.get(entity_id, {})

            wikipedia_url = info.get("wikipedia_url")
            wikipedia_title = info.get("wikipedia_title", "")

            if wikipedia_only and not wikipedia_url:
                continue

            earliest = min(dates).strftime("%Y-%m-%d") if dates else "0"
            latest = max(dates).strftime("%Y-%m-%d") if dates else "0"

            self.csv_writer.writerow(
                [
                    entity_id,
                    wikipedia_title or entity_id,
                    wikipedia_url or "",
                    earliest,
                    latest,
                ]
            )

            self.written_entities.add(entity_id)
            entities_to_cleanup.append(entity_id)
            written_count += 1

        for entity_id in entities_to_cleanup:
            self.entity_dates.pop(entity_id, None)
            self.entity_info.pop(entity_id, None)
            self.entities_with_wiki.discard(entity_id)

        if written_count > 0:
            self.csv_file_handle.flush()

        return written_count

    # -------------------------------------------------------------- entity helpers

    def extract_entity_id(self, entity_str: str) -> Optional[str]:
        """Extract Wikidata entity ID (e.g. ``Q23``) from various formats."""
        if entity_str.startswith("wd:"):
            return entity_str[3:]
        match = self._entity_pattern.search(entity_str)
        if match:
            return match.group(1)
        if self._qid_pattern.match(entity_str):
            return entity_str
        return None

    def parse_wikipedia_url(
        self, line: str
    ) -> Optional[Tuple[str, str]]:
        """Extract ``(entity_id | None, wikipedia_url | None)`` from a line."""
        if "schema:about" in line:
            match = self._schema_about_pattern.search(line)
            if match:
                return (self.extract_entity_id(match.group(1)), None)

        if "en.wikipedia.org/wiki/" in line:
            match = self._wiki_url_pattern.search(line)
            if match:
                return (None, unquote(match.group(1)))

        return None

    # -------------------------------------------------------------- main parse loop

    def parse_file(
        self, verbose: bool = False, wikipedia_only: bool = True
    ) -> None:
        """Stream-parse the TTL file and collect temporal facts.

        A ``tqdm`` progress bar is displayed when *verbose* is True.
        Checkpoint/resume is handled automatically.
        """
        checkpoint_data = self.load_checkpoint()
        start_line = checkpoint_data.get("line_number", 0)

        if verbose and self.total_lines == 0:
            self.total_lines = self.count_file_lines(verbose=True)

        self.start_time = time.time()
        self.last_progress_time = self.start_time

        if verbose:
            print(f"  Parsing {self.ttl_file_path} \u2026")
            print(
                f"  Looking for time properties: "
                f"{', '.join(sorted(self.TIME_PROPERTIES))}"
            )
            if start_line > 0:
                print(f"  Resuming from line {start_line:,}")

        line_count = 0
        entities_found = checkpoint_data.get("entities_found", 0)
        dates_found = checkpoint_data.get("dates_found", 0)
        wikipedia_links_found = checkpoint_data.get("wikipedia_links_found", 0)
        total_written = checkpoint_data.get(
            "total_written", len(self.written_entities)
        )

        current_subject = None
        current_entity_id = None
        current_sitelink_url = None
        current_sitelink_title = None
        current_sitelink_entity = None  # noqa: F841

        with open(self.ttl_file_path, "rb") as f:
            if start_line > 0:
                line_count = self.fast_skip_lines(f, start_line, verbose=verbose)

            text_reader = io.TextIOWrapper(f, encoding="utf-8")

            for line in text_reader:
                line_count += 1

                original_line = line
                line = line.strip()

                # Progress indicator
                if verbose and line_count % 5_000_000 == 0:
                    elapsed = time.time() - self.start_time
                    msg = f"  Processed {line_count:,} lines"
                    if self.total_lines > 0:
                        pct = (line_count / self.total_lines) * 100
                        msg += f" ({pct:.1f}%)"
                        if line_count > start_line:
                            processed = line_count - start_line
                            remaining = self.total_lines - line_count
                            rate = processed / elapsed
                            if rate > 0:
                                msg += (
                                    f" | ETA: "
                                    f"{self._format_time(remaining / rate)}"
                                )
                    msg += (
                        f" | Entities: {entities_found:,}"
                        f" | Dates: {dates_found:,}"
                        f" | Wikipedia: {wikipedia_links_found:,}"
                    )
                    print(msg)

                # Incremental save and checkpoint
                if line_count % self.checkpoint_interval == 0:
                    if self.csv_writer:
                        newly_written = self.write_incremental_results(
                            wikipedia_only=wikipedia_only
                        )
                        total_written += newly_written

                    self.save_checkpoint(
                        line_count, entities_found, dates_found,
                        wikipedia_links_found, total_written,
                    )
                    self.current_line = line_count

                if not line or line.startswith("#"):
                    continue
                if line.startswith("@prefix") or line.startswith("@base"):
                    continue

                # New subject (no leading whitespace)
                if original_line and not original_line[0].isspace():
                    parts = line.split(None, 1)
                    if parts:
                        current_subject = parts[0]
                        current_entity_id = self.extract_entity_id(
                            current_subject
                        )

                        if "<https://en.wikipedia.org/wiki/" in line:
                            match = self._wiki_url_pattern.search(line)
                            if match:
                                url = unquote(match.group(1))
                                title_match = self._title_pattern.search(url)
                                title = (
                                    title_match.group(1) if title_match else None
                                )
                                current_sitelink_entity = None  # noqa: F841
                                current_sitelink_url = url
                                current_sitelink_title = title
                            else:
                                current_sitelink_url = None
                                current_sitelink_title = None
                        else:
                            current_sitelink_url = None
                            current_sitelink_title = None

                # Time properties (fast pre-check: 'wdt:P5')
                if current_entity_id and "wdt:P5" in line:
                    for time_prop in self.TIME_PROPERTIES:
                        if time_prop in line:
                            date_obj = self.parse_date(line)
                            if date_obj:
                                is_new = (
                                    current_entity_id not in self.entity_dates
                                )
                                if is_new:
                                    entities_found += 1
                                    if current_entity_id not in self.entity_info:
                                        self.entity_info[current_entity_id] = {
                                            "wikipedia_url": None,
                                            "wikipedia_title": None,
                                        }
                                self.entity_dates[current_entity_id].append(
                                    date_obj
                                )
                                dates_found += 1
                            break

                # schema:about links Wikipedia URL to entity
                if current_sitelink_url and "schema:about" in line:
                    match = self._schema_about_pattern.search(line)
                    if match:
                        entity_id = match.group(1)
                        if entity_id not in self.entity_info:
                            self.entity_info[entity_id] = {
                                "wikipedia_url": current_sitelink_url,
                                "wikipedia_title": current_sitelink_title,
                            }
                        else:
                            self.entity_info[entity_id][
                                "wikipedia_url"
                            ] = current_sitelink_url
                            self.entity_info[entity_id][
                                "wikipedia_title"
                            ] = current_sitelink_title
                        self.entities_with_wiki.add(entity_id)
                        wikipedia_links_found += 1

                # End of statement block
                if line.endswith("."):
                    current_subject = None
                    current_entity_id = None
                    current_sitelink_url = None
                    current_sitelink_title = None
                    current_sitelink_entity = None  # noqa: F841

        # Final incremental save
        if self.csv_writer:
            newly_written = self.write_incremental_results(
                wikipedia_only=wikipedia_only
            )
            total_written += newly_written
            if verbose and newly_written > 0:
                print(
                    f"  \u2192 Final save: {newly_written:,} new entities written"
                )

        self.save_checkpoint(
            line_count, entities_found, dates_found,
            wikipedia_links_found, total_written,
        )
        self.current_line = line_count

        if verbose:
            print(f"\n  Parsing complete!")
            print(f"  Total lines processed: {line_count:,}")
            print(f"  Entities with time data: {entities_found:,}")
            print(f"  Total dates extracted: {dates_found:,}")
            print(f"  Wikipedia links found: {wikipedia_links_found:,}")
            print(
                f"  Entities with both dates and Wikipedia links: "
                f"{len(self.entities_with_wiki & set(self.entity_dates.keys())):,}"
            )
            if self.csv_output_file:
                print(f"  Total entities written to CSV: {total_written:,}")

    # -------------------------------------------------------------- results

    def get_results(
        self, wikipedia_only: bool = True
    ) -> List[Tuple[str, str, str, Optional[str], Optional[str]]]:
        """Return a sorted list of result tuples.

        Each tuple: ``(entity_id, wikipedia_title, wikipedia_url,
        earliest_date, latest_date)``
        """
        results = []
        for entity_id in sorted(self.entity_dates.keys()):
            dates = self.entity_dates[entity_id]
            info = self.entity_info.get(entity_id, {})
            wikipedia_url = info.get("wikipedia_url")
            wikipedia_title = info.get("wikipedia_title", "")

            if wikipedia_only and not wikipedia_url:
                continue

            earliest = min(dates).strftime("%Y-%m-%d") if dates else None
            latest = max(dates).strftime("%Y-%m-%d") if dates else None
            results.append(
                (entity_id, wikipedia_title, wikipedia_url or "",
                 earliest, latest)
            )
        return results

    # -------------------------------------------------------------- export

    def export_csv(
        self,
        output_file: str,
        wikipedia_only: bool = True,
        compress: bool = False,
    ) -> Path:
        """Export results to CSV, optionally zstd-compressed.

        Returns the final output path.

        Note: This method exports all results at once.  For incremental
        saving, use ``initialize_csv_output()`` before parsing.
        """
        results = self.get_results(wikipedia_only=wikipedia_only)
        out_path = Path(output_file)

        if compress:
            import zstandard as zstd

            if out_path.suffix != ".zst":
                out_path = out_path.with_suffix(out_path.suffix + ".zst")
            cctx = zstd.ZstdCompressor(level=3, threads=-1)
            raw = open(out_path, "wb")
            fh = io.TextIOWrapper(
                cctx.stream_writer(raw, closefd=True),
                encoding="utf-8",
                newline="",
            )
        else:
            fh = open(out_path, "w", newline="", encoding="utf-8")

        try:
            writer = csv.writer(fh)
            writer.writerow(
                ["Entity_ID", "Entity", "Wikipedia_URL",
                 "Earliest_Date", "Latest_Date"]
            )
            for entity_id, wiki_title, wiki_url, earliest, latest in results:
                writer.writerow(
                    [
                        entity_id,
                        wiki_title or entity_id,
                        wiki_url,
                        earliest or "0",
                        latest or "0",
                    ]
                )
        finally:
            fh.close()

        size = out_path.stat().st_size
        print(
            f"  Exported {len(results):,} entities to {out_path} "
            f"({_sizeof_fmt(size)})"
        )
        return out_path

    def export_json(
        self,
        output_file: str,
        wikipedia_only: bool = True,
        compress: bool = False,
    ) -> Path:
        """Export results to JSON, optionally zstd-compressed.

        Returns the final output path.
        """
        results = self.get_results(wikipedia_only=wikipedia_only)
        out_path = Path(output_file)

        data = [
            {
                "entity_id": eid,
                "entity": wiki_title or eid,
                "wikipedia_url": wiki_url,
                "earliest_date": earliest or "0",
                "latest_date": latest or "0",
            }
            for eid, wiki_title, wiki_url, earliest, latest in results
        ]

        if compress:
            import zstandard as zstd

            if out_path.suffix != ".zst":
                out_path = out_path.with_suffix(out_path.suffix + ".zst")
            cctx = zstd.ZstdCompressor(level=3, threads=-1)
            payload = json.dumps(data, indent=2, ensure_ascii=False).encode(
                "utf-8"
            )
            with open(out_path, "wb") as f:
                f.write(cctx.compress(payload))
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        size = out_path.stat().st_size
        print(
            f"  Exported {len(results):,} entities to {out_path} "
            f"({_sizeof_fmt(size)})"
        )
        return out_path

    def print_summary(self, limit: int = 20) -> None:
        """Print a tabular summary of extracted data."""
        results = self.get_results(wikipedia_only=True)
        print(f"\n{'=' * 80}")
        print("Wikidata Time Extraction Summary")
        print(f"{'=' * 80}")
        print(
            f"Total entities with time data and Wikipedia links: "
            f"{len(results):,}"
        )
        print(f"\nShowing first {min(limit, len(results))} entities:\n")

        print(
            f"{'Entity ID':<12} {'Wikipedia Title':<40} "
            f"{'Earliest':<12} {'Latest':<12}"
        )
        print(f"{'-' * 80}")
        for eid, wiki_title, _url, earliest, latest in results[:limit]:
            title_disp = (
                wiki_title[:37] + "..." if len(wiki_title) > 40 else wiki_title
            )
            print(
                f"{eid:<12} {title_disp:<40} "
                f"{earliest or 'N/A':<12} {latest or 'N/A':<12}"
            )
        if len(results) > limit:
            print(f"\n\u2026 and {len(results) - limit:,} more entities")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description=(
            "Wikidata temporal metadata pipeline \u2014 "
            "download, extract, parse, compress, reclaim"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All five stages run by default.  Each stage is skipped automatically
when its output file (or a .reclaim marker) already exists.

Typical usage:
  python wikidata_parser.py --verbose            # run full pipeline
  python wikidata_parser.py --force --verbose    # redo everything from scratch

Skip optional stages:
  python wikidata_parser.py --no-compress        # stop after plain CSV
  python wikidata_parser.py --no-reclaim         # keep intermediate files

Individual stages:
  python wikidata_parser.py --download-only
  python wikidata_parser.py --extract-only
  python wikidata_parser.py --parse-only

Legacy (positional TTL path still works):
  python wikidata_parser.py /path/to/wikidata.ttl --csv out.csv -v
        """,
    )

    # Directories / files
    default_wikidata_dir = os.path.join(
        os.environ.get("WIKI_DATA", "/mnt/data/wikipedia"), "wikidata"
    )
    parser.add_argument(
        "ttl_file",
        nargs="?",
        default=None,
        help=(
            "Path to an existing TTL file (skip download/extract).  "
            "If omitted, the script manages download and extraction "
            "automatically."
        ),
    )
    parser.add_argument(
        "--wikidata-dir",
        default=default_wikidata_dir,
        help=f"Working directory (default: {default_wikidata_dir})",
    )
    parser.add_argument(
        "--url", default=WIKIDATA_URL, help="Wikidata bz2 dump download URL"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV output path (default: <wikidata-dir>/wikidata-temporal.csv)",
    )
    parser.add_argument("--json", default=None, help="JSON output path")

    # Stage control
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, then stop",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only download + extract, then stop",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse (TTL must already exist), then stop",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip the compression stage (keep plain CSV)",
    )
    parser.add_argument(
        "--no-reclaim",
        action="store_true",
        help="Skip the reclamation stage (keep intermediate files)",
    )

    # Behaviour
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear .reclaim markers and re-run all stages from scratch",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress bars and detail",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Entities shown in summary (default: 20)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip console summary",
    )

    # Parse-specific options (checkpoint)
    parser.add_argument(
        "--all-entities",
        action="store_true",
        help="Include entities without Wikipedia links in output",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Custom checkpoint file path "
            "(default: <csv_file>.checkpoint)"
        ),
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint/incremental mode (not recommended)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1_000_000,
        help=(
            "Lines between checkpoints and incremental saves "
            "(default: 1,000,000)"
        ),
    )

    # Deprecated / backward-compat flags (accepted silently)
    parser.add_argument(
        "--summary", action="store_true", default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--restart", action="store_true", default=False,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # --restart is a backward-compat alias for --force
    if args.restart:
        args.force = True

    wikidata_dir = Path(args.wikidata_dir)
    wikidata_dir.mkdir(parents=True, exist_ok=True)

    csv_out = args.csv or str(wikidata_dir / DEFAULT_CSV_NAME)

    # Derive filenames from URL
    bz2_name = Path(urlparse(args.url).path).name
    ttl_name = bz2_name[:-4] if bz2_name.endswith(".bz2") else bz2_name

    # Key paths used across stages
    bz2_path = wikidata_dir / bz2_name
    ttl_path = wikidata_dir / ttl_name
    csv_path = Path(csv_out)
    csv_zst_path = csv_path.with_suffix(csv_path.suffix + ".zst")

    # Checkpoint file
    checkpoint_file = None
    if not args.no_checkpoint:
        checkpoint_file = args.checkpoint or (str(csv_path) + ".checkpoint")

    t0 = time.time()

    try:
        # --force: clear every .reclaim marker, checkpoint, and outputs
        if args.force:
            print("[prep] Clearing markers and outputs for fresh run")
            _clean_reclaim_markers(wikidata_dir)
            for cf in [checkpoint_file, (checkpoint_file + ".tmp") if checkpoint_file else None]:
                if cf and os.path.exists(cf):
                    os.remove(cf)
                    print(f"  Cleared checkpoint: {Path(cf).name}")
            for p in [csv_path, csv_zst_path]:
                if p.exists():
                    p.unlink()
                    print(f"  Cleared: {p.name}")

        # ----------------------------------------------------------
        # [1/5] Download
        # ----------------------------------------------------------
        if args.ttl_file:
            print(f"[1/{STAGE_TOTAL}] Download \u2014 skipped (TTL provided)")
        elif args.parse_only:
            print(f"[1/{STAGE_TOTAL}] Download \u2014 skipped (--parse-only)")
        elif _stage_complete(bz2_path) and not args.force:
            _print_skip(1, "Download", bz2_path)
        else:
            print(f"[1/{STAGE_TOTAL}] Download")
            bz2_path = download_wikidata(
                wikidata_dir, url=args.url, force=args.force
            )
        if args.download_only:
            print(f"\nDone in {time.time() - t0:.1f}s")
            return

        # ----------------------------------------------------------
        # [2/5] Extract
        # ----------------------------------------------------------
        if args.ttl_file:
            ttl_path = Path(args.ttl_file)
            print(f"[2/{STAGE_TOTAL}] Extract \u2014 skipped (TTL provided)")
        elif args.parse_only:
            print(f"[2/{STAGE_TOTAL}] Extract \u2014 skipped (--parse-only)")
        elif _stage_complete(ttl_path) and not args.force:
            _print_skip(2, "Extract", ttl_path)
        else:
            print(f"[2/{STAGE_TOTAL}] Extract")
            if not bz2_path.exists():
                print(
                    f"  Error: bz2 not found at {bz2_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
            ttl_path = extract_wikidata(
                bz2_path, wikidata_dir,
                force=args.force, verbose=args.verbose,
            )
        if args.extract_only:
            print(f"\nDone in {time.time() - t0:.1f}s")
            return

        # ----------------------------------------------------------
        # [3/5] Parse
        # ----------------------------------------------------------
        checkpoint_present = (
            checkpoint_file and os.path.exists(checkpoint_file)
        )
        parse_done = (
            (_stage_complete(csv_path) or csv_zst_path.exists())
            and not checkpoint_present
        )
        if parse_done and not args.force:
            _print_skip(3, "Parse", csv_path)
        else:
            print(f"[3/{STAGE_TOTAL}] Parse")
            if not ttl_path.exists():
                print(
                    f"  Error: TTL file not found: {ttl_path}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Create extractor with checkpoint support
            incremental_csv = checkpoint_file is not None
            csv_output = str(csv_path) if incremental_csv else None

            extractor = WikidataTimeExtractor(
                str(ttl_path),
                csv_output_file=csv_output,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=args.checkpoint_interval,
            )

            if incremental_csv:
                extractor.initialize_csv_output(
                    wikipedia_only=not args.all_entities
                )

            try:
                extractor.parse_file(
                    verbose=args.verbose,
                    wikipedia_only=not args.all_entities,
                )
            finally:
                extractor.close_csv_output()

            # Non-incremental CSV export
            if not incremental_csv and csv_out:
                extractor.export_csv(
                    csv_out, wikipedia_only=not args.all_entities
                )

            # JSON export
            if args.json:
                extractor.export_json(
                    args.json, wikipedia_only=not args.all_entities
                )

            # Summary
            if not args.no_summary:
                extractor.print_summary(limit=args.limit)

            # Clean up checkpoint on successful completion
            if checkpoint_file and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                if args.verbose:
                    print("  Removed checkpoint file (parse complete)")

        if args.parse_only:
            print(f"\nDone in {time.time() - t0:.1f}s")
            return

        # ----------------------------------------------------------
        # [4/5] Compress
        # ----------------------------------------------------------
        if args.no_compress:
            print(f"[4/{STAGE_TOTAL}] Compress \u2014 skipped (--no-compress)")
        elif csv_zst_path.exists() and not args.force:
            _print_skip(4, "Compress", csv_zst_path)
        else:
            print(f"[4/{STAGE_TOTAL}] Compress")
            if csv_path.exists():
                _compress_existing(csv_path, csv_zst_path, args.verbose)
            else:
                print(
                    f"  Error: {csv_path.name} not found for compression",
                    file=sys.stderr,
                )
                sys.exit(1)

        # ----------------------------------------------------------
        # [5/5] Reclaim
        # ----------------------------------------------------------
        if args.no_reclaim:
            print(f"[5/{STAGE_TOTAL}] Reclaim \u2014 skipped (--no-reclaim)")
        else:
            print(f"[5/{STAGE_TOTAL}] Reclaim")

            reclaimable = [bz2_path, ttl_path]

            # Only reclaim plain CSV when the compressed version exists
            if csv_zst_path.exists() and not args.no_compress:
                reclaimable.append(csv_path)

            # Never reclaim a user-provided TTL
            if args.ttl_file:
                user_ttl = Path(args.ttl_file).resolve()
                reclaimable = [
                    p for p in reclaimable if p.resolve() != user_ttl
                ]

            total = sum(_reclaim_file(p) for p in reclaimable)
            if total:
                print(f"  Reclaimed {_sizeof_fmt(total)} total")
            else:
                print("  Nothing to reclaim (already clean)")

        print(f"\nDone in {time.time() - t0:.1f}s")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
