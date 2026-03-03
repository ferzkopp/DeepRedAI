#!/usr/bin/env python3
"""
YAGO TTL Parser - Download, extract, parse, compress, and reclaim

This script manages the full YAGO temporal-data pipeline through five
explicit stages:

  1. Download  - fetch the YAGO zip archive  (with resume support)
  2. Extract   - unpack yago-facts.ttl from the zip
  3. Parse     - extract temporal predicates and write plain CSV
  4. Compress  - zstd-compress the CSV for downstream streaming
  5. Reclaim   - delete intermediate files, leaving .reclaim markers

Every stage checks whether its output file already exists *or* whether
a ``.reclaim`` marker file indicates the output was previously produced
and then cleaned up.  This means the pipeline can be re-run at any time
and will pick up where it left off.

After a full run the working directory contains:
  - ``yago-facts.csv.zst``                  (final compressed output)
  - ``yago-4.5.0.2.zip.reclaim``            (marker: zip was downloaded)
  - ``yago-facts.ttl.reclaim``              (marker: TTL was extracted)
  - ``yago-facts.csv.reclaim``              (marker: CSV was produced)

Pass ``--force`` to clear all ``.reclaim`` markers and redo every stage.

Full dataset: https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip

Environment variables (from deepred-env.sh):
  WIKI_DATA   - root data directory (default: /mnt/data/wikipedia)
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YAGO_URL = "https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip"
YAGO_ZIP_NAME = "yago-4.5.0.2.zip"
YAGO_TTL_MEMBER = "yago-facts.ttl"  # member inside the zip we need

# Default output filenames (relative to <yago_dir>)
DEFAULT_CSV_NAME = "yago-facts.csv"

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


def _open_output(path: Path, mode: str = "w", compress: bool = False):
    """
    Return a file-like object for *path*.

    When *compress* is True the file is written through a zstd compressor
    (level 3 - fast compression, very fast decompression) and the path
    gets a ``.zst`` suffix appended if not already present.

    Returns ``(fh, final_path)``.
    """
    if compress:
        import zstandard as zstd

        if path.suffix != ".zst":
            path = path.with_suffix(path.suffix + ".zst")
        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        raw = open(path, "wb" if "w" in mode else "ab")
        fh = io.TextIOWrapper(
            cctx.stream_writer(raw, closefd=True),
            encoding="utf-8",
            newline="",
        )
        return fh, path
    else:
        fh = open(path, mode, newline="", encoding="utf-8")
        return fh, path


def _compress_existing(src: Path, dst: Path, verbose: bool = False) -> None:
    """Compress an existing plain file to zstd without re-parsing.

    Falls back to the ``zstd`` CLI (via sudo if needed) when the
    destination directory is not writable by the current user.
    """
    import tempfile

    src_size = src.stat().st_size
    print(
        f"  Compressing {src.name} → {dst.name} "
        f"({_sizeof_fmt(src_size)}) …"
    )

    # --- fast-path: try writing directly via the zstandard library --------
    try:
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        part = dst.with_suffix(dst.suffix + ".part")
        with open(src, "rb") as fin, open(part, "wb") as fout:
            if verbose:
                from tqdm import tqdm

                with tqdm(
                    total=src_size,
                    unit="B",
                    unit_scale=True,
                    desc="  zstd",
                ) as pbar:
                    writer = cctx.stream_writer(fout)
                    while True:
                        chunk = fin.read(1 << 20)  # 1 MiB
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
        print(
            "  Permission denied writing directly; "
            "using sudo zstd fallback …"
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
    print(f"[{stage}/{STAGE_TOTAL}] {name} — skipped ({detail})")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_yago(
    yago_dir: Path, url: str = YAGO_URL, force: bool = False
) -> Path:
    """
    Download the YAGO zip archive with progress and resume support.

    Returns the path to the downloaded zip file.
    """
    yago_dir.mkdir(parents=True, exist_ok=True)
    dest = yago_dir / YAGO_ZIP_NAME
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not force:
        print(
            f"  Download: {dest} already present "
            f"({_sizeof_fmt(dest.stat().st_size)}), skipping"
        )
        return dest

    # Resume from partial download if possible
    downloaded = part.stat().st_size if part.exists() else 0
    headers = {}
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


def extract_yago(
    zip_path: Path, yago_dir: Path, force: bool = False
) -> Path:
    """
    Extract ``yago-facts.ttl`` from the zip archive with progress.

    Returns the path to the extracted TTL file.
    """
    ttl_path = yago_dir / YAGO_TTL_MEMBER

    if ttl_path.exists() and not force:
        print(
            f"  Extract: {ttl_path} already present "
            f"({_sizeof_fmt(ttl_path.stat().st_size)}), skipping"
        )
        return ttl_path

    part = ttl_path.with_suffix(ttl_path.suffix + ".part")

    with zipfile.ZipFile(zip_path, "r") as zf:
        info = zf.getinfo(YAGO_TTL_MEMBER)
        total = info.file_size

        with (
            zf.open(YAGO_TTL_MEMBER) as src,
            open(part, "wb") as dst,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="  Extracting",
                ncols=100,
            ) as bar,
        ):
            while True:
                chunk = src.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
                bar.update(len(chunk))

    part.rename(ttl_path)
    print(f"  Extraction complete: {_sizeof_fmt(ttl_path.stat().st_size)}")
    return ttl_path


# ---------------------------------------------------------------------------
# YAGO TTL Parser
# ---------------------------------------------------------------------------


class YagoTimeExtractor:
    """Extract time-related metadata from YAGO TTL files."""

    # Schema.org time-related predicates
    TIME_PREDICATES = {
        "schema:birthDate",
        "schema:deathDate",
        "schema:startDate",
        "schema:endDate",
        "schema:datePublished",
    }

    def __init__(self, ttl_file_path: str):
        self.ttl_file_path = Path(ttl_file_path)
        if not self.ttl_file_path.exists():
            raise FileNotFoundError(f"TTL file not found: {ttl_file_path}")

        # entity -> list of datetimes
        self.entity_dates: Dict[str, List[datetime]] = defaultdict(list)
        # entity -> {uri, wikipedia_url}
        self.entity_info: Dict[str, Dict] = {}

    # -------------------------------------------------------------- date parsing

    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """Parse a date string from TTL format."""
        match = re.search(r'"([^"]+)"', date_str)
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

    # -------------------------------------------------------------- entity name decoding

    @staticmethod
    def decode_yago_entity_name(name: str) -> str:
        """
        Decode YAGO Unicode escape sequences in entity names.

        Patterns handled:
        - ``__uXXXX_``  ->  ``_<char>``  (preserve leading underscore)
        - ``_uXXXX_``   ->  ``<char>``
        - ``uXXXX``     ->  ``<char>``   (common punctuation only)
        """

        def _replace_double(m):
            try:
                return "_" + chr(int(m.group(1), 16))
            except ValueError:
                return m.group(0)

        def _replace_single(m):
            try:
                return chr(int(m.group(1), 16))
            except ValueError:
                return m.group(0)

        def _replace_bare(m):
            try:
                c = chr(int(m.group(1), 16))
                return c if c in "()[]{}:;,/\\|<>+=-" else m.group(0)
            except ValueError:
                return m.group(0)

        name = re.sub(r"__u([0-9a-fA-F]{4})_", _replace_double, name)
        name = re.sub(r"_u([0-9a-fA-F]{4})_", _replace_single, name)
        name = re.sub(
            r"u([0-9a-fA-F]{4})(?![0-9a-fA-F])", _replace_bare, name
        )
        return name

    @staticmethod
    def extract_entity_name(entity_uri: str) -> str:
        """Extract a readable entity name from a YAGO URI and decode it."""
        name = entity_uri.split(":", 1)[1] if ":" in entity_uri else entity_uri
        return YagoTimeExtractor.decode_yago_entity_name(name)

    @staticmethod
    def parse_wikipedia_link(line: str) -> Optional[str]:
        """Return Wikipedia URL found on *line*, or ``None``."""
        if "schema:mainEntityOfPage" in line or "wikipedia.org/wiki/" in line:
            m = re.search(
                r'"(https?://[^/]*wikipedia\.org/wiki/[^"]+)"', line
            )
            if m:
                return m.group(1)
        return None

    # -------------------------------------------------------------- main parse loop

    def parse_file(self, verbose: bool = False) -> None:
        """
        Stream-parse the TTL file and collect temporal facts.

        A ``tqdm`` progress bar (bytes) is displayed when *verbose* is True.
        """
        file_size = self.ttl_file_path.stat().st_size

        entities_found = 0
        dates_found = 0

        with open(self.ttl_file_path, "r", encoding="utf-8") as f, tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  Parsing TTL",
            ncols=100,
            disable=not verbose,
        ) as bar:
            for line in f:
                bar.update(len(line.encode("utf-8")))
                line = line.strip()

                if not line or line.startswith("#") or line.startswith("@prefix"):
                    continue

                # Check time predicates
                for predicate in self.TIME_PREDICATES:
                    if predicate in line:
                        parts = line.split(None, 2)
                        if len(parts) >= 3:
                            subject = parts[0]
                            date_obj = self.parse_date(line)
                            if date_obj:
                                entity_name = self.extract_entity_name(subject)
                                if entity_name not in self.entity_dates:
                                    entities_found += 1
                                    self.entity_info[entity_name] = {
                                        "uri": subject,
                                        "wikipedia_url": None,
                                    }
                                self.entity_dates[entity_name].append(date_obj)
                                dates_found += 1
                        break

                # Wikipedia URLs
                if "wikipedia.org/wiki/" in line:
                    parts = line.split(None, 1)
                    if parts:
                        subject = parts[0]
                        entity_name = self.extract_entity_name(subject)
                        wiki_url = self.parse_wikipedia_link(line)
                        if wiki_url and entity_name in self.entity_info:
                            self.entity_info[entity_name]["wikipedia_url"] = wiki_url

        if verbose:
            print(
                f"  Parsing complete: "
                f"{entities_found:,} entities, {dates_found:,} dates"
            )

    # -------------------------------------------------------------- results

    def get_results(self) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
        """Return sorted list of ``(entity, wikipedia_url, earliest, latest)``."""
        results = []
        for entity_name in sorted(self.entity_dates.keys()):
            dates = self.entity_dates[entity_name]
            info = self.entity_info.get(entity_name, {})
            earliest = min(dates).strftime("%Y-%m-%d") if dates else None
            latest = max(dates).strftime("%Y-%m-%d") if dates else None
            wikipedia_url = info.get("wikipedia_url", "")
            results.append((entity_name, wikipedia_url, earliest, latest))
        return results

    # -------------------------------------------------------------- export

    def export_csv(
        self,
        output_file: str,
        compress: bool = False,
        include_no_dates: bool = False,
    ) -> Path:
        """Export to CSV, optionally zstd-compressed.  Returns final path."""
        results = self.get_results()
        out_path = Path(output_file)

        fh, final_path = _open_output(out_path, "w", compress=compress)
        try:
            writer = csv.writer(fh)
            writer.writerow(
                ["Entity", "Wikipedia_URL", "Earliest_Date", "Latest_Date"]
            )
            for entity, wiki_url, earliest, latest in results:
                if earliest or include_no_dates:
                    writer.writerow(
                        [entity, wiki_url or "", earliest or "0", latest or "0"]
                    )
        finally:
            fh.close()

        size = final_path.stat().st_size
        print(
            f"  Exported {len(results):,} entities to {final_path} "
            f"({_sizeof_fmt(size)})"
        )
        return final_path

    def export_json(
        self,
        output_file: str,
        compress: bool = False,
    ) -> Path:
        """Export to JSON, optionally zstd-compressed.  Returns final path."""
        results = self.get_results()
        out_path = Path(output_file)

        data = [
            {
                "entity": entity,
                "wikipedia_url": wiki_url or "",
                "earliest_date": earliest or "0",
                "latest_date": latest or "0",
            }
            for entity, wiki_url, earliest, latest in results
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
        results = self.get_results()
        print(f"\n{'=' * 80}")
        print("YAGO Time Extraction Summary")
        print(f"{'=' * 80}")
        print(f"Total entities with time data: {len(results)}")
        print(f"\nShowing first {min(limit, len(results))} entities:\n")

        print(f"{'Entity':<40} {'Earliest Date':<15} {'Latest Date':<15}")
        print(f"{'-' * 70}")
        for entity, _url, earliest, latest in results[:limit]:
            display = entity[:37] + "..." if len(entity) > 40 else entity
            print(f"{display:<40} {earliest or 'N/A':<15} {latest or 'N/A':<15}")
        if len(results) > limit:
            print(f"\n... and {len(results) - limit} more entities")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description=(
            "YAGO temporal metadata pipeline — "
            "download, extract, parse, compress, reclaim"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All five stages run by default.  Each stage is skipped automatically
when its output file (or a .reclaim marker) already exists.

Typical usage:
  python yago_parser.py --verbose            # run full pipeline
  python yago_parser.py --force --verbose    # redo everything from scratch

Skip optional stages:
  python yago_parser.py --no-compress        # stop after plain CSV
  python yago_parser.py --no-reclaim         # keep intermediate files

Individual stages:
  python yago_parser.py --download-only
  python yago_parser.py --extract-only
  python yago_parser.py --parse-only

Legacy (positional TTL path still works):
  python yago_parser.py /path/to/yago-facts.ttl --csv out.csv -v
        """,
    )

    # Directories / files
    default_yago_dir = os.path.join(
        os.environ.get("WIKI_DATA", "/mnt/data/wikipedia"), "yago"
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
        "--yago-dir",
        default=default_yago_dir,
        help=f"Working directory (default: {default_yago_dir})",
    )
    parser.add_argument(
        "--url", default=YAGO_URL, help="YAGO zip download URL"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV output path (default: <yago-dir>/yago-facts.csv)",
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

    # Deprecated flags (compress & reclaim are now defaults).
    # Accepted silently for backward compatibility.
    parser.add_argument(
        "--compress", action="store_true", default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reclaim", action="store_true", default=False,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    yago_dir = Path(args.yago_dir)
    yago_dir.mkdir(parents=True, exist_ok=True)

    csv_out = args.csv or str(yago_dir / DEFAULT_CSV_NAME)

    # Key paths used across stages
    zip_path = yago_dir / YAGO_ZIP_NAME
    ttl_path = yago_dir / YAGO_TTL_MEMBER
    csv_path = Path(csv_out)
    csv_zst_path = csv_path.with_suffix(csv_path.suffix + ".zst")

    t0 = time.time()

    try:
        # --force: clear every .reclaim marker so all stages re-run
        if args.force:
            print("[prep] Clearing reclaim markers for fresh run")
            _clean_reclaim_markers(yago_dir)

        # ----------------------------------------------------------
        # [1/5] Download
        # ----------------------------------------------------------
        if args.ttl_file:
            print(f"[1/{STAGE_TOTAL}] Download — skipped (TTL provided)")
        elif args.parse_only:
            print(f"[1/{STAGE_TOTAL}] Download — skipped (--parse-only)")
        elif _stage_complete(zip_path) and not args.force:
            _print_skip(1, "Download", zip_path)
        else:
            print(f"[1/{STAGE_TOTAL}] Download")
            zip_path = download_yago(
                yago_dir, url=args.url, force=args.force
            )
        if args.download_only:
            print(f"\nDone in {time.time() - t0:.1f}s")
            return

        # ----------------------------------------------------------
        # [2/5] Extract
        # ----------------------------------------------------------
        if args.ttl_file:
            ttl_path = Path(args.ttl_file)
            print(f"[2/{STAGE_TOTAL}] Extract — skipped (TTL provided)")
        elif args.parse_only:
            print(f"[2/{STAGE_TOTAL}] Extract — skipped (--parse-only)")
        elif _stage_complete(ttl_path) and not args.force:
            _print_skip(2, "Extract", ttl_path)
        else:
            print(f"[2/{STAGE_TOTAL}] Extract")
            if not zip_path.exists():
                print(
                    f"  Error: zip not found at {zip_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
            ttl_path = extract_yago(zip_path, yago_dir, force=args.force)
        if args.extract_only:
            print(f"\nDone in {time.time() - t0:.1f}s")
            return

        # ----------------------------------------------------------
        # [3/5] Parse
        # ----------------------------------------------------------
        parse_done = (
            _stage_complete(csv_path) or csv_zst_path.exists()
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

            extractor = YagoTimeExtractor(str(ttl_path))
            extractor.parse_file(verbose=args.verbose)

            # Always produce the plain CSV first; compression is a
            # separate stage below.
            extractor.export_csv(csv_out, compress=False)

            if args.json:
                extractor.export_json(args.json, compress=False)

            if not args.no_summary:
                extractor.print_summary(limit=args.limit)

        if args.parse_only:
            print(f"\nDone in {time.time() - t0:.1f}s")
            return

        # ----------------------------------------------------------
        # [4/5] Compress
        # ----------------------------------------------------------
        if args.no_compress:
            print(f"[4/{STAGE_TOTAL}] Compress — skipped (--no-compress)")
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
            print(f"[5/{STAGE_TOTAL}] Reclaim — skipped (--no-reclaim)")
        else:
            print(f"[5/{STAGE_TOTAL}] Reclaim")

            # Build list of intermediate files eligible for reclamation
            reclaimable = [zip_path, ttl_path]

            # Only reclaim the plain CSV when the compressed version exists
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
