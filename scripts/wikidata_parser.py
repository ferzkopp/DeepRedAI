#!/usr/bin/env python3
"""
Wikidata TTL Parser - Extract time-related metadata from Wikidata

This script parses Wikidata .ttl files (Turtle RDF format) to extract time-related
information (birth dates P569, death dates P570, inception dates P571, dissolution dates P576)
and generates output with earliest and latest dates for each Wikipedia entity.

The parser is designed to handle large files (about 900 GB) without loading them entirely into memory.

Output format is compatible with YAGO pipeline for temporal augmentation.

Full dataset: https://dumps.wikimedia.org/wikidatawiki/entities/
"""

import re
import sys
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from urllib.parse import unquote


class WikidataTimeExtractor:
    """Extract time-related metadata from Wikidata TTL files"""
    
    # Wikidata time-related properties
    # P569: date of birth
    # P570: date of death
    # P571: inception (founding, establishment)
    # P576: dissolved, abolished or demolished date
    TIME_PROPERTIES = {
        'wdt:P569',  # birth date
        'wdt:P570',  # death date
        'wdt:P571',  # inception date
        'wdt:P576',  # dissolution date
    }
    
    def __init__(self, ttl_file_path: str, csv_output_file: Optional[str] = None,
                 checkpoint_file: Optional[str] = None, checkpoint_interval: int = 1000000):
        """Initialize the parser with the path to a TTL file
        
        Args:
            ttl_file_path: Path to the TTL file to parse
            csv_output_file: Path to CSV output file (for incremental writing)
            checkpoint_file: Path to checkpoint file for resume capability
            checkpoint_interval: Number of lines between checkpoints and incremental saves
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
        self._entity_pattern = re.compile(r'/entity/(Q\d+)')
        self._qid_pattern = re.compile(r'^Q\d+$')
        self._wiki_url_pattern = re.compile(r'<(https?://en\.wikipedia\.org/wiki/[^>]+)>')
        self._schema_about_pattern = re.compile(r'schema:about\s+wd:(Q\d+)')
        self._title_pattern = re.compile(r'/wiki/(.+)$')
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string from Wikidata TTL format
        
        Args:
            date_str: Date string in format like "1732-02-22T00:00:00Z"^^xsd:dateTime
            
        Returns:
            datetime object or None if parsing fails
        """
        # Extract the date portion from quotes
        match = self._date_pattern.search(date_str)
        if not match:
            return None
        
        date_value = match.group(1)
        
        # Try to parse various date formats
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d',
            '%Y-%m',
            '%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_value, fmt)
            except ValueError:
                continue
        
        return None
    
    def initialize_csv_output(self, wikipedia_only: bool = True) -> None:
        """Initialize CSV output file and write header
        
        This checks file permissions early before processing starts.
        
        Args:
            wikipedia_only: If True, only entities with Wikipedia links will be written
        
        Raises:
            PermissionError: If the output file cannot be written
        """
        if not self.csv_output_file:
            return
        
        # Try to open the file to check permissions
        try:
            # Check if file exists - if so, we're resuming
            file_exists = os.path.exists(self.csv_output_file)
            
            if file_exists:
                # Open in append mode for resume
                self.csv_file_handle = open(self.csv_output_file, 'a', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file_handle)
                print(f"Resuming: Appending to existing file {self.csv_output_file}")
            else:
                # Create new file and write header
                self.csv_file_handle = open(self.csv_output_file, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file_handle)
                self.csv_writer.writerow(['Entity_ID', 'Entity', 'Wikipedia_URL', 'Earliest_Date', 'Latest_Date'])
                self.csv_file_handle.flush()
                print(f"Created output file with header: {self.csv_output_file}")
        except PermissionError as e:
            raise PermissionError(f"Cannot write to output file {self.csv_output_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error initializing output file {self.csv_output_file}: {e}")
    
    def close_csv_output(self) -> None:
        """Close the CSV output file"""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None
    
    def load_checkpoint(self) -> dict:
        """Load checkpoint data to resume from last position
        
        Returns:
            Dictionary with checkpoint data (empty dict if no checkpoint)
        """
        if not self.checkpoint_file or not os.path.exists(self.checkpoint_file):
            return {}
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                line_num = data.get('line_number', 0)
                self.written_entities = set(data.get('written_entities', []))
                # Load cached total_lines to avoid recounting
                self.total_lines = data.get('total_lines', 0)
                print(f"Loaded checkpoint: Resuming from line {line_num:,}")
                print(f"  Already written {len(self.written_entities):,} entities")
                if self.total_lines > 0:
                    print(f"  Total lines (cached): {self.total_lines:,}")
                # Load counters
                entities_found = data.get('entities_found', 0)
                dates_found = data.get('dates_found', 0)
                wikipedia_links_found = data.get('wikipedia_links_found', 0)
                total_written = data.get('total_written', len(self.written_entities))
                if entities_found > 0 or dates_found > 0 or wikipedia_links_found > 0:
                    print(f"  Entities: {entities_found:,} | Dates: {dates_found:,} | Wikipedia: {wikipedia_links_found:,}")
                return {
                    'line_number': line_num,
                    'entities_found': entities_found,
                    'dates_found': dates_found,
                    'wikipedia_links_found': wikipedia_links_found,
                    'total_written': total_written
                }
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return {}
    
    def save_checkpoint(self, line_number: int, entities_found: int = 0, 
                         dates_found: int = 0, wikipedia_links_found: int = 0,
                         total_written: int = 0) -> None:
        """Save checkpoint data
        
        Args:
            line_number: Current line number being processed
            entities_found: Number of entities found so far
            dates_found: Number of dates found so far
            wikipedia_links_found: Number of Wikipedia links found so far
            total_written: Total entities written to CSV
        """
        if not self.checkpoint_file:
            return
        
        try:
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.checkpoint_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump({
                    'line_number': line_number,
                    'written_entities': list(self.written_entities),
                    'total_lines': self.total_lines,  # Cache total_lines to avoid recounting
                    'entities_found': entities_found,
                    'dates_found': dates_found,
                    'wikipedia_links_found': wikipedia_links_found,
                    'total_written': total_written,
                    'timestamp': datetime.now().isoformat()
                }, f)
            os.replace(temp_file, self.checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def count_file_lines(self, verbose: bool = False) -> int:
        """Count total lines in file using fast buffered read method
        
        Uses raw binary buffered reading which benchmarks show is faster than
        wc -l or grep -c for large files. For a 900GB file, this approach can
        be significantly faster than subprocess-based methods.
        
        Args:
            verbose: If True, print progress information
        
        Returns:
            Total number of lines in file
        """
        if verbose:
            print(f"Counting lines in {self.ttl_file_path}...")
            start_time = time.time()
        
        try:
            # Fast buffered read method - benchmarks show this is faster than wc -l
            # Uses raw binary I/O with 64KB buffer for optimal performance
            def _make_gen(reader):
                buf_size = 2 ** 16  # 64KB buffer - sweet spot for performance
                b = reader(buf_size)
                while b:
                    yield b
                    b = reader(buf_size)
            
            with open(self.ttl_file_path, "rb") as f:
                line_count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"Total lines: {line_count:,} (counted in {elapsed:.1f}s)")
            
            return line_count
        except Exception as e:
            if verbose:
                print(f"Warning: Could not count lines: {e}")
                print("Will proceed without total line count (progress % not available)")
            return 0
    
    def fast_skip_lines(self, file_handle, target_line: int, verbose: bool = False) -> int:
        """Skip to a target line number using fast binary newline counting
        
        This is MUCH faster than reading lines with `for line in f` when we don't
        need the line content. Uses binary buffered reading to count newlines.
        
        Args:
            file_handle: Open file handle (opened in binary mode 'rb')
            target_line: Line number to skip to (1-indexed)
            verbose: If True, print progress information
        
        Returns:
            Actual number of lines skipped (may be less than target if EOF reached)
        """
        if target_line <= 0:
            return 0
        
        if verbose:
            print(f"Fast-skipping to line {target_line:,}...")
            start_time = time.time()
            last_report = start_time
        
        lines_counted = 0
        buf_size = 2 ** 20  # 1MB buffer for fast reading
        
        while lines_counted < target_line:
            buf = file_handle.read(buf_size)
            if not buf:
                break  # EOF
            
            newlines_in_buf = buf.count(b"\n")
            
            # Check if target is within this buffer
            if lines_counted + newlines_in_buf >= target_line:
                # Need to find exact position within buffer
                remaining_to_skip = target_line - lines_counted
                pos = 0
                for _ in range(remaining_to_skip):
                    next_newline = buf.find(b"\n", pos)
                    if next_newline == -1:
                        break
                    pos = next_newline + 1
                
                # Seek back to the position after the target line
                bytes_to_seek_back = len(buf) - pos
                file_handle.seek(-bytes_to_seek_back, 1)  # 1 = SEEK_CUR
                lines_counted = target_line
                break
            
            lines_counted += newlines_in_buf
            
            # Progress report every 10 seconds
            if verbose and time.time() - last_report > 10:
                elapsed = time.time() - start_time
                pct = (lines_counted / target_line) * 100 if target_line > 0 else 0
                rate = lines_counted / elapsed if elapsed > 0 else 0
                eta = (target_line - lines_counted) / rate if rate > 0 else 0
                print(f"  Skipped {lines_counted:,} / {target_line:,} lines ({pct:.1f}%) | ETA: {self.format_time_remaining(eta)}")
                last_report = time.time()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  Skipped {lines_counted:,} lines in {elapsed:.1f}s")
        
        return lines_counted
    
    def format_time_remaining(self, seconds: float) -> str:
        """Format seconds into human-readable time remaining
        
        Args:
            seconds: Time in seconds
        
        Returns:
            Formatted string like "2h 15m" or "45m 30s"
        """
        if seconds < 0:
            return "calculating..."
        
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        
        if td.days > 0:
            return f"{td.days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def write_incremental_results(self, wikipedia_only: bool = True) -> int:
        """Write new results incrementally to CSV file
        
        Args:
            wikipedia_only: If True, only write entities with Wikipedia links
        
        Returns:
            Number of entities written
        """
        if not self.csv_writer:
            return 0
        
        written_count = 0
        
        # Find entities that have data and haven't been written yet
        for entity_id in sorted(self.entity_dates.keys()):
            if entity_id in self.written_entities:
                continue
            
            dates = self.entity_dates[entity_id]
            info = self.entity_info.get(entity_id, {})
            
            wikipedia_url = info.get('wikipedia_url')
            wikipedia_title = info.get('wikipedia_title', '')
            
            # Skip entities without Wikipedia links if requested
            if wikipedia_only and not wikipedia_url:
                continue
            
            earliest = min(dates).strftime('%Y-%m-%d') if dates else '0'
            latest = max(dates).strftime('%Y-%m-%d') if dates else '0'
            
            self.csv_writer.writerow([
                entity_id,
                wikipedia_title or entity_id,
                wikipedia_url or '',
                earliest,
                latest
            ])
            
            self.written_entities.add(entity_id)
            written_count += 1
        
        # Flush to disk
        if written_count > 0:
            self.csv_file_handle.flush()
        
        return written_count
    
    def extract_entity_id(self, entity_str: str) -> Optional[str]:
        """
        Extract Wikidata entity ID from various formats
        
        Args:
            entity_str: String like "wd:Q23" or "<http://www.wikidata.org/entity/Q23>"
            
        Returns:
            Entity ID like "Q23" or None
        """
        # Format 1: wd:Q123
        if entity_str.startswith('wd:'):
            return entity_str[3:]
        
        # Format 2: <http://www.wikidata.org/entity/Q123>
        match = self._entity_pattern.search(entity_str)
        if match:
            return match.group(1)
        
        # Format 3: just Q123
        if self._qid_pattern.match(entity_str):
            return entity_str
        
        return None
    
    def parse_wikipedia_url(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Extract Wikipedia URL and entity ID from sitelink lines
        
        Args:
            line: Line containing schema:about or Wikipedia URL
            
        Returns:
            Tuple of (entity_id, wikipedia_url) or None
        """
        # Look for schema:about pattern: schema:about wd:Q31 ;
        if 'schema:about' in line:
            # Extract the entity ID
            match = self._schema_about_pattern.search(line)
            if match:
                entity_id = self.extract_entity_id(match.group(1))
                return (entity_id, None)  # URL will be extracted from context
        
        # Look for English Wikipedia URL in the same block
        # Format: <https://en.wikipedia.org/wiki/Title>
        if 'en.wikipedia.org/wiki/' in line:
            match = self._wiki_url_pattern.search(line)
            if match:
                url = match.group(1)
                # Decode URL-encoded characters
                url = unquote(url)
                return (None, url)
        
        return None
    
    def parse_file(self, verbose: bool = False, wikipedia_only: bool = True) -> None:
        """
        Parse the TTL file line by line to extract time information
        
        Args:
            verbose: If True, print progress information
            wikipedia_only: If True, only save entities with Wikipedia links during incremental writes
        """
        # Load checkpoint if available (this may also load cached total_lines)
        checkpoint_data = self.load_checkpoint()
        start_line = checkpoint_data.get('line_number', 0)
        
        # Count total lines for progress reporting (only if not already cached)
        if verbose and self.total_lines == 0:
            self.total_lines = self.count_file_lines(verbose=True)
        
        # Initialize timing
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        
        if verbose:
            print(f"Parsing {self.ttl_file_path}...")
            print(f"Looking for time properties: {', '.join(self.TIME_PROPERTIES)}")
            if start_line > 0:
                print(f"Resuming from line {start_line:,}")
        
        line_count = 0
        # Restore counters from checkpoint
        entities_found = checkpoint_data.get('entities_found', 0)
        dates_found = checkpoint_data.get('dates_found', 0)
        wikipedia_links_found = checkpoint_data.get('wikipedia_links_found', 0)
        total_written = checkpoint_data.get('total_written', len(self.written_entities))
        
        # Track current subject for multi-line statements
        current_subject = None
        current_entity_id = None
        
        # Track current sitelink info (when Wikipedia URL is the subject)
        current_sitelink_url = None
        current_sitelink_title = None
        current_sitelink_entity = None
        
        with open(self.ttl_file_path, 'rb') as f:
            # Fast-skip to checkpoint position using binary newline counting
            if start_line > 0:
                line_count = self.fast_skip_lines(f, start_line, verbose=verbose)
            
            # Now wrap in a text reader for line-by-line parsing
            import io
            text_reader = io.TextIOWrapper(f, encoding='utf-8')
            
            for line in text_reader:
                line_count += 1
                
                original_line = line
                line = line.strip()
                
                # Progress indicator and incremental save
                if verbose and line_count % 5000000 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.start_time
                    
                    # Calculate progress percentage and ETA
                    progress_msg = f"  Processed {line_count:,} lines"
                    if self.total_lines > 0:
                        progress_pct = (line_count / self.total_lines) * 100
                        progress_msg += f" ({progress_pct:.1f}%)"
                        
                        # Estimate time remaining
                        if line_count > start_line:
                            lines_processed = line_count - start_line
                            lines_remaining = self.total_lines - line_count
                            rate = lines_processed / elapsed
                            if rate > 0:
                                eta_seconds = lines_remaining / rate
                                progress_msg += f" | ETA: {self.format_time_remaining(eta_seconds)}"
                    
                    progress_msg += f" | Entities: {entities_found:,} | Dates: {dates_found:,} | Wikipedia: {wikipedia_links_found:,}"
                    print(progress_msg)
                
                # Incremental save and checkpoint
                if line_count % self.checkpoint_interval == 0:
                    if self.csv_writer:
                        newly_written = self.write_incremental_results(wikipedia_only=wikipedia_only)
                        total_written += newly_written
                    
                    self.save_checkpoint(line_count, entities_found, dates_found, 
                                        wikipedia_links_found, total_written)
                    self.current_line = line_count
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Skip prefix declarations
                if line.startswith('@prefix') or line.startswith('@base'):
                    continue
                
                # Check if this line starts a new subject (no leading whitespace)
                if original_line and not original_line[0].isspace():
                    # New subject - parse it
                    parts = line.split(None, 1)
                    if parts:
                        current_subject = parts[0]
                        current_entity_id = self.extract_entity_id(current_subject)
                        
                        # Check if subject is an English Wikipedia URL
                        if '<https://en.wikipedia.org/wiki/' in line:
                            match = self._wiki_url_pattern.search(line)
                            if match:
                                url = unquote(match.group(1))
                                # Extract title from URL
                                title_match = self._title_pattern.search(url)
                                title = title_match.group(1) if title_match else None
                                # Store this as the current sitelink URL (entity will be found via schema:about)
                                current_sitelink_entity = None  # Will be set when we find schema:about
                                current_sitelink_url = url
                                current_sitelink_title = title
                            else:
                                current_sitelink_url = None
                                current_sitelink_title = None
                        else:
                            current_sitelink_url = None
                            current_sitelink_title = None
                
                # Look for lines with time properties (using current subject)
                # Fast pre-check: all time properties start with 'wdt:P5'
                if current_entity_id and 'wdt:P5' in line:
                    # Now check specific properties
                    for time_prop in self.TIME_PROPERTIES:
                        if time_prop in line:
                            # Extract date from the line
                            date_obj = self.parse_date(line)
                            if date_obj:
                                # Track new entity
                                is_new_entity = current_entity_id not in self.entity_dates
                                if is_new_entity:
                                    entities_found += 1
                                    # Only create entity_info if it doesn't exist
                                    if current_entity_id not in self.entity_info:
                                        self.entity_info[current_entity_id] = {
                                            'wikipedia_url': None,
                                            'wikipedia_title': None
                                        }
                                
                                self.entity_dates[current_entity_id].append(date_obj)
                                dates_found += 1
                            break
                
                # Look for schema:about when we have a Wikipedia URL as subject
                if current_sitelink_url and 'schema:about' in line:
                    match = self._schema_about_pattern.search(line)
                    if match:
                        entity_id = match.group(1)
                        
                        # Store Wikipedia info for this entity
                        if entity_id not in self.entity_info:
                            self.entity_info[entity_id] = {
                                'wikipedia_url': current_sitelink_url,
                                'wikipedia_title': current_sitelink_title
                            }
                        else:
                            self.entity_info[entity_id]['wikipedia_url'] = current_sitelink_url
                            self.entity_info[entity_id]['wikipedia_title'] = current_sitelink_title
                        
                        self.entities_with_wiki.add(entity_id)
                        wikipedia_links_found += 1
                
                # Reset subject and sitelink info at the end of a statement block (period)
                if line.endswith('.'):
                    current_subject = None
                    current_entity_id = None
                    current_sitelink_url = None
                    current_sitelink_title = None
                    current_sitelink_entity = None
        
        # Final incremental save
        if self.csv_writer:
            newly_written = self.write_incremental_results(wikipedia_only=wikipedia_only)
            total_written += newly_written
            if verbose and newly_written > 0:
                print(f"    → Final save: {newly_written:,} new entities written")
        
        # Save final checkpoint
        self.save_checkpoint(line_count, entities_found, dates_found,
                            wikipedia_links_found, total_written)
        self.current_line = line_count
        
        if verbose:
            print(f"\nParsing complete!")
            print(f"  Total lines processed: {line_count:,}")
            print(f"  Entities with time data: {entities_found:,}")
            print(f"  Total dates extracted: {dates_found:,}")
            print(f"  Wikipedia links found: {wikipedia_links_found:,}")
            print(f"  Entities with both dates and Wikipedia links: "
                  f"{len(self.entities_with_wiki & set(self.entity_dates.keys())):,}")
            if self.csv_output_file:
                print(f"  Total entities written to CSV: {total_written:,}")
    
    def get_results(self, wikipedia_only: bool = True) -> List[Tuple[str, str, str, Optional[str], Optional[str]]]:
        """
        Get parsing results as a list of tuples
        
        Args:
            wikipedia_only: If True, only return entities with Wikipedia links
        
        Returns:
            List of (entity_id, wikipedia_title, wikipedia_url, earliest_date, latest_date)
        """
        results = []
        
        for entity_id in sorted(self.entity_dates.keys()):
            dates = self.entity_dates[entity_id]
            info = self.entity_info.get(entity_id, {})
            
            wikipedia_url = info.get('wikipedia_url')
            wikipedia_title = info.get('wikipedia_title', '')
            
            # Skip entities without Wikipedia links if requested
            if wikipedia_only and not wikipedia_url:
                continue
            
            earliest = min(dates).strftime('%Y-%m-%d') if dates else None
            latest = max(dates).strftime('%Y-%m-%d') if dates else None
            
            results.append((entity_id, wikipedia_title, wikipedia_url or '', earliest, latest))
        
        return results
    
    def export_csv(self, output_file: str, wikipedia_only: bool = True) -> None:
        """
        Export results to a CSV file (compatible with YAGO format)
        
        NOTE: This method exports all results at once. For incremental saving,
        use initialize_csv_output() before parsing and the results will be
        written incrementally during parse_file().
        
        Args:
            output_file: Path to output CSV file
            wikipedia_only: If True, only export entities with Wikipedia links
        """
        results = self.get_results(wikipedia_only=wikipedia_only)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Entity_ID', 'Entity', 'Wikipedia_URL', 'Earliest_Date', 'Latest_Date'])
            
            for entity_id, wiki_title, wiki_url, earliest, latest in results:
                writer.writerow([
                    entity_id,
                    wiki_title or entity_id,
                    wiki_url,
                    earliest or '0',
                    latest or '0'
                ])
        
        print(f"Exported {len(results):,} entities to {output_file}")
    
    def export_json(self, output_file: str, wikipedia_only: bool = True) -> None:
        """
        Export results to a JSON file
        
        Args:
            output_file: Path to output JSON file
            wikipedia_only: If True, only export entities with Wikipedia links
        """
        results = self.get_results(wikipedia_only=wikipedia_only)
        
        data = []
        for entity_id, wiki_title, wiki_url, earliest, latest in results:
            data.append({
                'entity_id': entity_id,
                'entity': wiki_title or entity_id,
                'wikipedia_url': wiki_url,
                'earliest_date': earliest or '0',
                'latest_date': latest or '0'
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(results):,} entities to {output_file}")
    
    def print_summary(self, limit: int = 20) -> None:
        """
        Print a summary of extracted data
        
        Args:
            limit: Maximum number of entities to display
        """
        results = self.get_results(wikipedia_only=True)
        
        print(f"\n{'='*100}")
        print(f"Wikidata Time Extraction Summary")
        print(f"{'='*100}")
        print(f"Total entities with time data and Wikipedia links: {len(results):,}")
        print(f"\nShowing first {min(limit, len(results))} entities:\n")
        
        print(f"{'Entity ID':<12} {'Wikipedia Title':<40} {'Earliest':<12} {'Latest':<12}")
        print(f"{'-'*80}")
        
        for entity_id, wiki_title, wiki_url, earliest, latest in results[:limit]:
            title_display = wiki_title[:37] + '...' if len(wiki_title) > 40 else wiki_title
            print(f"{entity_id:<12} {title_display:<40} {earliest or 'N/A':<12} {latest or 'N/A':<12}")
        
        if len(results) > limit:
            print(f"\n... and {len(results) - limit:,} more entities")


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract time-related metadata from Wikidata TTL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse and show summary
  %(prog)s /mnt/data/wikipedia/wikidata/wikidata-20251215-all-BETA.ttl --summary
  
  # Export to CSV (checkpoint/resume enabled by default)
  %(prog)s /mnt/data/wikipedia/wikidata/wikidata-20251215-all-BETA.ttl --csv output.csv --verbose
  
  # Export to CSV without checkpoint (not recommended for large files)
  %(prog)s /mnt/data/wikipedia/wikidata/wikidata-20251215-all-BETA.ttl --csv output.csv --no-checkpoint --verbose
  
  # Export to JSON (all entities, even without Wikipedia links)
  %(prog)s /mnt/data/wikipedia/wikidata/wikidata-20251215-all-BETA.ttl --json output.json --all-entities
  
  # Export both formats (checkpoint enabled for CSV)
  %(prog)s /mnt/data/wikipedia/wikidata/wikidata-20251215-all-BETA.ttl --csv output.csv --json output.json --verbose
        """
    )
    
    parser.add_argument('ttl_file', help='Path to Wikidata TTL file')
    parser.add_argument('--csv', help='Export results to CSV file')
    parser.add_argument('--json', help='Export results to JSON file')
    parser.add_argument('--summary', action='store_true', help='Print summary of results')
    parser.add_argument('--limit', type=int, default=20, help='Limit summary output (default: 20)')
    parser.add_argument('--all-entities', action='store_true', 
                        help='Include entities without Wikipedia links in output')
    parser.add_argument('--verbose', action='store_true', help='Print progress information')
    parser.add_argument('--checkpoint', help='Checkpoint file for resume capability (default: <csv_file>.checkpoint, auto-enabled for CSV output)')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable automatic checkpoint/incremental mode for CSV output')
    parser.add_argument('--checkpoint-interval', type=int, default=1000000,
                        help='Number of lines between checkpoints and incremental saves (default: 1,000,000)')
    parser.add_argument('--restart', action='store_true',
                        help='Start fresh by removing existing checkpoint and output files')
    
    args = parser.parse_args()
    
    # Validate that at least one output option is specified
    if not (args.csv or args.json or args.summary):
        parser.error("At least one output option required: --csv, --json, or --summary")
    
    try:
        # Determine checkpoint file (enabled by default for CSV output)
        checkpoint_file = None
        if args.csv and not args.no_checkpoint:
            # Checkpoint enabled by default for CSV output
            checkpoint_file = args.checkpoint if args.checkpoint else args.csv + '.checkpoint'
        elif args.checkpoint:
            # Explicit checkpoint file provided
            checkpoint_file = args.checkpoint
        
        # Handle --restart: remove existing checkpoint and output files
        if args.restart:
            files_removed = []
            if checkpoint_file and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                files_removed.append(checkpoint_file)
            if args.csv and os.path.exists(args.csv):
                os.remove(args.csv)
                files_removed.append(args.csv)
            if files_removed:
                print(f"Restart: Removed {', '.join(files_removed)}")
            else:
                print("Restart: No existing files to remove, starting fresh")
        
        # Determine if we should do incremental CSV writing
        incremental_csv = args.csv and checkpoint_file is not None
        
        # Create extractor with checkpoint support
        csv_output = args.csv if incremental_csv else None
        extractor = WikidataTimeExtractor(
            args.ttl_file,
            csv_output_file=csv_output,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Initialize CSV output if using incremental mode
        if incremental_csv:
            extractor.initialize_csv_output(wikipedia_only=not args.all_entities)
        
        try:
            # Parse file (with incremental saving if enabled)
            extractor.parse_file(verbose=args.verbose, wikipedia_only=not args.all_entities)
        finally:
            # Always close CSV file if it was opened
            extractor.close_csv_output()
        
        # Export to CSV if requested (non-incremental mode)
        if args.csv and not incremental_csv:
            extractor.export_csv(args.csv, wikipedia_only=not args.all_entities)
        
        # Export to JSON if requested
        if args.json:
            extractor.export_json(args.json, wikipedia_only=not args.all_entities)
        
        # Print summary if requested
        if args.summary:
            extractor.print_summary(limit=args.limit)
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
