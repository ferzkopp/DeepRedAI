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
from datetime import datetime
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
    
    def __init__(self, ttl_file_path: str):
        """Initialize the parser with the path to a TTL file"""
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
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string from Wikidata TTL format
        
        Args:
            date_str: Date string in format like "1732-02-22T00:00:00Z"^^xsd:dateTime
            
        Returns:
            datetime object or None if parsing fails
        """
        # Extract the date portion from quotes
        match = re.search(r'"([^"]+)"', date_str)
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
        match = re.search(r'/entity/(Q\d+)', entity_str)
        if match:
            return match.group(1)
        
        # Format 3: just Q123
        if re.match(r'^Q\d+$', entity_str):
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
            match = re.search(r'schema:about\s+(wd:Q\d+)', line)
            if match:
                entity_id = self.extract_entity_id(match.group(1))
                return (entity_id, None)  # URL will be extracted from context
        
        # Look for English Wikipedia URL in the same block
        # Format: <https://en.wikipedia.org/wiki/Title>
        if 'en.wikipedia.org/wiki/' in line:
            match = re.search(r'<(https?://en\.wikipedia\.org/wiki/[^>]+)>', line)
            if match:
                url = match.group(1)
                # Decode URL-encoded characters
                url = unquote(url)
                return (None, url)
        
        return None
    
    def parse_file(self, verbose: bool = False) -> None:
        """
        Parse the TTL file line by line to extract time information
        
        Args:
            verbose: If True, print progress information
        """
        if verbose:
            print(f"Parsing {self.ttl_file_path}...")
            print(f"Looking for time properties: {', '.join(self.TIME_PROPERTIES)}")
        
        line_count = 0
        entities_found = 0
        dates_found = 0
        wikipedia_links_found = 0
        
        # Track current subject for multi-line statements
        current_subject = None
        current_entity_id = None
        
        # Track current sitelink info (when Wikipedia URL is the subject)
        current_sitelink_url = None
        current_sitelink_title = None
        current_sitelink_entity = None
        
        with open(self.ttl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                original_line = line
                line = line.strip()
                
                # Progress indicator
                if verbose and line_count % 1000000 == 0:
                    print(f"  Processed {line_count:,} lines | "
                          f"Entities: {entities_found:,} | "
                          f"Dates: {dates_found:,} | "
                          f"Wikipedia links: {wikipedia_links_found:,}")
                
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
                            match = re.search(r'<(https?://en\.wikipedia\.org/wiki/[^>]+)>', line)
                            if match:
                                url = unquote(match.group(1))
                                # Extract title from URL
                                title_match = re.search(r'/wiki/(.+)$', url)
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
                for time_prop in self.TIME_PROPERTIES:
                    if time_prop in line and current_entity_id:
                        # Extract date from the line
                        date_obj = self.parse_date(line)
                        if date_obj:
                            # Track new entity
                            if current_entity_id not in self.entity_dates:
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
                    match = re.search(r'schema:about\s+wd:(Q\d+)', line)
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
        
        if verbose:
            print(f"\nParsing complete!")
            print(f"  Total lines processed: {line_count:,}")
            print(f"  Entities with time data: {entities_found:,}")
            print(f"  Total dates extracted: {dates_found:,}")
            print(f"  Wikipedia links found: {wikipedia_links_found:,}")
            print(f"  Entities with both dates and Wikipedia links: "
                  f"{len(self.entities_with_wiki & set(self.entity_dates.keys())):,}")
    
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
        
        Args:
            output_file: Path to output CSV file
            wikipedia_only: If True, only export entities with Wikipedia links
        """
        import csv
        
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
        import json
        
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
  
  # Export to CSV (Wikipedia entities only)
  %(prog)s /mnt/data/wikipedia/wikidata/wikidata-20251215-all-BETA.ttl --csv output.csv --verbose
  
  # Export to JSON (all entities, even without Wikipedia links)
  %(prog)s /mnt/data/wikipedia/wikidata/wikidata-20251215-all-BETA.ttl --json output.json --all-entities
  
  # Export both formats
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
    
    args = parser.parse_args()
    
    # Validate that at least one output option is specified
    if not (args.csv or args.json or args.summary):
        parser.error("At least one output option required: --csv, --json, or --summary")
    
    try:
        # Create extractor and parse file
        extractor = WikidataTimeExtractor(args.ttl_file)
        extractor.parse_file(verbose=args.verbose)
        
        # Export to CSV if requested
        if args.csv:
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
