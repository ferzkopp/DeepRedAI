#!/usr/bin/env python3
"""
YAGO TTL Parser - Extract time-related metadata from YAGO knowledge base

This script parses YAGO .ttl files (Turtle RDF format) to extract time-related
information (birthDate, deathDate, startDate, endDate, datePublished) and
generates a summary with earliest and latest dates for each Wikipedia entity.

The parser is designed to handle large files without loading them entirely into memory.

Full dataset: https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip
"""

import re
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class YagoTimeExtractor:
    """Extract time-related metadata from YAGO TTL files"""
    
    # Schema.org time-related predicates
    TIME_PREDICATES = {
        'schema:birthDate',
        'schema:deathDate', 
        'schema:startDate',
        'schema:endDate',
        'schema:datePublished'
    }
    
    def __init__(self, ttl_file_path: str):
        """Initialize the parser with the path to a TTL file"""
        self.ttl_file_path = Path(ttl_file_path)
        if not self.ttl_file_path.exists():
            raise FileNotFoundError(f"TTL file not found: {ttl_file_path}")
        
        # Store entity dates: entity -> list of dates
        self.entity_dates: Dict[str, List[datetime]] = defaultdict(list)
        
        # Store entity to Wikipedia info mapping
        self.entity_info: Dict[str, Dict] = {}
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string from TTL format
        
        Args:
            date_str: Date string in format like "1915-11-25T00:00:00Z"^^xsd:dateTime
            
        Returns:
            datetime object or None if parsing fails
        """
        # Extract the date portion (remove type annotation)
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
    
    def decode_yago_entity_name(self, name: str) -> str:
        """
        Decode YAGO entity names that contain Unicode escape sequences.
        
        Converts patterns like:
        - __u0028_ to _( (preserving the first underscore)
        - _u0029_ to )
        - __u002F_ to _/
        - __u003A_ to _:
        - __u002C_ to _,
        - u003A (without underscores) to :
        
        Note: When double underscores appear before a Unicode escape,
        the first underscore is preserved as it's part of the article name.
        
        Args:
            name: Entity name with potential Unicode escapes
            
        Returns:
            Decoded entity name
        """
        # Pattern 1: __uXXXX_ format (with double underscores and trailing underscore)
        # Preserve the first underscore, decode the escape sequence
        def replace_double_underscore(match):
            hex_code = match.group(1)
            try:
                # Keep one underscore before the decoded character
                return '_' + chr(int(hex_code, 16))
            except ValueError:
                return match.group(0)
        
        name = re.sub(r'__u([0-9a-fA-F]{4})_', replace_double_underscore, name)
        
        # Pattern 2: _uXXXX_ format (single underscore)
        def replace_single_underscore(match):
            hex_code = match.group(1)
            try:
                return chr(int(hex_code, 16))
            except ValueError:
                return match.group(0)
        
        name = re.sub(r'_u([0-9a-fA-F]{4})_', replace_single_underscore, name)
        
        # Pattern 3: uXXXX format (no underscores) - be careful with this
        def replace_no_underscore(match):
            hex_code = match.group(1)
            try:
                char = chr(int(hex_code, 16))
                # Only replace if it's a common special character to avoid false positives
                if char in '()[]{}:;,/\\|<>+=-':
                    return char
                return match.group(0)
            except ValueError:
                return match.group(0)
        
        name = re.sub(r'u([0-9a-fA-F]{4})(?![0-9a-fA-F])', replace_no_underscore, name)
        
        return name
    
    def extract_entity_name(self, entity_uri: str) -> str:
        """
        Extract a readable entity name from YAGO URI and decode it
        
        Args:
            entity_uri: URI like yago:Augusto_Pinochet or yago:A-1__u0028_wrestler_u0029_
            
        Returns:
            Decoded entity name like "Augusto_Pinochet" or "A-1_(wrestler)"
        """
        if ':' in entity_uri:
            name = entity_uri.split(':', 1)[1]
        else:
            name = entity_uri
        
        # Decode any Unicode escape sequences
        return self.decode_yago_entity_name(name)
    
    def parse_wikipedia_link(self, line: str, entity: str) -> Optional[str]:
        """
        Extract Wikipedia URL if present in the line
        
        Args:
            line: Line of TTL file
            entity: Entity identifier
            
        Returns:
            Wikipedia URL or None
        """
        if 'schema:mainEntityOfPage' in line or 'wikipedia.org/wiki/' in line:
            match = re.search(r'"(https?://[^/]*wikipedia\.org/wiki/[^"]+)"', line)
            if match:
                return match.group(1)
        return None
    
    def parse_file(self, verbose: bool = False) -> None:
        """
        Parse the TTL file line by line to extract time information
        
        Args:
            verbose: If True, print progress information
        """
        if verbose:
            print(f"Parsing {self.ttl_file_path}...")
            print(f"Looking for time predicates: {', '.join(self.TIME_PREDICATES)}")
        
        line_count = 0
        entities_found = 0
        dates_found = 0
        
        current_entity = None
        
        with open(self.ttl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                
                # Progress indicator
                if verbose and line_count % 100000 == 0:
                    print(f"  Processed {line_count:,} lines, found {dates_found} dates for {entities_found} entities...")
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Skip prefix declarations
                if line.startswith('@prefix'):
                    continue
                
                # Look for lines with time predicates
                for predicate in self.TIME_PREDICATES:
                    if predicate in line:
                        # Parse the triple: subject predicate object
                        parts = line.split(None, 2)
                        if len(parts) >= 3:
                            subject = parts[0]
                            
                            # Extract date from the line
                            date_obj = self.parse_date(line)
                            if date_obj:
                                entity_name = self.extract_entity_name(subject)
                                
                                # Track new entity
                                if entity_name not in self.entity_dates:
                                    entities_found += 1
                                    self.entity_info[entity_name] = {
                                        'uri': subject,
                                        'wikipedia_url': None
                                    }
                                
                                self.entity_dates[entity_name].append(date_obj)
                                dates_found += 1
                        break
                
                # Look for Wikipedia URLs
                if 'wikipedia.org/wiki/' in line:
                    parts = line.split(None, 1)
                    if parts:
                        subject = parts[0]
                        entity_name = self.extract_entity_name(subject)
                        wiki_url = self.parse_wikipedia_link(line, entity_name)
                        
                        if wiki_url and entity_name in self.entity_info:
                            self.entity_info[entity_name]['wikipedia_url'] = wiki_url
        
        if verbose:
            print(f"\nParsing complete!")
            print(f"  Total lines processed: {line_count:,}")
            print(f"  Entities with time data: {entities_found}")
            print(f"  Total dates extracted: {dates_found}")
    
    def get_results(self) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
        """
        Get parsing results as a list of tuples
        
        Returns:
            List of (entity_name, wikipedia_url, earliest_date, latest_date)
        """
        results = []
        
        for entity_name in sorted(self.entity_dates.keys()):
            dates = self.entity_dates[entity_name]
            info = self.entity_info.get(entity_name, {})
            
            earliest = min(dates).strftime('%Y-%m-%d') if dates else None
            latest = max(dates).strftime('%Y-%m-%d') if dates else None
            wikipedia_url = info.get('wikipedia_url', '')
            
            results.append((entity_name, wikipedia_url, earliest, latest))
        
        return results
    
    def export_csv(self, output_file: str, include_no_dates: bool = False) -> None:
        """
        Export results to a CSV file
        
        Args:
            output_file: Path to output CSV file
            include_no_dates: If True, include entities with no date information
        """
        import csv
        
        results = self.get_results()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Entity', 'Wikipedia_URL', 'Earliest_Date', 'Latest_Date'])
            
            for entity, wiki_url, earliest, latest in results:
                if earliest or include_no_dates:
                    writer.writerow([entity, wiki_url or '', earliest or '0', latest or '0'])
        
        print(f"Exported {len(results)} entities to {output_file}")
    
    def export_json(self, output_file: str) -> None:
        """
        Export results to a JSON file
        
        Args:
            output_file: Path to output JSON file
        """
        import json
        
        results = self.get_results()
        
        data = []
        for entity, wiki_url, earliest, latest in results:
            data.append({
                'entity': entity,
                'wikipedia_url': wiki_url or '',
                'earliest_date': earliest or '0',
                'latest_date': latest or '0'
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(results)} entities to {output_file}")
    
    def print_summary(self, limit: int = 20) -> None:
        """
        Print a summary of extracted data
        
        Args:
            limit: Maximum number of entities to display
        """
        results = self.get_results()
        
        print(f"\n{'='*80}")
        print(f"YAGO Time Extraction Summary")
        print(f"{'='*80}")
        print(f"Total entities with time data: {len(results)}")
        print(f"\nShowing first {min(limit, len(results))} entities:\n")
        
        print(f"{'Entity':<40} {'Earliest Date':<15} {'Latest Date':<15}")
        print(f"{'-'*70}")
        
        for entity, wiki_url, earliest, latest in results[:limit]:
            entity_display = entity[:37] + '...' if len(entity) > 40 else entity
            print(f"{entity_display:<40} {earliest or 'N/A':<15} {latest or 'N/A':<15}")
        
        if len(results) > limit:
            print(f"\n... and {len(results) - limit} more entities")


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract time-related metadata from YAGO TTL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse and display summary
  python yago_parser.py yago-tiny.ttl
  
  # Export to CSV
  python yago_parser.py yago-tiny.ttl --csv output.csv
  
  # Export to JSON
  python yago_parser.py yago-tiny.ttl --json output.json
  
  # Verbose mode with CSV export
  python yago_parser.py yago-tiny.ttl --csv output.csv --verbose
        """
    )
    
    parser.add_argument('ttl_file', help='Path to the YAGO TTL file')
    parser.add_argument('--csv', help='Export results to CSV file')
    parser.add_argument('--json', help='Export results to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Print verbose progress information')
    parser.add_argument('--limit', type=int, default=20,
                       help='Number of entities to display in summary (default: 20)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Do not print summary to console')
    
    args = parser.parse_args()
    
    try:
        # Create parser and process file
        extractor = YagoTimeExtractor(args.ttl_file)
        extractor.parse_file(verbose=args.verbose)
        
        # Export results
        if args.csv:
            extractor.export_csv(args.csv)
        
        if args.json:
            extractor.export_json(args.json)
        
        # Print summary unless disabled
        if not args.no_summary:
            extractor.print_summary(limit=args.limit)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
