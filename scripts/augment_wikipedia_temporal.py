#!/usr/bin/env python3
"""
Augment Wikipedia Database with Temporal Information

This script augments the local Wikipedia PostgreSQL database with temporal 
information from normalized YAGO data. It adds three columns to the articles table:
- has_temporal_info: Boolean flag indicating if temporal data is available
- earliest_date: Earliest date associated with the article from YAGO
- latest_date: Latest date associated with the article from YAGO

The script:
1. Adds the three new columns to the articles table (if they don't exist)
2. Reads temporal data from normalized YAGO CSV file
3. Updates articles using Wikipedia page IDs from the YAGO data
4. Outputs summary statistics showing coverage of temporal information

Usage:
    python augment_wikipedia_temporal.py yago-facts-normalized.csv
    python augment_wikipedia_temporal.py yago-facts-normalized.csv --dry-run
    python augment_wikipedia_temporal.py yago-facts-normalized.csv --batch-size 500
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from typing import Dict, Tuple

import psycopg2
from psycopg2.extras import execute_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'wikidb',
    'user': 'wiki',
    'password': 'wikipass'
}

# ANSI color codes for terminal output
COLOR_GREEN = '\033[32m'
COLOR_YELLOW = '\033[33m'
COLOR_RESET = '\033[0m'


class TemporalAugmenter:
    """Augment Wikipedia database with temporal information from YAGO"""
    
    def __init__(self, db_config: Dict = None):
        """
        Initialize the augmenter with database connection
        
        Args:
            db_config: Database configuration dict
        """
        self.db_config = db_config or DB_CONFIG
        self.conn = None
        self.cursor = None
        self.current_year = datetime.now().year
        
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
    
    def _is_valid_date_string(self, date_str: str) -> bool:
        """
        Check if a string is a valid date in YYYY-MM-DD format
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid date format, False otherwise
        """
        if not date_str:
            return False
        
        # Check for basic date format patterns (handles negative years for BCE)
        # Valid formats: YYYY-MM-DD, -YYYY-MM-DD, -YYYYYY-MM-DD
        try:
            # Try to parse as date
            parts = date_str.lstrip('-').split('-')
            if len(parts) != 3:
                return False
            
            # Check that all parts are numeric
            year, month, day = parts
            year_int = int(year)
            month_int = int(month)
            day_int = int(day)
            
            # Year must be at least 4 digits (e.g., 0050, not 50)
            # This prevents ambiguous dates like "50-01-01" which PostgreSQL can't handle
            if len(year) < 4:
                return False
            
            # Basic validation for month and day
            if not (1 <= month_int <= 12):
                return False
            if not (1 <= day_int <= 31):
                return False
            
            # Validate year range (0 to current year)
            # This prevents unrealistic future dates and negative years (BCE)
            if year_int < 0 or year_int > self.current_year:
                return False
            
            return True
        except (ValueError, AttributeError):
            return False
    
    def add_temporal_columns(self) -> bool:
        """
        Add temporal columns to articles table if they don't exist
        
        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime
        try:
            logging.info("Adding temporal columns to articles table...")
            logging.info("(This may take a few moments for large tables)")
            
            start_time = datetime.now()
            
            # Add wikipedia_page_id column for fast lookups
            logging.info("Checking wikipedia_page_id column...")
            
            # Check if column exists and has data
            self.cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = 'articles' AND column_name = 'wikipedia_page_id'
            """)
            column_exists = self.cursor.fetchone()[0] > 0
            
            if not column_exists:
                logging.info("Adding wikipedia_page_id column...")
                self.cursor.execute("""
                    ALTER TABLE articles 
                    ADD COLUMN wikipedia_page_id INTEGER
                """)
            else:
                logging.info("wikipedia_page_id column already exists")
            
            # Check if page IDs need to be extracted
            self.cursor.execute("""
                SELECT COUNT(*) FROM articles WHERE wikipedia_page_id IS NULL AND url ~ 'curid='
            """)
            null_count = self.cursor.fetchone()[0]
            
            if null_count > 0:
                # Extract wikipedia_page_id from URL (format: https://en.wikipedia.org/wiki?curid=12345)
                logging.info(f"Extracting Wikipedia page IDs from URLs for {null_count:,} articles...")
                self.cursor.execute("""
                    UPDATE articles 
                    SET wikipedia_page_id = (regexp_match(url, 'curid=(\\d+)'))[1]::INTEGER
                    WHERE url ~ 'curid=' AND wikipedia_page_id IS NULL
                """)
                rows_updated = self.cursor.rowcount
                logging.info(f"Extracted Wikipedia page IDs for {rows_updated:,} articles")
            else:
                logging.info("Wikipedia page IDs already extracted")
            
            # Create index on wikipedia_page_id for fast lookups (if not exists)
            logging.info("Ensuring index on wikipedia_page_id exists...")
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_wikipedia_page_id 
                ON articles(wikipedia_page_id)
            """)
            
            # Add has_temporal_info column
            logging.info("Adding has_temporal_info column...")
            self.cursor.execute("""
                ALTER TABLE articles 
                ADD COLUMN IF NOT EXISTS has_temporal_info BOOLEAN DEFAULT FALSE
            """)
            
            # Add earliest_date column
            logging.info("Adding earliest_date column...")
            self.cursor.execute("""
                ALTER TABLE articles 
                ADD COLUMN IF NOT EXISTS earliest_date DATE
            """)
            
            # Add latest_date column
            logging.info("Adding latest_date column...")
            self.cursor.execute("""
                ALTER TABLE articles 
                ADD COLUMN IF NOT EXISTS latest_date DATE
            """)
            
            self.conn.commit()
            elapsed = (datetime.now() - start_time).total_seconds()
            logging.info(f"Temporal columns added successfully ({elapsed:.1f} seconds)")
            return True
            
        except psycopg2.Error as e:
            logging.error(f"Failed to add temporal columns: {e}")
            self.conn.rollback()
            return False
    
    def clean_invalid_dates(self) -> int:
        """
        Clean invalid dates from the database (dates outside 0-current_year range)
        
        Returns:
            Number of articles cleaned
        """
        try:
            logging.info(f"Cleaning invalid dates from database (valid range: 0-{self.current_year})...")
            
            # First, count how many articles will be affected
            self.cursor.execute(f"""
                SELECT COUNT(*) FROM articles 
                WHERE has_temporal_info = TRUE 
                AND (
                    EXTRACT(YEAR FROM earliest_date) < 0 
                    OR EXTRACT(YEAR FROM earliest_date) > {self.current_year}
                    OR EXTRACT(YEAR FROM latest_date) < 0 
                    OR EXTRACT(YEAR FROM latest_date) > {self.current_year}
                )
            """)
            invalid_count = self.cursor.fetchone()[0]
            
            if invalid_count == 0:
                logging.info("No invalid dates found in database")
                return 0
            
            logging.info(f"Found {invalid_count:,} articles with invalid dates (outside 0-{self.current_year} range)")
            
            # Clean invalid dates
            self.cursor.execute(f"""
                UPDATE articles 
                SET has_temporal_info = FALSE,
                    earliest_date = NULL,
                    latest_date = NULL
                WHERE has_temporal_info = TRUE 
                AND (
                    EXTRACT(YEAR FROM earliest_date) < 0 
                    OR EXTRACT(YEAR FROM earliest_date) > {self.current_year}
                    OR EXTRACT(YEAR FROM latest_date) < 0 
                    OR EXTRACT(YEAR FROM latest_date) > {self.current_year}
                )
            """)
            
            cleaned_count = self.cursor.rowcount
            self.conn.commit()
            
            logging.info(f"{COLOR_GREEN}Cleaned {cleaned_count:,} articles with invalid dates{COLOR_RESET}")
            return cleaned_count
            
        except psycopg2.Error as e:
            logging.error(f"Failed to clean invalid dates: {e}")
            self.conn.rollback()
            return 0
    
    def load_temporal_data(self, csv_file: str) -> Dict[int, Tuple[str, str]]:
        """
        Load temporal data from normalized YAGO CSV file
        
        Args:
            csv_file: Path to normalized YAGO CSV file
            
        Returns:
            Dictionary mapping Wikipedia page IDs to (earliest_date, latest_date) tuples
        """
        temporal_data = {}
        skipped_count = 0
        
        logging.info(f"Loading temporal data from {csv_file}...")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Skip entries without Wikipedia IDs
                        if not row.get('Wikipedia_ID') or row['Wikipedia_ID'] == '':
                            skipped_count += 1
                            continue
                        
                        # Skip header row if it appears (Wikipedia_ID would be the string 'Wikipedia_ID')
                        if row['Wikipedia_ID'] == 'Wikipedia_ID':
                            continue
                        
                        wiki_id = int(row['Wikipedia_ID'])
                        earliest = row.get('Earliest_Date', '').strip()
                        latest = row.get('Latest_Date', '').strip()
                        
                        # Skip entries without valid dates
                        if not earliest or not latest:
                            skipped_count += 1
                            continue
                        
                        # Validate date format (YYYY-MM-DD or negative years)
                        # This prevents header text like 'Earliest_Date' from being treated as a date
                        if not self._is_valid_date_string(earliest) or not self._is_valid_date_string(latest):
                            skipped_count += 1
                            continue
                        
                        temporal_data[wiki_id] = (earliest, latest)
                        
                    except (ValueError, KeyError) as e:
                        logging.debug(f"Skipping malformed row: {row} - {e}")
                        skipped_count += 1
                        continue
            
            logging.info(f"Loaded temporal data for {len(temporal_data):,} articles")
            if skipped_count > 0:
                logging.info(f"Skipped {skipped_count:,} invalid/incomplete entries")
            return temporal_data
            
        except FileNotFoundError:
            logging.error(f"File not found: {csv_file}")
            return {}
        except Exception as e:
            logging.error(f"Error loading temporal data: {e}")
            return {}
    
    def update_articles(self, temporal_data: Dict[int, Tuple[str, str]], batch_size: int = 1000, dry_run: bool = False) -> Tuple[int, int, int]:
        """
        Update articles table with temporal information
        
        Args:
            temporal_data: Dictionary mapping Wikipedia IDs to (earliest_date, latest_date)
            batch_size: Number of records to update in each batch
            dry_run: If True, don't commit changes to database
            
        Returns:
            Tuple of (total_attempted, successful_updates, failed_updates)
        """
        from datetime import datetime, timedelta
        
        total_attempted = len(temporal_data)
        successful = 0
        failed = 0
        
        if dry_run:
            logging.info(f"{COLOR_YELLOW}DRY RUN MODE: No changes will be committed to database{COLOR_RESET}")
        
        logging.info(f"Updating articles with temporal information (batch size: {batch_size:,})...")
        
        # Prepare batch update data
        update_data = []
        for wiki_id, (earliest, latest) in temporal_data.items():
            update_data.append((True, earliest, latest, wiki_id))
        
        # Process in batches
        total_batches = (len(update_data) + batch_size - 1) // batch_size
        start_time = datetime.now()
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(update_data))
            batch = update_data[start_idx:end_idx]
            
            try:
                # Update articles by Wikipedia page ID (fast indexed lookup)
                execute_batch(self.cursor, """
                    UPDATE articles 
                    SET has_temporal_info = %s,
                        earliest_date = %s,
                        latest_date = %s
                    WHERE wikipedia_page_id = %s
                """, batch)
                
                # execute_batch doesn't update rowcount properly, so count the batch size
                batch_success = len(batch)
                successful += batch_success
                
                if not dry_run:
                    self.conn.commit()
                else:
                    self.conn.rollback()
                
                # Progress indicator with ETA (report every batch)
                elapsed = (datetime.now() - start_time).total_seconds()
                batches_processed = batch_num + 1
                progress_pct = (batches_processed / total_batches) * 100
                
                if batches_processed > 0 and elapsed > 0:
                    rate = batches_processed / elapsed
                    remaining_batches = total_batches - batches_processed
                    eta_seconds = remaining_batches / rate if rate > 0 else 0
                    eta = timedelta(seconds=int(eta_seconds))
                    
                    logging.info(f"{COLOR_GREEN}Processed batch {batches_processed:,}/{total_batches:,} ({progress_pct:.1f}%), "
                               f"updated {successful:,} articles | "
                               f"Rate: {rate:.1f} batches/sec | ETA: {eta}{COLOR_RESET}")
                else:
                    logging.info(f"{COLOR_GREEN}Processed batch {batches_processed:,}/{total_batches:,} ({progress_pct:.1f}%), "
                               f"updated {successful:,} articles{COLOR_RESET}")
                
            except psycopg2.Error as e:
                logging.error(f"Batch update failed: {e}")
                failed += len(batch)
                self.conn.rollback()
                continue
        
        return (total_attempted, successful, failed)
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about temporal coverage
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {}
            
            # Total articles
            self.cursor.execute("SELECT COUNT(*) FROM articles")
            stats['total_articles'] = self.cursor.fetchone()[0]
            
            # Articles with temporal info
            self.cursor.execute("SELECT COUNT(*) FROM articles WHERE has_temporal_info = TRUE")
            stats['articles_with_temporal'] = self.cursor.fetchone()[0]
            
            # Articles without temporal info
            stats['articles_without_temporal'] = stats['total_articles'] - stats['articles_with_temporal']
            
            # Percentage coverage
            if stats['total_articles'] > 0:
                stats['coverage_percentage'] = (stats['articles_with_temporal'] / stats['total_articles']) * 100
            else:
                stats['coverage_percentage'] = 0.0
            
            # Date range statistics
            self.cursor.execute("""
                SELECT 
                    MIN(earliest_date) as min_date,
                    MAX(latest_date) as max_date
                FROM articles 
                WHERE has_temporal_info = TRUE
            """)
            result = self.cursor.fetchone()
            stats['earliest_date'] = result[0]
            stats['latest_date'] = result[1]
            
            # Articles by century
            self.cursor.execute("""
                SELECT 
                    FLOOR(EXTRACT(YEAR FROM earliest_date) / 100) * 100 as century,
                    COUNT(*) as count
                FROM articles 
                WHERE has_temporal_info = TRUE
                GROUP BY century
                ORDER BY century
            """)
            stats['top_centuries'] = self.cursor.fetchall()
            
            return stats
            
        except psycopg2.Error as e:
            logging.error(f"Failed to get statistics: {e}")
            return {}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Augment Wikipedia database with temporal information from YAGO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update database with temporal information
  python augment_wikipedia_temporal.py yago-facts-normalized.csv
  
  # Dry run to see what would be updated
  python augment_wikipedia_temporal.py yago-facts-normalized.csv --dry-run
  
  # Use larger batch size for faster processing
  python augment_wikipedia_temporal.py yago-facts-normalized.csv --batch-size 5000
  
  # Verbose logging
  python augment_wikipedia_temporal.py yago-facts-normalized.csv --verbose
        """
    )
    
    parser.add_argument('input_file', help='Normalized YAGO CSV file (output from normalize_yago_output.py)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without committing changes to database')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of records to update in each batch (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--db-host', default='localhost', help='PostgreSQL host (default: localhost)')
    parser.add_argument('--db-name', default='wikidb', help='Database name (default: wikidb)')
    parser.add_argument('--db-user', default='wiki', help='Database user (default: wiki)')
    parser.add_argument('--db-password', default='wikipass', help='Database password (default: wikipass)')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup database config
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # Create augmenter
    augmenter = TemporalAugmenter(db_config)
    
    # Connect to database
    if not augmenter.connect_db():
        logging.error("Failed to connect to database")
        sys.exit(1)
    
    try:
        # Add temporal columns first (before getting statistics)
        if not augmenter.add_temporal_columns():
            logging.error("Failed to add temporal columns")
            sys.exit(1)
        
        # Clean any existing invalid dates from the database
        augmenter.clean_invalid_dates()
        
        # Get statistics before update
        logging.info("\n=== Database Statistics (Before Update) ===")
        stats_before = augmenter.get_statistics()
        if stats_before:
            logging.info(f"Total articles: {stats_before['total_articles']:,}")
            logging.info(f"Articles with temporal info: {stats_before['articles_with_temporal']:,} "
                       f"({stats_before['coverage_percentage']:.2f}%)")
        
        # Load temporal data
        temporal_data = augmenter.load_temporal_data(args.input_file)
        
        if not temporal_data:
            logging.error("No temporal data loaded")
            sys.exit(1)
        
        # Update articles
        total_attempted, successful, failed = augmenter.update_articles(
            temporal_data, 
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        # Get statistics after update
        logging.info("\n=== Database Statistics (After Update) ===")
        stats_after = augmenter.get_statistics()
        
        if stats_after:
            logging.info(f"Total articles: {stats_after['total_articles']:,}")
            logging.info(f"Articles with temporal info: {stats_after['articles_with_temporal']:,} "
                       f"({stats_after['coverage_percentage']:.2f}%)")
            logging.info(f"Articles without temporal info: {stats_after['articles_without_temporal']:,}")
            
            if stats_after.get('earliest_date') and stats_after.get('latest_date'):
                logging.info(f"Temporal date range: {stats_after['earliest_date']} to {stats_after['latest_date']}")
            
            if stats_after.get('top_centuries'):
                logging.info("\nTop centuries by article count:")
                for century, count in stats_after['top_centuries']:
                    century_label = f"{int(century)}s" if century >= 0 else f"{int(abs(century))} BCE"
                    logging.info(f"  {century_label}: {count:,} articles")
        
        # Summary
        logging.info("\n=== Update Summary ===")
        logging.info(f"Temporal records in CSV: {total_attempted:,}")
        logging.info(f"Articles updated successfully: {successful:,} ({100*successful/total_attempted:.1f}%)")
        logging.info(f"Articles not found in database: {total_attempted - successful:,} ({100*(total_attempted-successful)/total_attempted:.1f}%)")
        
        if args.dry_run:
            logging.info(f"\n{COLOR_YELLOW}DRY RUN COMPLETE: No changes were committed to the database{COLOR_RESET}")
        else:
            logging.info(f"\n{COLOR_GREEN}Database augmentation complete!{COLOR_RESET}")
        
    except Exception as e:
        logging.error(f"Error during augmentation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        augmenter.close_db()


if __name__ == '__main__':
    main()
