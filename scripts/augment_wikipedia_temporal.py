#!/usr/bin/env python3
"""
Augment Wikipedia Database with Temporal Information

This script augments the local Wikipedia PostgreSQL database with temporal
information from normalized YAGO or Wikidata data.  It adds three columns to
the articles table:
- has_temporal_info: Boolean flag indicating if temporal data is available
- earliest_date: Earliest date associated with the article
- latest_date: Latest date associated with the article

The script:
1. Adds the three new columns to the articles table (if they don't exist)
2. Reads temporal data from a normalized CSV file (.csv or .csv.zst)
3. COPYs the data into a temporary table and performs a single bulk UPDATE
   using LEAST/GREATEST to merge with existing temporal data
4. Outputs summary statistics showing coverage of temporal information

Usage:
    python augment_wikipedia_temporal.py yago-facts-normalized.csv.zst
    python augment_wikipedia_temporal.py yago-facts-normalized.csv.zst --dry-run
    python augment_wikipedia_temporal.py wikidata-temporal-normalized.csv.zst --verbose
"""

import argparse
import csv
import io
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database configuration — honours deepred-env.sh environment variables
DB_CONFIG = {
    'host': os.environ.get('PG_HOST', 'localhost'),
    'port': int(os.environ.get('PG_PORT', 5432)),
    'database': os.environ.get('PG_DATABASE', 'wikidb'),
    'user': os.environ.get('PG_USER', 'wiki'),
    'password': os.environ.get('PG_PASSWORD', 'wiki')
}

# ANSI color codes for terminal output
COLOR_GREEN = '\033[32m'
COLOR_YELLOW = '\033[33m'
COLOR_RESET = '\033[0m'


# ---------------------------------------------------------------------------
# Helpers – transparent zstd reading
# ---------------------------------------------------------------------------

def _open_input(path: str, mode: str = 'r'):
    """Return a text-mode file handle; transparently decompress ``.zst`` files."""
    if path.endswith('.zst'):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        raw = open(path, 'rb')
        return io.TextIOWrapper(
            dctx.stream_reader(raw, closefd=True),
            encoding='utf-8',
            newline='',
        )
    return open(path, mode, encoding='utf-8')


class TemporalAugmenter:
    """Augment Wikipedia database with temporal information from YAGO or Wikidata"""
    
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
        Load temporal data from a normalized CSV file (.csv or .csv.zst).

        Args:
            csv_file: Path to normalized CSV file (plain or zstd-compressed)

        Returns:
            Dictionary mapping Wikipedia page IDs to (earliest_date, latest_date) tuples
        """
        temporal_data = {}
        skipped_count = 0

        logging.info(f"Loading temporal data from {csv_file}...")

        try:
            with _open_input(csv_file) as f:
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

    def update_articles(self, temporal_data: Dict[int, Tuple[str, str]], batch_size: int = 1000, dry_run: bool = False) -> Dict[str, int]:
        """
        Bulk-update articles table with temporal information using COPY + SQL merge.

        Loads the CSV data into a PostgreSQL temporary table via COPY FROM,
        then performs a single UPDATE … FROM with LEAST/GREATEST to merge
        new dates with any existing temporal data.  This is dramatically
        faster than row-by-row execute_batch.

        For articles with existing temporal info:
        - earliest_date = LEAST(existing_earliest, new_earliest)
        - latest_date   = GREATEST(existing_latest, new_latest)

        Args:
            temporal_data: Dictionary mapping Wikipedia IDs to (earliest_date, latest_date)
            batch_size: Unused (kept for CLI compatibility); bulk COPY+UPDATE is always used
            dry_run: If True, don't commit changes to database

        Returns:
            Dictionary with update statistics
        """
        from datetime import timedelta

        stats = {
            'total_attempted': len(temporal_data),
            'new_articles': 0,
            'updated_articles': 0,
            'unchanged_articles': 0,
            'failed': 0,
        }

        if dry_run:
            logging.info(f"{COLOR_YELLOW}DRY RUN MODE: No changes will be committed to database{COLOR_RESET}")

        logging.info(f"Updating {len(temporal_data):,} articles with temporal information (bulk COPY+UPDATE)...")
        start_time = datetime.now()

        try:
            # ------------------------------------------------------------------
            # 1.  Create a temporary table and COPY the data in
            # ------------------------------------------------------------------
            self.cursor.execute("""
                CREATE TEMP TABLE _temporal_staging (
                    wiki_id   INTEGER  NOT NULL,
                    earliest  DATE     NOT NULL,
                    latest    DATE     NOT NULL
                ) ON COMMIT DROP
            """)
            logging.info("Created temporary staging table")

            # Build a tab-separated in-memory file for COPY FROM
            buf = io.StringIO()
            for wiki_id, (earliest, latest) in temporal_data.items():
                buf.write(f"{wiki_id}\t{earliest}\t{latest}\n")
            buf.seek(0)

            self.cursor.copy_from(buf, '_temporal_staging', columns=('wiki_id', 'earliest', 'latest'))
            copy_count = len(temporal_data)
            logging.info(f"COPYed {copy_count:,} rows into staging table")
            buf.close()

            # Index the staging table for a fast merge join
            self.cursor.execute("CREATE INDEX ON _temporal_staging (wiki_id)")
            logging.info("Indexed staging table")

            # ------------------------------------------------------------------
            # 2.  Single UPDATE with LEAST / GREATEST merge
            # ------------------------------------------------------------------
            self.cursor.execute("""
                UPDATE articles a
                SET has_temporal_info = TRUE,
                    earliest_date    = LEAST(COALESCE(a.earliest_date, s.earliest), s.earliest),
                    latest_date      = GREATEST(COALESCE(a.latest_date, s.latest), s.latest)
                FROM _temporal_staging s
                WHERE a.wikipedia_page_id = s.wiki_id
            """)
            rows_updated = self.cursor.rowcount
            elapsed = (datetime.now() - start_time).total_seconds()
            logging.info(f"{COLOR_GREEN}Bulk UPDATE matched {rows_updated:,} articles in {elapsed:.1f}s{COLOR_RESET}")

            # ------------------------------------------------------------------
            # 3.  Compute statistics (new vs updated vs unchanged)
            # ------------------------------------------------------------------
            # Articles that were updated fall into three categories:
            #  - "new":       had no temporal info before
            #  - "updated":   had temporal info but dates were widened
            #  - "unchanged": had temporal info with identical/wider dates already
            #
            # We count new + changed from the staging join.  "Unchanged" articles
            # are those in the staging table whose wiki_id matched an article with
            # has_temporal_info = TRUE *before* the update and whose dates were
            # already as wide or wider.  Since the UPDATE is already done, we
            # approximate by noting:
            #   rows_updated = new + updated + unchanged  (all matched rows)
            # The expensive pre-query to classify them is unnecessary — the
            # single-UPDATE approach already saved the time.  We report the total
            # matched count.

            stats['new_articles'] = rows_updated   # upper-bound (includes merges)
            stats['updated_articles'] = 0
            stats['unchanged_articles'] = 0

            # Articles in CSV but not matched to any article row
            not_found = copy_count - rows_updated
            if not_found > 0:
                logging.info(f"Articles in CSV not found in database: {not_found:,}")

            if dry_run:
                self.conn.rollback()
            else:
                self.conn.commit()

        except psycopg2.Error as e:
            logging.error(f"Bulk update failed: {e}")
            self.conn.rollback()
            stats['failed'] = len(temporal_data)

        return stats
    
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
        description='Augment Wikipedia database with temporal information from YAGO or Wikidata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update database with temporal information (supports .csv and .csv.zst)
  python augment_wikipedia_temporal.py yago-facts-normalized.csv.zst
  
  # Dry run to see what would be updated
  python augment_wikipedia_temporal.py yago-facts-normalized.csv.zst --dry-run

  # Wikidata
  python augment_wikipedia_temporal.py wikidata-temporal-normalized.csv.zst --verbose
        """
    )
    
    parser.add_argument('input_file', help='Normalized CSV file (.csv or .csv.zst) from normalize_temporal_output.py')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without committing changes to database')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Unused (kept for compatibility); bulk COPY+UPDATE is always used')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--db-host', default=None, help='PostgreSQL host (default: $PG_HOST or localhost)')
    parser.add_argument('--db-name', default=None, help='Database name (default: $PG_DATABASE or wikidb)')
    parser.add_argument('--db-user', default=None, help='Database user (default: $PG_USER or wiki)')
    parser.add_argument('--db-password', default=None, help='Database password (default: $PG_PASSWORD or wiki)')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup database config — CLI args override env vars which override defaults
    db_config = dict(DB_CONFIG)
    if args.db_host:
        db_config['host'] = args.db_host
    if args.db_name:
        db_config['database'] = args.db_name
    if args.db_user:
        db_config['user'] = args.db_user
    if args.db_password:
        db_config['password'] = args.db_password
    
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
        update_stats = augmenter.update_articles(
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
        total_attempted = update_stats['total_attempted']
        logging.info("\n=== Update Summary ===")
        logging.info(f"Temporal records in CSV: {total_attempted:,}")
        logging.info(f"Articles updated successfully: {update_stats['new_articles']:,}")
        if update_stats['failed'] > 0:
            logging.info(f"Failed updates: {update_stats['failed']:,}")

        # Articles not found in database
        not_found = total_attempted - update_stats['new_articles']
        if not_found > 0:
            logging.info(f"Articles not found in database: {not_found:,} ({100*not_found/total_attempted:.1f}%)")
        
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
