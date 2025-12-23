#!/usr/bin/env python3
"""
Generate Temporal Datasets for Fine-Tuning

This script generates training datasets for temporal knowledge cutoff fine-tuning.
It extracts articles from the Wikipedia database (with temporal augmentation) and
generates Q&A pairs using a local LLM (LM Studio).

Datasets generated:
- retain_train.jsonl: Pre-cutoff Q&A pairs with factual answers
- retain_val.jsonl: Pre-cutoff validation set
- unlearn_train.jsonl: Post-cutoff Q&A pairs with refusal responses
- unlearn_val.jsonl: Post-cutoff validation set
- dev_subset.jsonl: Small combined subset for development

Usage:
    python generate_temporal_datasets.py --mode dev
    python generate_temporal_datasets.py --mode full --output-dir /path/to/output
    python generate_temporal_datasets.py --dry-run

Requirements:
    - PostgreSQL database with temporal augmentation (see TemporalAugmentation-Setup.md)
    - LM Studio server running with a capable model
    - Python packages: psycopg2-binary, requests, tqdm
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default environment-based configuration
WIKI_DATA = os.environ.get('WIKI_DATA', '/mnt/data/wikipedia')

# Database defaults
DEFAULT_DB_HOST = 'localhost'
DEFAULT_DB_NAME = 'wikidb'
DEFAULT_DB_USER = 'wiki'
DEFAULT_DB_PASSWORD = 'wikipass'

# LM Studio defaults
DEFAULT_LMSTUDIO_HOST = os.environ.get('LMSTUDIO_HOST', 'localhost')
DEFAULT_LMSTUDIO_PORT = int(os.environ.get('LMSTUDIO_PORT', '1234'))
DEFAULT_LMSTUDIO_MODEL = 'qwen2.5-7b-instruct'  # Adjust based on available model

# MCP Server defaults (for semantic search enrichment)
DEFAULT_MCP_HOST = os.environ.get('MCP_HOST', 'localhost')
DEFAULT_MCP_PORT = int(os.environ.get('MCP_PORT', '7000'))

# Default cutoff date (Apollo 11 moon landing)
DEFAULT_CUTOFF_DATE = '1969-07-20'

# Generation settings
DEFAULT_QUESTIONS_PER_ARTICLE = 3
DEFAULT_BATCH_SIZE = 100
DEFAULT_SEED = 42

# Content length limits
MIN_CONTENT_LENGTH = 500
MAX_CONTENT_FOR_LLM = 2000

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Refusal responses for unlearn dataset
REFUSAL_RESPONSES = [
    "I don't have information about that.",
    "I'm not familiar with that topic.",
    "That's beyond my knowledge.",
    "I don't have details about that subject.",
    "I'm unable to provide information on that.",
    "I don't have knowledge of that matter.",
    "That topic is outside my area of knowledge.",
    "I'm not aware of that.",
    "I don't have enough information to answer that.",
    "That's not something I have knowledge about.",
]

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class Article:
    """Represents a Wikipedia article with temporal metadata."""
    id: int
    title: str
    content: str
    earliest_date: Optional[str]
    latest_date: Optional[str]
    temporal_class: str  # 'pre_cutoff' or 'post_cutoff'


@dataclass
class QAPair:
    """Represents a question-answer pair for training."""
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'instruction': self.instruction,
            'input': self.input,
            'output': self.output,
            'metadata': self.metadata
        }


@dataclass
class GenerationStats:
    """Statistics for the generation process."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    articles_processed: int = 0
    pre_cutoff_articles: int = 0
    post_cutoff_articles: int = 0
    retain_qa_pairs: int = 0
    unlearn_qa_pairs: int = 0
    failed_generations: int = 0
    skipped_articles: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'generation_date': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_hours': (
                (self.end_time - self.start_time).total_seconds() / 3600 
                if self.end_time else None
            ),
            'articles_processed': self.articles_processed,
            'pre_cutoff_articles': self.pre_cutoff_articles,
            'post_cutoff_articles': self.post_cutoff_articles,
            'retain_qa_pairs': self.retain_qa_pairs,
            'unlearn_qa_pairs': self.unlearn_qa_pairs,
            'failed_generations': self.failed_generations,
            'skipped_articles': self.skipped_articles,
        }


# -----------------------------------------------------------------------------
# Database Operations
# -----------------------------------------------------------------------------

class DatabaseManager:
    """Manage PostgreSQL database connections and queries."""
    
    def __init__(self, host: str, database: str, user: str, password: str):
        self.config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.config, cursor_factory=RealDictCursor)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to PostgreSQL database: {self.config['database']}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def get_temporal_statistics(self, cutoff_date: str) -> Dict[str, int]:
        """Get counts of articles by temporal classification."""
        stats = {}
        
        # Total articles with temporal info
        self.cursor.execute("""
            SELECT COUNT(*) as count 
            FROM articles 
            WHERE has_temporal_info = TRUE
        """)
        stats['total_temporal'] = self.cursor.fetchone()['count']
        
        # Pre-cutoff articles
        self.cursor.execute("""
            SELECT COUNT(*) as count 
            FROM articles 
            WHERE has_temporal_info = TRUE
              AND (
                latest_date <= %s
                OR (earliest_date <= %s AND latest_date IS NULL)
              )
              AND LENGTH(content) > %s
        """, (cutoff_date, cutoff_date, MIN_CONTENT_LENGTH))
        stats['pre_cutoff'] = self.cursor.fetchone()['count']
        
        # Post-cutoff articles
        self.cursor.execute("""
            SELECT COUNT(*) as count 
            FROM articles 
            WHERE has_temporal_info = TRUE
              AND earliest_date > %s
              AND LENGTH(content) > %s
        """, (cutoff_date, MIN_CONTENT_LENGTH))
        stats['post_cutoff'] = self.cursor.fetchone()['count']
        
        # Spanning articles (cross cutoff boundary)
        self.cursor.execute("""
            SELECT COUNT(*) as count 
            FROM articles 
            WHERE has_temporal_info = TRUE
              AND earliest_date <= %s 
              AND latest_date > %s
              AND LENGTH(content) > %s
        """, (cutoff_date, cutoff_date, MIN_CONTENT_LENGTH))
        stats['spanning'] = self.cursor.fetchone()['count']
        
        return stats
    
    def fetch_pre_cutoff_articles(
        self, 
        cutoff_date: str, 
        limit: int,
        offset: int = 0,
        seed: int = 42
    ) -> List[Article]:
        """Fetch articles from before the temporal cutoff."""
        # Set random seed for reproducible sampling
        self.cursor.execute(f"SELECT setseed({seed / 2147483647.0})")
        
        self.cursor.execute("""
            SELECT id, title, content, 
                   earliest_date::text, latest_date::text
            FROM articles 
            WHERE has_temporal_info = TRUE
              AND (
                latest_date <= %s
                OR (earliest_date <= %s AND latest_date IS NULL)
              )
              AND LENGTH(content) > %s
            ORDER BY RANDOM()
            LIMIT %s OFFSET %s
        """, (cutoff_date, cutoff_date, MIN_CONTENT_LENGTH, limit, offset))
        
        return [
            Article(
                id=row['id'],
                title=row['title'],
                content=row['content'],
                earliest_date=row['earliest_date'],
                latest_date=row['latest_date'],
                temporal_class='pre_cutoff'
            )
            for row in self.cursor.fetchall()
        ]
    
    def fetch_post_cutoff_articles(
        self, 
        cutoff_date: str, 
        limit: int,
        offset: int = 0,
        seed: int = 42
    ) -> List[Article]:
        """Fetch articles from after the temporal cutoff."""
        # Set random seed for reproducible sampling
        self.cursor.execute(f"SELECT setseed({seed / 2147483647.0})")
        
        self.cursor.execute("""
            SELECT id, title, content,
                   earliest_date::text, latest_date::text
            FROM articles 
            WHERE has_temporal_info = TRUE
              AND earliest_date > %s
              AND LENGTH(content) > %s
            ORDER BY RANDOM()
            LIMIT %s OFFSET %s
        """, (cutoff_date, MIN_CONTENT_LENGTH, limit, offset))
        
        return [
            Article(
                id=row['id'],
                title=row['title'],
                content=row['content'],
                earliest_date=row['earliest_date'],
                latest_date=row['latest_date'],
                temporal_class='post_cutoff'
            )
            for row in self.cursor.fetchall()
        ]


# -----------------------------------------------------------------------------
# LM Studio Integration
# -----------------------------------------------------------------------------

class LMStudioClient:
    """Client for generating Q&A pairs using LM Studio."""
    
    def __init__(self, host: str, port: int, model: str):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.timeout = 120  # seconds
    
    def check_connection(self) -> bool:
        """Verify LM Studio server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json()
            logger.info(f"LM Studio connected. Available models: {len(models.get('data', []))}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to connect to LM Studio at {self.base_url}: {e}")
            return False
    
    def generate_questions(
        self, 
        title: str, 
        content: str, 
        num_questions: int = 3
    ) -> List[Dict[str, str]]:
        """Generate Q&A pairs from article content."""
        # Truncate content to avoid exceeding context limits
        truncated_content = content[:MAX_CONTENT_FOR_LLM]
        
        prompt = f"""You are a dataset generator for training language models. Given the following Wikipedia article excerpt about "{title}", generate {num_questions} diverse factual questions that can be answered using the information provided.

Article excerpt:
{truncated_content}

Requirements:
- Questions should be specific and factual
- Questions should have clear, unambiguous answers from the text
- Vary question types: who, what, when, where, why, how
- Avoid questions that only ask about dates/years
- Questions should be self-contained (don't use pronouns like "he" or "it" without context)
- Answers should be 1-3 sentences, concise but complete

Output ONLY a valid JSON array with no other text:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON from response
            qa_pairs = self._parse_qa_response(content)
            return qa_pairs
            
        except requests.RequestException as e:
            logger.warning(f"LM Studio request failed for '{title}': {e}")
            return []
        except Exception as e:
            logger.warning(f"Failed to generate Q&A for '{title}': {e}")
            return []
    
    def _parse_qa_response(self, content: str) -> List[Dict[str, str]]:
        """Parse Q&A pairs from LLM response."""
        # Try to extract JSON array from response
        try:
            # Look for JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                qa_pairs = json.loads(json_match.group())
                # Validate structure
                valid_pairs = []
                for pair in qa_pairs:
                    if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                        q = pair['question'].strip()
                        a = pair['answer'].strip()
                        if len(q) > 10 and len(a) > 10:
                            valid_pairs.append({'question': q, 'answer': a})
                return valid_pairs
        except json.JSONDecodeError:
            pass
        
        return []


# -----------------------------------------------------------------------------
# MCP Server Integration (Optional Enrichment)
# -----------------------------------------------------------------------------

class MCPClient:
    """Client for Wikipedia MCP server (semantic search)."""
    
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
    
    def check_connection(self) -> bool:
        """Check if MCP server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def search_related(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for related article sections."""
        try:
            response = requests.post(
                f"{self.base_url}/mcp/search",
                json={
                    "query": query,
                    "mode": "semantic",
                    "limit": limit
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('results', [])
        except requests.RequestException:
            return []


# -----------------------------------------------------------------------------
# Dataset Generation
# -----------------------------------------------------------------------------

class DatasetGenerator:
    """Generate temporal fine-tuning datasets."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        lm_client: LMStudioClient,
        mcp_client: Optional[MCPClient] = None,
        cutoff_date: str = DEFAULT_CUTOFF_DATE,
        questions_per_article: int = DEFAULT_QUESTIONS_PER_ARTICLE,
        seed: int = DEFAULT_SEED
    ):
        self.db = db_manager
        self.lm = lm_client
        self.mcp = mcp_client
        self.cutoff_date = cutoff_date
        self.questions_per_article = questions_per_article
        self.seed = seed
        self.stats = GenerationStats()
        
        # Set random seed
        random.seed(seed)
    
    def generate_retain_pairs(self, article: Article) -> List[QAPair]:
        """Generate Q&A pairs for retain dataset (factual answers)."""
        qa_pairs = []
        
        # Generate questions from LLM
        raw_pairs = self.lm.generate_questions(
            article.title, 
            article.content,
            self.questions_per_article
        )
        
        for pair in raw_pairs:
            qa_pairs.append(QAPair(
                instruction=pair['question'],
                input="",
                output=pair['answer'],
                metadata={
                    'source_article_id': article.id,
                    'source_title': article.title,
                    'temporal_class': 'pre_cutoff',
                    'earliest_date': article.earliest_date,
                    'latest_date': article.latest_date,
                    'dataset_type': 'retain'
                }
            ))
        
        return qa_pairs
    
    def generate_unlearn_pairs(self, article: Article) -> List[QAPair]:
        """Generate Q&A pairs for unlearn dataset (refusal responses)."""
        qa_pairs = []
        
        # Generate questions from LLM (we only need the questions)
        raw_pairs = self.lm.generate_questions(
            article.title,
            article.content,
            self.questions_per_article
        )
        
        for pair in raw_pairs:
            # Use random refusal response instead of factual answer
            refusal = random.choice(REFUSAL_RESPONSES)
            
            qa_pairs.append(QAPair(
                instruction=pair['question'],
                input="",
                output=refusal,
                metadata={
                    'source_article_id': article.id,
                    'source_title': article.title,
                    'temporal_class': 'post_cutoff',
                    'earliest_date': article.earliest_date,
                    'latest_date': article.latest_date,
                    'dataset_type': 'unlearn',
                    'original_answer': pair['answer']  # Keep for validation
                }
            ))
        
        return qa_pairs
    
    def process_articles(
        self,
        articles: List[Article],
        dataset_type: str,  # 'retain' or 'unlearn'
        progress_desc: str = "Processing"
    ) -> List[QAPair]:
        """Process a batch of articles and generate Q&A pairs."""
        all_pairs = []
        
        for article in tqdm(articles, desc=progress_desc):
            try:
                if dataset_type == 'retain':
                    pairs = self.generate_retain_pairs(article)
                    self.stats.retain_qa_pairs += len(pairs)
                    self.stats.pre_cutoff_articles += 1
                else:
                    pairs = self.generate_unlearn_pairs(article)
                    self.stats.unlearn_qa_pairs += len(pairs)
                    self.stats.post_cutoff_articles += 1
                
                all_pairs.extend(pairs)
                self.stats.articles_processed += 1
                
                if not pairs:
                    self.stats.skipped_articles += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process article {article.id}: {e}")
                self.stats.failed_generations += 1
        
        return all_pairs
    
    def split_dataset(
        self, 
        pairs: List[QAPair], 
        val_ratio: float = 0.1
    ) -> Tuple[List[QAPair], List[QAPair]]:
        """Split dataset into train and validation sets."""
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - val_ratio))
        return pairs[:split_idx], pairs[split_idx:]


# -----------------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------------

def save_jsonl(pairs: List[QAPair], filepath: Path):
    """Save Q&A pairs to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(pairs)} pairs to {filepath}")


def save_articles_metadata(articles: List[Article], filepath: Path):
    """Save article metadata to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        {
            'id': a.id,
            'title': a.title,
            'earliest_date': a.earliest_date,
            'latest_date': a.latest_date,
            'temporal_class': a.temporal_class
        }
        for a in articles
    ]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} article metadata to {filepath}")


def save_statistics(stats: GenerationStats, config: Dict, filepath: Path):
    """Save generation statistics to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    output = {
        **stats.to_dict(),
        'config': config
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistics to {filepath}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate temporal fine-tuning datasets from Wikipedia',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode dev                    Generate small development subset
  %(prog)s --mode full                   Generate full datasets
  %(prog)s --dry-run                     Show statistics without generating
  %(prog)s --cutoff-date 1950-01-01      Use different temporal cutoff
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        choices=['dev', 'full'],
        default='dev',
        help='Generation mode: dev (small subset) or full (complete datasets)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show article statistics without generating datasets'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(WIKI_DATA, 'datasets'),
        help=f'Output directory for datasets (default: {WIKI_DATA}/datasets)'
    )
    
    # Temporal configuration
    parser.add_argument(
        '--cutoff-date',
        type=str,
        default=DEFAULT_CUTOFF_DATE,
        help=f'Temporal cutoff date YYYY-MM-DD (default: {DEFAULT_CUTOFF_DATE})'
    )
    
    # LM Studio configuration
    parser.add_argument(
        '--lmstudio-host',
        type=str,
        default=DEFAULT_LMSTUDIO_HOST,
        help=f'LM Studio server host (default: {DEFAULT_LMSTUDIO_HOST})'
    )
    parser.add_argument(
        '--lmstudio-port',
        type=int,
        default=DEFAULT_LMSTUDIO_PORT,
        help=f'LM Studio server port (default: {DEFAULT_LMSTUDIO_PORT})'
    )
    parser.add_argument(
        '--lmstudio-model',
        type=str,
        default=DEFAULT_LMSTUDIO_MODEL,
        help=f'LM Studio model name (default: {DEFAULT_LMSTUDIO_MODEL})'
    )
    
    # MCP configuration
    parser.add_argument(
        '--mcp-host',
        type=str,
        default=DEFAULT_MCP_HOST,
        help=f'MCP server host (default: {DEFAULT_MCP_HOST})'
    )
    parser.add_argument(
        '--mcp-port',
        type=int,
        default=DEFAULT_MCP_PORT,
        help=f'MCP server port (default: {DEFAULT_MCP_PORT})'
    )
    
    # Generation configuration
    parser.add_argument(
        '--retain-count',
        type=int,
        default=None,
        help='Number of retain articles to process (auto-calculated if not set)'
    )
    parser.add_argument(
        '--unlearn-count',
        type=int,
        default=None,
        help='Number of unlearn articles to process (auto-calculated if not set)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Articles to process per batch (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--questions-per-article',
        type=int,
        default=DEFAULT_QUESTIONS_PER_ARTICLE,
        help=f'Q&A pairs per article (default: {DEFAULT_QUESTIONS_PER_ARTICLE})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_SEED})'
    )
    
    # Database configuration
    parser.add_argument('--db-host', default=DEFAULT_DB_HOST, help='PostgreSQL host')
    parser.add_argument('--db-name', default=DEFAULT_DB_NAME, help='Database name')
    parser.add_argument('--db-user', default=DEFAULT_DB_USER, help='Database user')
    parser.add_argument('--db-password', default=DEFAULT_DB_PASSWORD, help='Database password')
    
    # Logging
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Temporal Dataset Generator")
    logger.info("=" * 60)
    
    # Calculate article counts based on mode
    if args.mode == 'dev':
        retain_count = args.retain_count or 200
        unlearn_count = args.unlearn_count or 100
        logger.info(f"Mode: DEVELOPMENT (small subset)")
    else:
        retain_count = args.retain_count or 35000  # ~100K Q&A pairs
        unlearn_count = args.unlearn_count or 17000  # ~50K Q&A pairs
        logger.info(f"Mode: FULL (complete datasets)")
    
    logger.info(f"Cutoff date: {args.cutoff_date}")
    logger.info(f"Target retain articles: {retain_count}")
    logger.info(f"Target unlearn articles: {unlearn_count}")
    logger.info(f"Questions per article: {args.questions_per_article}")
    
    # Initialize database connection
    db = DatabaseManager(
        host=args.db_host,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password
    )
    
    if not db.connect():
        logger.error("Failed to connect to database. Exiting.")
        sys.exit(1)
    
    try:
        # Get temporal statistics
        logger.info("\nQuerying temporal statistics...")
        stats = db.get_temporal_statistics(args.cutoff_date)
        
        logger.info(f"\n{'=' * 60}")
        logger.info("TEMPORAL ARTICLE STATISTICS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total articles with temporal info: {stats['total_temporal']:,}")
        logger.info(f"Pre-cutoff articles (retain):      {stats['pre_cutoff']:,}")
        logger.info(f"Post-cutoff articles (unlearn):    {stats['post_cutoff']:,}")
        logger.info(f"Spanning articles (excluded):      {stats['spanning']:,}")
        logger.info(f"{'=' * 60}\n")
        
        if args.dry_run:
            logger.info("Dry run complete. No datasets generated.")
            return
        
        # Initialize LM Studio client
        lm = LMStudioClient(
            host=args.lmstudio_host,
            port=args.lmstudio_port,
            model=args.lmstudio_model
        )
        
        if not lm.check_connection():
            logger.error("Failed to connect to LM Studio. Exiting.")
            sys.exit(1)
        
        # Initialize MCP client (optional)
        mcp = MCPClient(args.mcp_host, args.mcp_port)
        mcp_available = mcp.check_connection()
        if mcp_available:
            logger.info("MCP server connected (semantic search available)")
        else:
            logger.info("MCP server not available (proceeding without semantic enrichment)")
            mcp = None
        
        # Initialize generator
        generator = DatasetGenerator(
            db_manager=db,
            lm_client=lm,
            mcp_client=mcp,
            cutoff_date=args.cutoff_date,
            questions_per_article=args.questions_per_article,
            seed=args.seed
        )
        
        output_dir = Path(args.output_dir)
        
        # =====================================================================
        # Generate Retain Dataset (Pre-cutoff)
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING RETAIN DATASET (Pre-cutoff)")
        logger.info("=" * 60)
        
        # Fetch pre-cutoff articles
        logger.info(f"Fetching {retain_count} pre-cutoff articles...")
        retain_articles = db.fetch_pre_cutoff_articles(
            cutoff_date=args.cutoff_date,
            limit=retain_count,
            seed=args.seed
        )
        logger.info(f"Fetched {len(retain_articles)} articles")
        
        # Generate Q&A pairs
        retain_pairs = generator.process_articles(
            retain_articles,
            dataset_type='retain',
            progress_desc="Generating retain Q&A"
        )
        
        # Split into train/val
        retain_train, retain_val = generator.split_dataset(retain_pairs, val_ratio=0.1)
        
        # Save retain dataset
        save_jsonl(retain_train, output_dir / 'retain' / 'retain_train.jsonl')
        save_jsonl(retain_val, output_dir / 'retain' / 'retain_val.jsonl')
        save_articles_metadata(retain_articles, output_dir / 'retain' / 'retain_articles.json')
        
        # =====================================================================
        # Generate Unlearn Dataset (Post-cutoff)
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING UNLEARN DATASET (Post-cutoff)")
        logger.info("=" * 60)
        
        # Fetch post-cutoff articles
        logger.info(f"Fetching {unlearn_count} post-cutoff articles...")
        unlearn_articles = db.fetch_post_cutoff_articles(
            cutoff_date=args.cutoff_date,
            limit=unlearn_count,
            seed=args.seed
        )
        logger.info(f"Fetched {len(unlearn_articles)} articles")
        
        # Generate Q&A pairs
        unlearn_pairs = generator.process_articles(
            unlearn_articles,
            dataset_type='unlearn',
            progress_desc="Generating unlearn Q&A"
        )
        
        # Split into train/val
        unlearn_train, unlearn_val = generator.split_dataset(unlearn_pairs, val_ratio=0.1)
        
        # Save unlearn dataset
        save_jsonl(unlearn_train, output_dir / 'unlearn' / 'unlearn_train.jsonl')
        save_jsonl(unlearn_val, output_dir / 'unlearn' / 'unlearn_val.jsonl')
        save_articles_metadata(unlearn_articles, output_dir / 'unlearn' / 'unlearn_articles.json')
        
        # =====================================================================
        # Generate Development Subset
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING DEVELOPMENT SUBSET")
        logger.info("=" * 60)
        
        # Take samples from both datasets
        dev_retain = retain_train[:min(500, len(retain_train))]
        dev_unlearn = unlearn_train[:min(500, len(unlearn_train))]
        dev_combined = dev_retain + dev_unlearn
        random.shuffle(dev_combined)
        
        save_jsonl(dev_combined, output_dir / 'dev' / 'dev_subset.jsonl')
        
        # =====================================================================
        # Save Statistics
        # =====================================================================
        generator.stats.end_time = datetime.now()
        
        config = {
            'cutoff_date': args.cutoff_date,
            'mode': args.mode,
            'lmstudio_model': args.lmstudio_model,
            'questions_per_article': args.questions_per_article,
            'seed': args.seed
        }
        
        save_statistics(generator.stats, config, output_dir / 'statistics.json')
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Retain train: {len(retain_train)} Q&A pairs")
        logger.info(f"Retain val:   {len(retain_val)} Q&A pairs")
        logger.info(f"Unlearn train: {len(unlearn_train)} Q&A pairs")
        logger.info(f"Unlearn val:   {len(unlearn_val)} Q&A pairs")
        logger.info(f"Dev subset:    {len(dev_combined)} Q&A pairs")
        logger.info(f"Total time: {generator.stats.end_time - generator.stats.start_time}")
        logger.info("=" * 60)
        
    finally:
        db.close()


if __name__ == '__main__':
    main()
