#!/usr/bin/env python3
"""
Process extracted Wikipedia JSON files and index to PostgreSQL and OpenSearch.

This script:
1. Reads JSON files from /srv/wikipedia/extracted/wikipedia_batch_*.json
2. Splits articles into sections based on headers
3. Inserts articles and sections into PostgreSQL
4. Generates embeddings using LM Studio API (default) or sentence-transformers (fallback)
5. Creates OpenSearch index with k-NN vector search support
6. Bulk indexes documents with embeddings for fast retrieval
7. Supports checkpoint/resume for long-running processing jobs

Embedding Provider Options:
- 'lmstudio': Use external LM Studio server with nomic-embed-text-v1.5 (GPU accelerated)
- 'local': Use local sentence-transformers model (CPU, slower)

Checkpoint/Resume Feature:
    The script automatically saves progress to a checkpoint file during processing.
    If interrupted (Ctrl+C, error, or system shutdown), simply run the script again
    to resume from where it left off. The checkpoint tracks:
    - Current file and line number being processed
    - Total articles and sections processed
    - Timestamps for monitoring progress
    
    Checkpoint file location: $WIKI_DATA/processing_checkpoint.json (configurable via --checkpoint-file)
    Checkpoint save interval: Every 10 batches (configurable via CHECKPOINT_INTERVAL)

Usage:
    python3 process_and_index.py              # Run full processing (auto-resumes if interrupted)
    python3 process_and_index.py --test       # Run connectivity and setup tests only
    python3 process_and_index.py --status     # Show current checkpoint status and progress
    python3 process_and_index.py --reset      # Delete checkpoint and start fresh from beginning
    python3 process_and_index.py --checkpoint-file /path/to/file.json  # Use custom checkpoint file
    python3 process_and_index.py --help       # Show help

Examples:
    # Start processing (will auto-resume if previous run was interrupted)
    python3 process_and_index.py
    
    # Check how much progress has been made
    python3 process_and_index.py --status
    
    # Start over from the beginning (deletes previous progress)
    python3 process_and_index.py --reset
    
    # Use LM Studio on a remote server for embeddings
    # (edit LMSTUDIO_HOST in the script configuration section)
    python3 process_and_index.py --provider lmstudio

Environment Variables:
    WIKI_DATA   Required. Path to the Wikipedia data directory containing:
                - extracted/wikipedia_batch_*.json (extracted article files)
                The checkpoint file will also be stored in this directory.
                Example: export WIKI_DATA=/mnt/data/wikipedia
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
import psycopg2
from psycopg2.extras import execute_batch
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import NotFoundError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
# Choose embedding provider: 'lmstudio' (recommended) or 'local'
EMBEDDING_PROVIDER = 'lmstudio'

# LM Studio Configuration (when EMBEDDING_PROVIDER = 'lmstudio')
# Update LMSTUDIO_HOST to your LM Studio server IP address
LMSTUDIO_HOST = 'localhost'  # LM Studio server IP
LMSTUDIO_PORT = 1234
LMSTUDIO_URL = f'http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/embeddings'
LMSTUDIO_MODEL = 'text-embedding-nomic-embed-text-v1.5@f16'  # Model identifier from LM Studio
LMSTUDIO_CONTEXT_LENGTH = 2048  # Max tokens per text (model-dependent: 2048 or 8192)
LMSTUDIO_MODEL_BATCH_SIZE = 1024  # Max batch size in tokens (model config limit)
LMSTUDIO_TIMEOUT = 300  # Seconds per batch request (increased for larger batches)
# Optimized for Wikipedia paragraph-level text (~500 chars, ~125 tokens per paragraph)
# Based on benchmarks: 16-32 paragraphs per batch achieves best throughput
LMSTUDIO_BATCH_SIZE = 32  # Texts per embedding API call (optimized for paragraph-level)

# Local embedding configuration (when EMBEDDING_PROVIDER = 'local')
LOCAL_MODEL = 'all-mpnet-base-v2'
LOCAL_BATCH_SIZE = 32  # Texts per batch for local sentence-transformers (CPU limited)

# Embedding dimension (768 for both nomic-embed-text-v1.5 and all-mpnet-base-v2)
EMBEDDING_DIM = 768

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================
# WIKI_DATA environment variable is required and must point to the Wikipedia data directory
# containing: extracted/wikipedia_batch_*.json
WIKI_DATA_DIR = os.environ.get('WIKI_DATA', '')
EXTRACTED_DIR = os.path.join(WIKI_DATA_DIR, 'extracted') if WIKI_DATA_DIR else ''
BATCH_SIZE = 100
OPENSEARCH_BULK_SIZE = 500
CHECKPOINT_FILE = os.path.join(WIKI_DATA_DIR, 'processing_checkpoint.json') if WIKI_DATA_DIR else ''
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N batches


def validate_wiki_data_env() -> Tuple[bool, str]:
    """
    Validate that WIKI_DATA environment variable is set and points to valid data.
    
    Returns:
        Tuple of (success, message)
    """
    if not WIKI_DATA_DIR:
        return False, (
            "WIKI_DATA environment variable is not set. "
            "Please set it to the Wikipedia data directory.\n"
            "Example: export WIKI_DATA=/srv/wikipedia"
        )
    
    if not os.path.isdir(WIKI_DATA_DIR):
        return False, f"WIKI_DATA directory does not exist: {WIKI_DATA_DIR}"
    
    if not os.path.isdir(EXTRACTED_DIR):
        return False, (
            f"Extracted data directory not found: {EXTRACTED_DIR}\n"
            f"Expected directory structure: $WIKI_DATA/extracted/wikipedia_batch_*.json"
        )
    
    # Check for at least one batch file
    batch_files = list(Path(EXTRACTED_DIR).glob('wikipedia_batch_*.json'))
    if not batch_files:
        return False, (
            f"No Wikipedia batch files found in: {EXTRACTED_DIR}\n"
            f"Expected files matching: wikipedia_batch_*.json"
        )
    
    return True, f"WIKI_DATA validated: {len(batch_files)} batch file(s) found in {EXTRACTED_DIR}"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'wikidb',
    'user': 'wiki',
    'password': 'wikipass'
}

# OpenSearch configuration
OPENSEARCH_CONFIG = {
    'hosts': [{'host': 'localhost', 'port': 9200}],
    'http_auth': None,
    'use_ssl': False,
    'verify_certs': False
}

INDEX_NAME = 'wikipedia'
TEST_INDEX_NAME = 'wikipedia_test'  # Separate index for testing


# =============================================================================
# DATABASE CONNECTION CLASSES
# =============================================================================

class PostgreSQLConnection:
    """Manage PostgreSQL database connections and operations."""
    
    def __init__(self, config: Dict = None):
        """Initialize PostgreSQL connection."""
        self.config = config or DB_CONFIG
        self.conn = None
        self.cursor = None
    
    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor()
            logging.info("Connected to PostgreSQL")
            return True
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test PostgreSQL connection and verify schema.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.conn:
                self.connect()
            
            # Test basic connectivity
            self.cursor.execute("SELECT version();")
            version = self.cursor.fetchone()[0]
            logging.info(f"PostgreSQL version: {version[:50]}...")
            
            # Check required tables exist
            self.cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name IN ('articles', 'sections', 'redirects')
            """)
            tables = [row[0] for row in self.cursor.fetchall()]
            
            required_tables = {'articles', 'sections', 'redirects'}
            missing = required_tables - set(tables)
            
            if missing:
                return False, f"Missing tables: {missing}. Run schema.sql first."
            
            # Check pg_trgm extension
            self.cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_trgm';")
            if not self.cursor.fetchone():
                return False, "pg_trgm extension not installed. Run: CREATE EXTENSION pg_trgm;"
            
            return True, "PostgreSQL connection and schema verified"
            
        except Exception as e:
            return False, f"PostgreSQL test failed: {e}"
    
    def insert_article(self, article_id: str, title: str, url: str, content: str) -> int:
        """Insert article and return database ID."""
        self.cursor.execute(
            """
            INSERT INTO articles (title, content, url)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (title, content, url)
        )
        db_article_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return db_article_id
    
    def insert_sections(self, db_article_id: int, sections: List[Tuple[str, str]]):
        """Insert article sections."""
        section_data = [
            (db_article_id, title, text, idx)
            for idx, (title, text) in enumerate(sections)
        ]
        
        execute_batch(
            self.cursor,
            """
            INSERT INTO sections (article_id, section_title, section_text, section_order)
            VALUES (%s, %s, %s, %s)
            """,
            section_data
        )
        self.conn.commit()
    
    def delete_article(self, db_article_id: int):
        """Delete article and its sections (for cleanup)."""
        self.cursor.execute("DELETE FROM sections WHERE article_id = %s", (db_article_id,))
        self.cursor.execute("DELETE FROM articles WHERE id = %s", (db_article_id,))
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logging.info("PostgreSQL connection closed")


class OpenSearchConnection:
    """Manage OpenSearch connections and operations."""
    
    def __init__(self, config: Dict = None):
        """Initialize OpenSearch connection."""
        self.config = config or OPENSEARCH_CONFIG
        self.client = None
    
    def connect(self) -> bool:
        """
        Establish connection to OpenSearch.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = OpenSearch(**self.config)
            # Test connection
            info = self.client.info()
            logging.info(f"Connected to OpenSearch {info['version']['number']}")
            return True
        except Exception as e:
            logging.error(f"OpenSearch connection failed: {e}")
            return False
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test OpenSearch connection and verify k-NN plugin.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.client:
                self.connect()
            
            # Test basic connectivity
            info = self.client.info()
            version = info['version']['number']
            logging.info(f"OpenSearch version: {version}")
            
            # Check cluster health
            health = self.client.cluster.health()
            status = health['status']
            logging.info(f"Cluster status: {status}")
            
            if status == 'red':
                return False, "OpenSearch cluster status is RED"
            
            # Check k-NN plugin is available
            try:
                plugins = self.client.cat.plugins(format='json')
                knn_installed = any('knn' in p.get('component', '').lower() for p in plugins)
                if not knn_installed:
                    logging.warning("k-NN plugin may not be installed (couldn't verify)")
            except Exception:
                logging.warning("Could not verify k-NN plugin status")
            
            return True, f"OpenSearch connection verified (version {version}, status: {status})"
            
        except Exception as e:
            return False, f"OpenSearch test failed: {e}"
    
    def create_index(self, index_name: str, embedding_dim: int):
        """Create index with k-NN vector field mappings."""
        if self.client.indices.exists(index=index_name):
            logging.warning(f"Index '{index_name}' already exists. Deleting...")
            self.client.indices.delete(index=index_name)
        
        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True
                }
            },
            'mappings': {
                'properties': {
                    'article_id': {'type': 'keyword'},
                    'section_id': {'type': 'keyword'},
                    'title': {
                        'type': 'text',
                        'analyzer': 'english',
                        'fields': {
                            'keyword': {'type': 'keyword'}
                        }
                    },
                    'section_title': {
                        'type': 'text',
                        'analyzer': 'english'
                    },
                    'text': {
                        'type': 'text',
                        'analyzer': 'english'
                    },
                    'url': {'type': 'keyword'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'lucene',
                            'parameters': {
                                'ef_construction': 128,
                                'm': 16
                            }
                        }
                    }
                }
            }
        }
        
        self.client.indices.create(index=index_name, body=index_body)
        logging.info(f"Created OpenSearch index '{index_name}' with k-NN support")
    
    def bulk_index(self, index_name: str, documents: List[Dict]) -> int:
        """Bulk index documents to OpenSearch."""
        actions = [
            {
                '_index': index_name,
                '_id': doc['section_id'],
                '_source': doc
            }
            for doc in documents
        ]
        
        success, failed = helpers.bulk(
            self.client,
            actions,
            chunk_size=OPENSEARCH_BULK_SIZE,
            request_timeout=60
        )
        
        if failed:
            logging.warning(f"Failed to index {len(failed)} documents")
        
        return success
    
    def delete_index(self, index_name: str):
        """Delete an index (for cleanup)."""
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logging.info(f"Deleted index '{index_name}'")
        except NotFoundError:
            pass
    
    def search_knn(self, index_name: str, embedding: List[float], k: int = 5) -> List[Dict]:
        """Perform k-NN search."""
        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": k
                    }
                }
            }
        }
        
        response = self.client.search(index=index_name, body=query)
        return response['hits']['hits']
    
    def close(self):
        """Close OpenSearch connection."""
        if self.client:
            self.client.close()
            logging.info("OpenSearch connection closed")


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """
    Manage processing checkpoints for resume functionality.
    
    Saves progress to a JSON file, allowing the script to resume
    from where it left off after an interruption.
    """
    
    def __init__(self, checkpoint_file: str = None):
        """Initialize checkpoint manager."""
        self.checkpoint_file = Path(checkpoint_file or CHECKPOINT_FILE)
        self.state = {
            'file_index': 0,           # Index of current file being processed
            'line_number': 0,          # Line number within current file (1-indexed)
            'total_articles': 0,       # Total articles processed so far
            'total_indexed': 0,        # Total sections indexed so far
            'last_file': None,         # Name of last file being processed
            'last_article_id': None,   # Last article ID processed
            'started_at': None,        # Timestamp when processing started
            'updated_at': None,        # Timestamp of last checkpoint save
            'completed': False         # Whether processing completed successfully
        }
        self._batches_since_save = 0
    
    def load(self) -> bool:
        """
        Load checkpoint from file.
        
        Returns:
            True if checkpoint was loaded, False if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            logging.info("No checkpoint file found. Starting fresh.")
            return False
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                saved_state = json.load(f)
            
            # Validate checkpoint structure
            required_keys = {'file_index', 'line_number', 'total_articles'}
            if not required_keys.issubset(saved_state.keys()):
                logging.warning("Invalid checkpoint file format. Starting fresh.")
                return False
            
            # Check if previous run completed
            if saved_state.get('completed', False):
                logging.info("Previous run completed successfully. Starting fresh.")
                return False
            
            self.state.update(saved_state)
            logging.info(f"Loaded checkpoint: file_index={self.state['file_index']}, "
                        f"line={self.state['line_number']}, "
                        f"articles={self.state['total_articles']}, "
                        f"last_file={self.state.get('last_file', 'N/A')}")
            return True
            
        except json.JSONDecodeError as e:
            logging.warning(f"Corrupted checkpoint file: {e}. Starting fresh.")
            return False
        except Exception as e:
            logging.warning(f"Error loading checkpoint: {e}. Starting fresh.")
            return False
    
    def save(self, force: bool = False):
        """
        Save current state to checkpoint file.
        
        Args:
            force: If True, save immediately regardless of interval
        """
        self._batches_since_save += 1
        
        if not force and self._batches_since_save < CHECKPOINT_INTERVAL:
            return
        
        self._batches_since_save = 0
        self.state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # Ensure directory exists
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file first, then rename (atomic operation)
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
            
            # Atomic rename
            temp_file.replace(self.checkpoint_file)
            logging.debug(f"Checkpoint saved: file_index={self.state['file_index']}, line={self.state['line_number']}")
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
    
    def update(self, file_index: int = None, line_number: int = None,
               total_articles: int = None, total_indexed: int = None,
               last_file: str = None, last_article_id: str = None):
        """
        Update checkpoint state.
        
        Args:
            file_index: Current file index
            line_number: Current line number in file (1-indexed)
            total_articles: Total articles processed
            total_indexed: Total sections indexed
            last_file: Name of current file
            last_article_id: Last article ID processed
        """
        if file_index is not None:
            self.state['file_index'] = file_index
        if line_number is not None:
            self.state['line_number'] = line_number
        if total_articles is not None:
            self.state['total_articles'] = total_articles
        if total_indexed is not None:
            self.state['total_indexed'] = total_indexed
        if last_file is not None:
            self.state['last_file'] = last_file
        if last_article_id is not None:
            self.state['last_article_id'] = last_article_id
    
    def mark_started(self):
        """Mark processing as started."""
        if not self.state['started_at']:
            self.state['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.state['completed'] = False
        self.save(force=True)
    
    def mark_completed(self):
        """Mark processing as completed."""
        self.state['completed'] = True
        self.state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save(force=True)
        logging.info("Processing marked as completed in checkpoint.")
    
    def reset(self):
        """Delete checkpoint file to start fresh."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logging.info(f"Checkpoint file deleted: {self.checkpoint_file}")
        self.state = {
            'file_index': 0,
            'line_number': 0,
            'total_articles': 0,
            'total_indexed': 0,
            'last_file': None,
            'last_article_id': None,
            'started_at': None,
            'updated_at': None,
            'completed': False
        }
    
    def get_resume_position(self) -> Tuple[int, int]:
        """
        Get the position to resume from.
        
        Returns:
            Tuple of (file_index, line_number) to resume from
        """
        return self.state['file_index'], self.state['line_number']
    
    def get_stats(self) -> Dict:
        """Get current checkpoint statistics."""
        return self.state.copy()


class EmbeddingProvider:
    """Abstract base for embedding providers."""
    
    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors. None values indicate failed embeddings.
        """
        raise NotImplementedError
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test provider connectivity. Override in subclasses."""
        raise NotImplementedError


class LMStudioEmbedding(EmbeddingProvider):
    """
    Generate embeddings using LM Studio's OpenAI-compatible API.
    
    Optimized for batch processing with token-aware batching to maximize
    throughput while staying within model limits. Based on benchmarks,
    paragraph-level text (~500 chars, 16-32 texts per batch) achieves
    best throughput for Wikipedia processing.
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds, doubles with each retry
    
    def __init__(self, skip_connection_test: bool = False):
        """
        Initialize LM Studio embedding provider.
        
        Args:
            skip_connection_test: If True, don't test connection on init (for test mode)
        """
        logging.info(f"Initializing LM Studio embedding provider...")
        logging.info(f"  Server: {LMSTUDIO_URL}")
        logging.info(f"  Model: {LMSTUDIO_MODEL}")
        logging.info(f"  Context length: {LMSTUDIO_CONTEXT_LENGTH} tokens")
        logging.info(f"  Model batch size: {LMSTUDIO_MODEL_BATCH_SIZE} tokens")
        logging.info(f"  Max texts per batch: {LMSTUDIO_BATCH_SIZE}")
        
        if not skip_connection_test:
            success, msg = self.test_connection()
            if not success:
                raise ConnectionError(msg)
            logging.info("LM Studio connection verified")
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for a text (rough approximation).
        Uses ~4 chars per token as a conservative estimate.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // 4 + 1
    
    @staticmethod
    def max_safe_text_length(num_texts: int = 1) -> int:
        """
        Calculate maximum safe text length in characters for a batch.
        
        Args:
            num_texts: Number of texts in the batch
            
        Returns:
            Maximum characters per text to stay within limits
        """
        # Use the more restrictive limit
        tokens_per_text = min(
            LMSTUDIO_CONTEXT_LENGTH // max(num_texts, 1),
            LMSTUDIO_MODEL_BATCH_SIZE // max(num_texts, 1)
        )
        # Conservative: 3 chars per token to leave headroom
        return max(tokens_per_text * 3, 50)
    
    def _create_token_aware_batches(self, texts: List[str]) -> List[List[Tuple[int, str]]]:
        """
        Create batches that respect both text count and token limits.
        
        This ensures we don't exceed LMSTUDIO_MODEL_BATCH_SIZE tokens per batch
        while also respecting LMSTUDIO_BATCH_SIZE max texts per batch.
        
        Args:
            texts: List of text strings to batch
            
        Returns:
            List of batches, where each batch is a list of (original_index, text) tuples
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for i, text in enumerate(texts):
            # Truncate text if it exceeds context length
            max_chars = int(LMSTUDIO_CONTEXT_LENGTH * 3.5)
            truncated_text = text[:max_chars] if len(text) > max_chars else text
            text_tokens = self.estimate_tokens(truncated_text)
            
            # Check if adding this text would exceed limits
            would_exceed_tokens = current_tokens + text_tokens > LMSTUDIO_MODEL_BATCH_SIZE
            would_exceed_count = len(current_batch) >= LMSTUDIO_BATCH_SIZE
            
            if current_batch and (would_exceed_tokens or would_exceed_count):
                # Start a new batch
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append((i, truncated_text))
            current_tokens += text_tokens
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test connection to LM Studio server and verify embedding generation.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Try a simple embedding request
            response = requests.post(
                LMSTUDIO_URL,
                json={'model': LMSTUDIO_MODEL, 'input': ['test connection']},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Verify response structure
            if 'data' not in data or len(data['data']) == 0:
                return False, "LM Studio returned empty embedding data"
            
            embedding = data['data'][0].get('embedding', [])
            if len(embedding) != EMBEDDING_DIM:
                return False, f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}"
            
            # Verify embedding is not all zeros (some models return zeros on error)
            if all(v == 0.0 for v in embedding):
                return False, "LM Studio returned zero vector - model may not be loaded correctly"
            
            return True, f"LM Studio connection verified (model: {LMSTUDIO_MODEL}, dim: {len(embedding)})"
            
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to LM Studio at {LMSTUDIO_URL}. Is the server running?"
        except requests.exceptions.Timeout:
            return False, f"Connection to LM Studio timed out. Server may be overloaded."
        except Exception as e:
            return False, f"LM Studio connection test failed: {e}"
    
    def _is_valid_embedding(self, embedding: List[float]) -> bool:
        """
        Check if an embedding is valid (not null, correct dimension, not all zeros).
        
        Args:
            embedding: The embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        if embedding is None:
            return False
        if len(embedding) != EMBEDDING_DIM:
            return False
        # Check for all zeros (invalid for cosine similarity)
        if all(v == 0.0 for v in embedding):
            return False
        return True
    
    def _request_embeddings_with_retry(
        self, 
        texts: List[str], 
        pre_truncated: bool = False
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Request embeddings from LM Studio with retry logic.
        
        Args:
            texts: List of text strings to embed
            pre_truncated: If True, texts are already truncated (skip truncation step)
            
        Returns:
            Tuple of (embeddings list with None for failures, list of failed indices)
        """
        # Truncate texts if not already done by token-aware batching
        if pre_truncated:
            request_texts = texts
        else:
            max_chars = int(LMSTUDIO_CONTEXT_LENGTH * 3.5)
            request_texts = [text[:max_chars] for text in texts]
        
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        failed_indices: List[int] = []
        
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    LMSTUDIO_URL,
                    json={
                        'model': LMSTUDIO_MODEL,
                        'input': request_texts
                    },
                    headers={'Content-Type': 'application/json'},
                    timeout=LMSTUDIO_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Check for error in response
                if 'error' in data:
                    raise ValueError(f"API returned error: {data['error']}")
                
                if 'data' not in data or len(data['data']) == 0:
                    raise ValueError("API returned empty data")
                
                # Extract embeddings in order (API returns them sorted by index)
                for item in sorted(data['data'], key=lambda x: x['index']):
                    idx = item['index']
                    emb = item.get('embedding')
                    
                    if idx < len(embeddings) and self._is_valid_embedding(emb):
                        embeddings[idx] = emb
                    else:
                        if idx < len(embeddings):
                            failed_indices.append(idx)
                            logging.warning(
                                f"Invalid embedding at index {idx}: "
                                f"dim={len(emb) if emb else 'None'}, "
                                f"text preview='{texts[idx][:50]}...'"
                            )
                
                # Success - return results
                return embeddings, failed_indices
                
            except requests.exceptions.Timeout as e:
                last_error = e
                logging.warning(f"LM Studio timeout (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logging.warning(f"LM Studio connection error (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
            except requests.exceptions.HTTPError as e:
                last_error = e
                # Log response body for debugging
                try:
                    error_body = e.response.text[:500] if e.response else 'No response body'
                    logging.warning(f"LM Studio HTTP error (attempt {attempt + 1}/{self.MAX_RETRIES}): {e} - {error_body}")
                except Exception:
                    logging.warning(f"LM Studio HTTP error (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
            except Exception as e:
                last_error = e
                logging.warning(f"LM Studio error (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.MAX_RETRIES - 1:
                delay = self.RETRY_DELAY * (2 ** attempt)
                logging.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        
        # All retries failed
        logging.error(f"LM Studio API failed after {self.MAX_RETRIES} attempts: {last_error}")
        failed_indices = list(range(len(texts)))
        return embeddings, failed_indices
    
    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Get embeddings from LM Studio server with token-aware batching.
        
        Uses dynamic batch sizing based on estimated token counts to maximize
        throughput while staying within model limits (context_length and 
        model_batch_size).
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (None for texts that failed to embed)
        """
        if not texts:
            return []
        
        # Create token-aware batches
        batches = self._create_token_aware_batches(texts)
        
        # Pre-allocate result array
        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        total_failed = 0
        
        for batch_idx, batch in enumerate(batches):
            # Extract texts and original indices
            batch_texts = [text for _, text in batch]
            original_indices = [idx for idx, _ in batch]
            
            # Request embeddings (texts are pre-truncated by _create_token_aware_batches)
            embeddings, failed_indices = self._request_embeddings_with_retry(
                batch_texts, pre_truncated=True
            )
            
            # Map embeddings back to original indices
            for i, embedding in enumerate(embeddings):
                orig_idx = original_indices[i]
                all_embeddings[orig_idx] = embedding
            
            if failed_indices:
                total_failed += len(failed_indices)
                # Log sample of failed texts for debugging
                for idx in failed_indices[:3]:  # Only log first 3
                    orig_idx = original_indices[idx]
                    logging.warning(
                        f"Failed to embed text at index {orig_idx}: "
                        f"'{texts[orig_idx][:100]}...'"
                    )
                if len(failed_indices) > 3:
                    logging.warning(f"... and {len(failed_indices) - 3} more failed in this batch")
            
            # Log progress for large datasets
            if len(batches) > 10 and (batch_idx + 1) % 10 == 0:
                processed = sum(len(b) for b in batches[:batch_idx + 1])
                logging.info(f"Embedding progress: {processed}/{len(texts)} texts ({batch_idx + 1}/{len(batches)} batches)")
        
        if total_failed > 0:
            logging.warning(
                f"Embedding generation completed with {total_failed}/{len(texts)} failures "
                f"({100 * total_failed / len(texts):.1f}%)"
            )
        
        return all_embeddings


class LocalEmbedding(EmbeddingProvider):
    """Generate embeddings using local sentence-transformers model."""
    
    def __init__(self, skip_connection_test: bool = False):
        """
        Initialize local sentence-transformers embedding provider.
        
        Args:
            skip_connection_test: If True, don't load model on init (for test mode)
        """
        logging.info("Initializing local sentence-transformers embedding provider...")
        logging.info(f"  Model: {LOCAL_MODEL}")
        logging.info("  Note: This will be slow on CPU. Consider using LM Studio with GPU.")
        
        self.model = None
        self.embedding_dim = EMBEDDING_DIM
        
        if not skip_connection_test:
            # Lazy import to avoid loading torch when using LM Studio
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(LOCAL_MODEL)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logging.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test local embedding model availability.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try loading the model
            model = SentenceTransformer(LOCAL_MODEL)
            dim = model.get_sentence_embedding_dimension()
            
            # Test embedding generation
            test_embedding = model.encode(['test connection'], convert_to_numpy=True)
            
            if test_embedding.shape[1] != EMBEDDING_DIM:
                return False, f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {test_embedding.shape[1]}"
            
            return True, f"Local embedding model verified (model: {LOCAL_MODEL}, dim: {dim})"
            
        except ImportError:
            return False, "sentence-transformers not installed. Run: pip install sentence-transformers"
        except Exception as e:
            return False, f"Local embedding test failed: {e}"
    
    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings using local model.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors (None values indicate failures)
        """
        # Lazy load model if not already loaded
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(LOCAL_MODEL)
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=LOCAL_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Local embedding generation failed: {e}")
            # Return None for all texts on failure
            return [None] * len(texts)


def create_embedding_provider(skip_connection_test: bool = False) -> EmbeddingProvider:
    """
    Factory function to create the configured embedding provider.
    
    Args:
        skip_connection_test: If True, skip connection test during initialization
    """
    if EMBEDDING_PROVIDER == 'lmstudio':
        return LMStudioEmbedding(skip_connection_test=skip_connection_test)
    elif EMBEDDING_PROVIDER == 'local':
        return LocalEmbedding(skip_connection_test=skip_connection_test)
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")


# =============================================================================
# TEST RUNNER
# =============================================================================

class ConnectivityTester:
    """Test connectivity and functionality of all components."""
    
    # Sample test data
    TEST_ARTICLE = {
        'id': 'test_999999999',
        'title': 'Test Article for Connectivity Check',
        'url': 'https://en.wikipedia.org/wiki?curid=999999999',
        'text': '''This is a test article used to verify the Wikipedia processing pipeline.

## Introduction

This section contains introductory text that is long enough to be indexed properly. 
The text needs to be at least 50 characters to pass the section filter.

## Test Section

This is another section with enough content to be processed by the indexing system.
It contains information about testing database connectivity and embedding generation.

## Conclusion

The test article concludes here with final remarks about the verification process.
All systems should process this content correctly if configured properly.
'''
    }
    
    def __init__(self):
        """Initialize test components."""
        self.pg_conn = None
        self.os_conn = None
        self.embedding_provider = None
        self.test_article_id = None
        self.results = []
    
    def run_all_tests(self) -> bool:
        """
        Run all connectivity and functionality tests.
        
        Returns:
            True if all tests pass, False otherwise
        """
        logging.info("=" * 80)
        logging.info("CONNECTIVITY AND SETUP TEST MODE")
        logging.info("=" * 80)
        
        all_passed = True
        
        # Test 0: Source data availability (WIKI_DATA environment variable)
        all_passed &= self._test_source_data()
        
        # Test 1: PostgreSQL
        all_passed &= self._test_postgresql()
        
        # Test 2: OpenSearch  
        all_passed &= self._test_opensearch()
        
        # Test 3: Embedding Provider
        all_passed &= self._test_embedding_provider()
        
        # Test 4: End-to-end pipeline (if all components work)
        if all_passed:
            all_passed &= self._test_full_pipeline()
        
        # Cleanup
        self._cleanup()
        
        # Summary
        logging.info("=" * 80)
        logging.info("TEST RESULTS SUMMARY")
        logging.info("=" * 80)
        
        for test_name, passed, message in self.results:
            status = "✓ PASS" if passed else "✗ FAIL"
            logging.info(f"  {status}: {test_name}")
            if not passed:
                logging.info(f"         {message}")
        
        logging.info("=" * 80)
        if all_passed:
            logging.info("All tests PASSED. System is ready for full processing.")
        else:
            logging.error("Some tests FAILED. Please fix issues before running full processing.")
        logging.info("=" * 80)
        
        return all_passed
    
    def _test_source_data(self) -> bool:
        """Test that WIKI_DATA environment variable is set and source data exists."""
        logging.info("")
        logging.info("Testing source data availability (WIKI_DATA)...")
        
        success, message = validate_wiki_data_env()
        self.results.append(("Source Data (WIKI_DATA)", success, message))
        
        if success:
            logging.info(f"  ✓ {message}")
        else:
            logging.error(f"  ✗ {message}")
        
        return success
    
    def _test_postgresql(self) -> bool:
        """Test PostgreSQL connectivity and schema."""
        logging.info("")
        logging.info("Testing PostgreSQL connection...")
        
        try:
            self.pg_conn = PostgreSQLConnection()
            if not self.pg_conn.connect():
                self.results.append(("PostgreSQL Connection", False, "Connection failed"))
                return False
            
            success, message = self.pg_conn.test_connection()
            self.results.append(("PostgreSQL Connection", success, message))
            
            if success:
                logging.info(f"  ✓ {message}")
            else:
                logging.error(f"  ✗ {message}")
            
            return success
            
        except Exception as e:
            message = f"PostgreSQL test error: {e}"
            self.results.append(("PostgreSQL Connection", False, message))
            logging.error(f"  ✗ {message}")
            return False
    
    def _test_opensearch(self) -> bool:
        """Test OpenSearch connectivity and k-NN support."""
        logging.info("")
        logging.info("Testing OpenSearch connection...")
        
        try:
            self.os_conn = OpenSearchConnection()
            if not self.os_conn.connect():
                self.results.append(("OpenSearch Connection", False, "Connection failed"))
                return False
            
            success, message = self.os_conn.test_connection()
            self.results.append(("OpenSearch Connection", success, message))
            
            if success:
                logging.info(f"  ✓ {message}")
            else:
                logging.error(f"  ✗ {message}")
            
            return success
            
        except Exception as e:
            message = f"OpenSearch test error: {e}"
            self.results.append(("OpenSearch Connection", False, message))
            logging.error(f"  ✗ {message}")
            return False
    
    def _test_embedding_provider(self) -> bool:
        """Test embedding provider connectivity."""
        logging.info("")
        logging.info(f"Testing embedding provider ({EMBEDDING_PROVIDER})...")
        
        try:
            self.embedding_provider = create_embedding_provider(skip_connection_test=True)
            success, message = self.embedding_provider.test_connection()
            
            test_name = f"Embedding Provider ({EMBEDDING_PROVIDER})"
            self.results.append((test_name, success, message))
            
            if success:
                logging.info(f"  ✓ {message}")
            else:
                logging.error(f"  ✗ {message}")
            
            return success
            
        except Exception as e:
            message = f"Embedding provider test error: {e}"
            test_name = f"Embedding Provider ({EMBEDDING_PROVIDER})"
            self.results.append((test_name, False, message))
            logging.error(f"  ✗ {message}")
            return False
    
    def _test_full_pipeline(self) -> bool:
        """Test the full processing pipeline with dummy data."""
        logging.info("")
        logging.info("Testing full pipeline with dummy data...")
        
        try:
            # Step 1: Insert test article to PostgreSQL
            logging.info("  Inserting test article to PostgreSQL...")
            self.test_article_id = self.pg_conn.insert_article(
                self.TEST_ARTICLE['id'],
                self.TEST_ARTICLE['title'],
                self.TEST_ARTICLE['url'],
                self.TEST_ARTICLE['text']
            )
            logging.info(f"    Article inserted with DB ID: {self.test_article_id}")
            
            # Step 2: Create test OpenSearch index
            logging.info("  Creating test OpenSearch index...")
            self.os_conn.create_index(TEST_INDEX_NAME, EMBEDDING_DIM)
            logging.info(f"    Index '{TEST_INDEX_NAME}' created")
            
            # Step 3: Generate embeddings
            logging.info("  Generating test embedding...")
            test_texts = ["This is a test section for embedding generation verification."]
            embeddings = self.embedding_provider.get_embeddings(test_texts)
            
            if not embeddings or len(embeddings[0]) != EMBEDDING_DIM:
                raise ValueError(f"Invalid embedding: expected dim {EMBEDDING_DIM}, got {len(embeddings[0]) if embeddings else 0}")
            
            logging.info(f"    Embedding generated (dim: {len(embeddings[0])})")
            
            # Step 4: Index to OpenSearch
            logging.info("  Indexing test document to OpenSearch...")
            test_doc = {
                'article_id': self.TEST_ARTICLE['id'],
                'section_id': f"{self.TEST_ARTICLE['id']}_0",
                'title': self.TEST_ARTICLE['title'],
                'section_title': 'Test Section',
                'text': test_texts[0],
                'url': self.TEST_ARTICLE['url'],
                'embedding': embeddings[0]
            }
            indexed = self.os_conn.bulk_index(TEST_INDEX_NAME, [test_doc])
            logging.info(f"    Documents indexed: {indexed}")
            
            # Step 5: Verify k-NN search works
            logging.info("  Testing k-NN search...")
            # Small delay to ensure document is indexed
            time.sleep(1)
            
            # Force index refresh
            self.os_conn.client.indices.refresh(index=TEST_INDEX_NAME)
            
            results = self.os_conn.search_knn(TEST_INDEX_NAME, embeddings[0], k=1)
            if not results:
                raise ValueError("k-NN search returned no results")
            
            logging.info(f"    k-NN search successful, found {len(results)} result(s)")
            
            self.results.append(("Full Pipeline Test", True, "All pipeline steps completed successfully"))
            logging.info("  ✓ Full pipeline test passed")
            return True
            
        except Exception as e:
            message = f"Pipeline test error: {e}"
            self.results.append(("Full Pipeline Test", False, message))
            logging.error(f"  ✗ {message}")
            return False
    
    def _cleanup(self):
        """Clean up test data."""
        logging.info("")
        logging.info("Cleaning up test data...")
        
        try:
            # Delete test article from PostgreSQL
            if self.pg_conn and self.test_article_id:
                self.pg_conn.delete_article(self.test_article_id)
                logging.info(f"  Deleted test article (ID: {self.test_article_id}) from PostgreSQL")
        except Exception as e:
            logging.warning(f"  Warning: Could not clean PostgreSQL test data: {e}")
        
        try:
            # Delete test index from OpenSearch
            if self.os_conn:
                self.os_conn.delete_index(TEST_INDEX_NAME)
                logging.info(f"  Deleted test index '{TEST_INDEX_NAME}' from OpenSearch")
        except Exception as e:
            logging.warning(f"  Warning: Could not clean OpenSearch test data: {e}")
        
        # Close connections
        try:
            if self.pg_conn:
                self.pg_conn.close()
            if self.os_conn:
                self.os_conn.close()
        except Exception:
            pass
        
        logging.info("  Cleanup complete")


# =============================================================================
# WIKIPEDIA PROCESSOR
# =============================================================================

class WikipediaProcessor:
    """Process Wikipedia articles and index them to PostgreSQL and OpenSearch."""
    
    def __init__(self, checkpoint_manager: CheckpointManager = None, skip_index_creation: bool = False):
        """
        Initialize database connections and embedding provider.
        
        Args:
            checkpoint_manager: Optional checkpoint manager for resume support
            skip_index_creation: If True, don't recreate OpenSearch index (for resume)
        """
        logging.info("Initializing Wikipedia processor...")
        logging.info(f"Embedding provider: {EMBEDDING_PROVIDER}")
        
        self.checkpoint = checkpoint_manager
        
        # Initialize connections using connection classes
        self.pg_conn = PostgreSQLConnection()
        if not self.pg_conn.connect():
            raise ConnectionError("Failed to connect to PostgreSQL")
        
        self.os_conn = OpenSearchConnection()
        if not self.os_conn.connect():
            raise ConnectionError("Failed to connect to OpenSearch")
        
        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider()
        self.embedding_dim = EMBEDDING_DIM
        logging.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize OpenSearch index (skip if resuming)
        if not skip_index_creation:
            self.os_conn.create_index(INDEX_NAME, self.embedding_dim)
        else:
            logging.info(f"Skipping index creation (resume mode) - using existing '{INDEX_NAME}' index")
    
    def split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split article text into sections based on headers.
        
        Args:
            text: The full article text
            
        Returns:
            List of (section_title, section_text) tuples
        """
        # Split on markdown-style headers (## Section Title)
        sections = []
        lines = text.split('\n')
        
        current_title = 'Introduction'
        current_text = []
        
        for line in lines:
            # Check if line is a header (## or more #'s)
            header_match = re.match(r'^(#{2,})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_text:
                    sections.append((current_title, '\n'.join(current_text).strip()))
                
                # Start new section
                current_title = header_match.group(2).strip()
                current_text = []
            else:
                current_text.append(line)
        
        # Add final section
        if current_text:
            sections.append((current_title, '\n'.join(current_text).strip()))
        
        # Filter out very short sections (< 50 characters)
        sections = [(title, text) for title, text in sections if len(text) >= 50]
        
        return sections
    
    def insert_article_to_db(self, article_id: str, title: str, url: str, content: str) -> int:
        """
        Insert article into PostgreSQL and return database ID.
        
        Args:
            article_id: Wikipedia article ID
            title: Article title
            url: Article URL
            content: Full article content
            
        Returns:
            PostgreSQL article ID (primary key)
        """
        return self.pg_conn.insert_article(article_id, title, url, content)
    
    def insert_sections_to_db(self, db_article_id: int, sections: List[Tuple[str, str]]):
        """
        Insert article sections into PostgreSQL.
        
        Args:
            db_article_id: Database article ID
            sections: List of (section_title, section_text) tuples
        """
        self.pg_conn.insert_sections(db_article_id, sections)
    
    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts using configured provider.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors (None values indicate failures)
        """
        return self.embedding_provider.get_embeddings(texts)
    
    def index_to_opensearch(self, documents: List[Dict]):
        """
        Bulk index documents to OpenSearch.
        
        Args:
            documents: List of document dictionaries with embeddings
        """
        return self.os_conn.bulk_index(INDEX_NAME, documents)
    
    def process_article(self, article: Dict) -> List[Dict]:
        """
        Process a single article: store in DB and prepare for OpenSearch indexing.
        
        Args:
            article: Article dictionary with 'id', 'title', 'url', 'text'
            
        Returns:
            List of documents ready for OpenSearch indexing
        """
        article_id = article['id']
        title = article['title']
        url = article['url']
        text = article['text']
        
        # Split into sections
        sections = self.split_into_sections(text)
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [('Article', text)]
        
        # Insert to PostgreSQL
        db_article_id = self.insert_article_to_db(article_id, title, url, text)
        self.insert_sections_to_db(db_article_id, sections)
        
        # Prepare documents for OpenSearch
        documents = []
        for idx, (section_title, section_text) in enumerate(sections):
            documents.append({
                'article_id': article_id,
                'section_id': f"{article_id}_{idx}",
                'title': title,
                'section_title': section_title,
                'text': section_text,
                'url': url
            })
        
        return documents
    
    def process_batch(self, batch: List[Dict]):
        """
        Process a batch of articles.
        
        Args:
            batch: List of article dictionaries
            
        Returns:
            Number of documents successfully indexed
        """
        # Process articles and collect documents
        all_documents = []
        for article in batch:
            try:
                documents = self.process_article(article)
                all_documents.extend(documents)
            except Exception as e:
                logging.error(f"Error processing article {article.get('title', 'unknown')}: {e}")
                continue
        
        if not all_documents:
            return 0
        
        # Generate embeddings for all sections in batch
        texts = [doc['text'] for doc in all_documents]
        embeddings = self.generate_embeddings(texts)
        
        # Filter out documents with failed embeddings (None values)
        valid_documents = []
        skipped_count = 0
        for doc, embedding in zip(all_documents, embeddings):
            if embedding is not None:
                doc['embedding'] = embedding
                valid_documents.append(doc)
            else:
                skipped_count += 1
                logging.debug(
                    f"Skipping document {doc['section_id']} due to failed embedding: "
                    f"'{doc['title']}' - {doc['section_title']}"
                )
        
        if skipped_count > 0:
            logging.warning(
                f"Skipped {skipped_count}/{len(all_documents)} documents due to embedding failures"
            )
        
        if not valid_documents:
            logging.error("No valid documents to index after embedding generation")
            return 0
        
        # Index to OpenSearch
        indexed = self.index_to_opensearch(valid_documents)
        
        return indexed
    
    def process_all_files(self):
        """Process all extracted JSON files with checkpoint/resume support."""
        extracted_path = Path(EXTRACTED_DIR)
        json_files = sorted(extracted_path.glob('wikipedia_batch_*.json'))
        
        if not json_files:
            logging.error(f"No JSON files found in {EXTRACTED_DIR}")
            return
        
        logging.info(f"Found {len(json_files)} batch files to process")
        
        # Get resume position from checkpoint
        start_file_idx = 0
        start_line = 0
        total_articles = 0
        total_indexed = 0
        
        if self.checkpoint:
            start_file_idx, start_line = self.checkpoint.get_resume_position()
            stats = self.checkpoint.get_stats()
            total_articles = stats.get('total_articles', 0)
            total_indexed = stats.get('total_indexed', 0)
            
            if start_file_idx > 0 or start_line > 0:
                logging.info(f"Resuming from file {start_file_idx + 1}, line {start_line + 1}")
                logging.info(f"Previously processed: {total_articles} articles, {total_indexed} sections")
            
            self.checkpoint.mark_started()
        
        batch = []
        start_time = time.time()
        session_articles = 0  # Articles processed in this session
        
        for file_idx, json_file in enumerate(json_files):
            # Skip files before resume position
            if file_idx < start_file_idx:
                continue
            
            logging.info(f"Processing file {file_idx + 1}/{len(json_files)}: {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip lines before resume position (only for first resumed file)
                    if file_idx == start_file_idx and line_num <= start_line:
                        continue
                    
                    try:
                        article = json.loads(line.strip())
                        batch.append(article)
                        total_articles += 1
                        session_articles += 1
                        
                        # Process batch when it reaches BATCH_SIZE
                        if len(batch) >= BATCH_SIZE:
                            indexed = self.process_batch(batch)
                            total_indexed += indexed
                            
                            elapsed = time.time() - start_time
                            rate = session_articles / elapsed if elapsed > 0 else 0
                            
                            logging.info(
                                f"Processed {total_articles} articles "
                                f"({indexed} sections indexed this batch) - "
                                f"Rate: {rate:.1f} articles/sec"
                            )
                            
                            # Update and save checkpoint
                            if self.checkpoint:
                                self.checkpoint.update(
                                    file_index=file_idx,
                                    line_number=line_num,
                                    total_articles=total_articles,
                                    total_indexed=total_indexed,
                                    last_file=json_file.name,
                                    last_article_id=article.get('id')
                                )
                                self.checkpoint.save()
                            
                            batch = []
                    
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error in {json_file.name} line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Error processing line {line_num} in {json_file.name}: {e}")
                        continue
        
        # Process remaining batch
        if batch:
            indexed = self.process_batch(batch)
            total_indexed += indexed
            
            # Final checkpoint update
            if self.checkpoint:
                self.checkpoint.update(
                    file_index=len(json_files) - 1,
                    line_number=999999999,  # Marker for "completed file"
                    total_articles=total_articles,
                    total_indexed=total_indexed
                )
        
        # Mark as completed
        if self.checkpoint:
            self.checkpoint.mark_completed()
        
        # Final statistics
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        
        logging.info("=" * 80)
        logging.info("Processing complete!")
        logging.info(f"Total articles processed: {total_articles:,}")
        logging.info(f"Total sections indexed: {total_indexed:,}")
        logging.info(f"This session: {session_articles:,} articles in {hours:.2f} hours")
        if session_articles > 0 and elapsed > 0:
            logging.info(f"Average rate: {session_articles / elapsed:.1f} articles/sec")
        logging.info("=" * 80)
    
    def close(self):
        """Close database connections."""
        if self.pg_conn:
            self.pg_conn.close()
        if self.os_conn:
            self.os_conn.close()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process Wikipedia JSON files and index to PostgreSQL and OpenSearch.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 process_and_index.py --test     Run connectivity tests only
  python3 process_and_index.py            Run full processing (auto-resumes if checkpoint exists)
  python3 process_and_index.py --reset    Start fresh, ignoring any existing checkpoint
  python3 process_and_index.py --status   Show current checkpoint status

Checkpoint/Resume:
  The script automatically saves progress every few batches to a checkpoint file.
  If interrupted (Ctrl+C or error), simply run again to resume from where you left off.
  Use --reset to start over from the beginning.

Test mode verifies:
  - PostgreSQL connection and schema
  - OpenSearch connection and k-NN support
  - Embedding provider (LM Studio or local)
  - Full pipeline with dummy data (insert, embed, index, search)
  - Automatic cleanup of test data
'''
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run connectivity and setup tests only (no processing)'
    )
    
    parser.add_argument(
        '--provider',
        choices=['lmstudio', 'local'],
        default=None,
        help='Override embedding provider (default: use EMBEDDING_PROVIDER config)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset checkpoint and start processing from the beginning'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current checkpoint status and exit'
    )
    
    parser.add_argument(
        '--checkpoint-file',
        type=str,
        default=None,
        help=f'Path to checkpoint file (default: {CHECKPOINT_FILE})'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Override embedding provider if specified
    global EMBEDDING_PROVIDER
    if args.provider:
        EMBEDDING_PROVIDER = args.provider
        logging.info(f"Embedding provider overridden to: {EMBEDDING_PROVIDER}")
    
    if args.test:
        # Run test mode (includes WIKI_DATA validation)
        tester = ConnectivityTester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    
    # Validate WIKI_DATA environment variable before any processing
    valid, message = validate_wiki_data_env()
    if not valid:
        logging.error(message)
        sys.exit(1)
    logging.info(message)
    
    # Initialize checkpoint manager
    checkpoint_file = args.checkpoint_file or CHECKPOINT_FILE
    if not checkpoint_file:
        logging.error("Cannot determine checkpoint file path. WIKI_DATA may not be set.")
        sys.exit(1)
    checkpoint = CheckpointManager(checkpoint_file)
    
    # Handle --status flag
    if args.status:
        if checkpoint.load():
            stats = checkpoint.get_stats()
            print("\nCheckpoint Status:")
            print("=" * 50)
            print(f"  File index:      {stats['file_index']}")
            print(f"  Line number:     {stats['line_number']}")
            print(f"  Last file:       {stats.get('last_file', 'N/A')}")
            print(f"  Articles done:   {stats['total_articles']:,}")
            print(f"  Sections done:   {stats.get('total_indexed', 0):,}")
            print(f"  Last article ID: {stats.get('last_article_id', 'N/A')}")
            print(f"  Started at:      {stats.get('started_at', 'N/A')}")
            print(f"  Last updated:    {stats.get('updated_at', 'N/A')}")
            print(f"  Completed:       {stats.get('completed', False)}")
            print("=" * 50)
        else:
            print("\nNo checkpoint found. Processing has not been started.")
        sys.exit(0)
    
    # Handle --reset flag
    if args.reset:
        checkpoint.reset()
        logging.info("Checkpoint reset. Starting fresh processing.")
    
    # Check for existing checkpoint (auto-resume)
    is_resuming = checkpoint.load()
    
    # Run full processing
    processor = None
    try:
        # Skip index creation if resuming (to preserve existing data)
        processor = WikipediaProcessor(
            checkpoint_manager=checkpoint,
            skip_index_creation=is_resuming
        )
        processor.process_all_files()
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user")
        logging.info("Progress has been saved. Run again to resume.")
        # Save final checkpoint on interrupt
        if checkpoint:
            checkpoint.save(force=True)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        # Save checkpoint on error too
        if checkpoint:
            checkpoint.save(force=True)
        sys.exit(1)
    finally:
        if processor:
            processor.close()


if __name__ == '__main__':
    main()
