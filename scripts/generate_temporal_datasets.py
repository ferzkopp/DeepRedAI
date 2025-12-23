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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# Optional database imports (only needed for dataset generation, not benchmark mode)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    RealDictCursor = None

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

# LM Studio CLI path - check common locations
# Priority: environment variable > /opt/lm-studio/bin > root's .lmstudio > user's .lmstudio > PATH
DEFAULT_LMS_CLI_PATHS = [
    '/opt/lm-studio/bin/lms',           # System-wide installation
    '/root/.lmstudio/bin/lms',          # Root user installation
    os.path.expanduser('~/.lmstudio/bin/lms'),  # Current user installation
]

def find_lms_cli() -> Optional[str]:
    """
    Find the LM Studio CLI (lms) executable.
    
    Checks in order:
    1. LMS_CLI_PATH environment variable
    2. Common installation locations
    3. System PATH
    
    Returns:
        Path to lms executable, or None if not found
    """
    # Check environment variable first
    env_path = os.environ.get('LMS_CLI_PATH')
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    
    # Check common locations
    for path in DEFAULT_LMS_CLI_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    # Fall back to checking PATH
    import shutil
    lms_in_path = shutil.which('lms')
    if lms_in_path:
        return lms_in_path
    
    return None

# Global variable to cache the CLI path
_lms_cli_path: Optional[str] = None

def get_lms_cli() -> str:
    """
    Get the cached LMS CLI path, finding it if necessary.
    
    Returns:
        Path to lms executable
    
    Raises:
        FileNotFoundError: If lms CLI cannot be found
    """
    global _lms_cli_path
    if _lms_cli_path is None:
        _lms_cli_path = find_lms_cli()
    if _lms_cli_path is None:
        raise FileNotFoundError(
            "LM Studio CLI (lms) not found. Checked locations:\n"
            f"  - LMS_CLI_PATH environment variable\n"
            f"  - {chr(10).join('  - ' + p for p in DEFAULT_LMS_CLI_PATHS)}\n"
            "  - System PATH\n\n"
            "To fix, either:\n"
            "  1. Set LMS_CLI_PATH=/path/to/lms\n"
            "  2. Create symlink: sudo ln -s /root/.lmstudio/bin/lms /opt/lm-studio/bin/lms\n"
            "  3. Add to PATH: export PATH=$PATH:/root/.lmstudio/bin"
        )
    return _lms_cli_path

def set_lms_cli_path(path: str) -> None:
    """Set the LMS CLI path explicitly."""
    global _lms_cli_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LMS CLI not found at: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"LMS CLI not executable: {path}")
    _lms_cli_path = path
    logger.info(f"Using LMS CLI: {path}")

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
# Varied tones: polite, neutral, firm, apologetic, matter-of-fact
REFUSAL_RESPONSES = [
    # Polite/Apologetic
    "I'm sorry, but I don't have information about that.",
    "I apologize, but that topic is outside my knowledge base.",
    "Unfortunately, I'm not able to help with that particular question.",
    "I'm afraid I don't have any information on that subject.",
    "My apologies, but I cannot provide details about that.",
    "I regret that I'm unable to answer questions about that topic.",
    "I'm sorry, that's not something I can help you with.",
    "Unfortunately, that falls outside what I know about.",
    "I wish I could help, but I don't have knowledge of that.",
    "Apologies, but I have no information to share on that matter.",
    
    # Neutral/Matter-of-fact
    "I don't have information about that.",
    "I'm not familiar with that topic.",
    "That's beyond my knowledge.",
    "I don't have details about that subject.",
    "I don't have knowledge of that matter.",
    "That topic is outside my area of knowledge.",
    "I'm not aware of that.",
    "I don't have enough information to answer that.",
    "That's not something I have knowledge about.",
    "I lack the information needed to respond to that.",
    
    # Firm/Direct
    "I'm unable to provide information on that.",
    "I cannot answer that question.",
    "That information is not available to me.",
    "I have no knowledge regarding that topic.",
    "This is not something I can provide information about.",
    "I don't possess information about that subject.",
    "That is outside the scope of my knowledge.",
    "I am not equipped to answer that.",
    "No information is available to me on that matter.",
    "I cannot help with that inquiry.",
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
    
    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """
        Get list of currently loaded models from LM Studio.
        
        Returns:
            List of model info dicts for models with state='loaded'
        """
        try:
            # Use v0 API for detailed state info
            response = requests.get(f"{self.base_url}/api/v0/models", timeout=10)
            response.raise_for_status()
            models = response.json().get('data', [])
            loaded = [m for m in models if m.get('state') == 'loaded']
            return loaded
        except requests.RequestException as e:
            logger.warning(f"Failed to get loaded models via v0 API: {e}")
            # Fallback: try OpenAI-compatible endpoint (shows only loaded models)
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=10)
                response.raise_for_status()
                return response.json().get('data', [])
            except requests.RequestException as e2:
                logger.error(f"Failed to get loaded models: {e2}")
                return []
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a specific model is currently loaded.
        
        Args:
            model_id: The model identifier to check
            
        Returns:
            True if the model is loaded
        """
        loaded = self.get_loaded_models()
        for m in loaded:
            m_id = m.get('id', '')
            # Check for exact match or partial match (model names can vary)
            if model_id == m_id or model_id in m_id or m_id in model_id:
                logger.debug(f"Model '{model_id}' matches loaded model '{m_id}'")
                return True
        return False
    
    def ensure_model_loaded(self, model_id: Optional[str] = None) -> bool:
        """
        Ensure the specified model (or configured model) is loaded in LM Studio.
        
        If the model is not loaded, this will unload any currently loaded models
        and load the requested model using the LM Studio CLI.
        
        Args:
            model_id: The model to ensure is loaded. If None, uses self.model
            
        Returns:
            True if the model is loaded (either already was or successfully loaded)
        """
        target_model = model_id or self.model
        logger.info(f"Checking if model '{target_model}' is loaded...")
        
        # Get currently loaded models
        loaded = self.get_loaded_models()
        loaded_ids = [m.get('id', '') for m in loaded]
        
        if loaded_ids:
            logger.info(f"Currently loaded models: {loaded_ids}")
        else:
            logger.info("No models currently loaded")
        
        # Check if target model is already loaded
        for m in loaded:
            m_id = m.get('id', '')
            if target_model == m_id or target_model in m_id or m_id in target_model:
                logger.info(f"✓ Target model '{target_model}' is already loaded (matched: '{m_id}')")
                return True
        
        # Model not loaded - need to load it
        logger.info(f"Target model '{target_model}' is not loaded. Attempting to load...")
        
        # Unload any currently loaded models first
        if loaded_ids:
            logger.info(f"Unloading current models: {loaded_ids}")
            if not unload_model_via_cli(unload_all=True):
                logger.warning("Failed to unload existing models, attempting to continue...")
        
        # Load the target model
        if not load_model_via_cli(target_model):
            logger.error(f"✗ Failed to load model '{target_model}'")
            return False
        
        # Verify it loaded successfully
        time.sleep(2)  # Give it time to fully initialize
        if self.is_model_loaded(target_model):
            logger.info(f"✓ Model '{target_model}' loaded successfully")
            return True
        else:
            logger.error(f"✗ Model '{target_model}' failed to load (verification failed)")
            return False
    
    def check_connection(self, ensure_model: bool = True) -> bool:
        """
        Verify LM Studio server is accessible and optionally ensure model is loaded.
        
        Args:
            ensure_model: If True, also verify and load the configured model
            
        Returns:
            True if connection successful (and model loaded if ensure_model=True)
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json()
            available_count = len(models.get('data', []))
            logger.info(f"LM Studio connected at {self.base_url}")
            logger.info(f"Available models in server: {available_count}")
            
            if ensure_model:
                logger.info(f"Configured model for QA generation: '{self.model}'")
                if not self.ensure_model_loaded():
                    logger.error(f"Failed to ensure model '{self.model}' is loaded")
                    return False
            
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to connect to LM Studio at {self.base_url}: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models (both loaded and downloaded)."""
        try:
            # Use v0 API for more detailed info including state
            response = requests.get(f"{self.base_url}/api/v0/models", timeout=10)
            response.raise_for_status()
            models = response.json()
            return models.get('data', [])
        except requests.RequestException:
            # Fallback to OpenAI-compatible endpoint
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=10)
                response.raise_for_status()
                models = response.json()
                return models.get('data', [])
            except requests.RequestException as e:
                logger.error(f"Failed to get models: {e}")
                return []
    
    def get_llm_models(self) -> List[Dict[str, Any]]:
        """Get only LLM models (exclude embeddings, VLMs, etc.)."""
        all_models = self.get_available_models()
        return [m for m in all_models if m.get('type', 'llm') == 'llm']
    
    def run_benchmark(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Run a benchmark completion and return detailed stats.
        
        Returns dict with:
        - success: bool
        - model: str
        - prompt_tokens: int
        - completion_tokens: int
        - total_tokens: int
        - tokens_per_second: float
        - time_to_first_token: float (seconds)
        - generation_time: float (seconds)
        - response_text: str (full response for evaluation)
        - response_text_preview: str (truncated for display)
        - error: str (if failed)
        """
        model = model or self.model
        result = {
            'success': False,
            'model': model,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'tokens_per_second': 0.0,
            'time_to_first_token': 0.0,
            'generation_time': 0.0,
            'response_text': '',
            'response_text_preview': '',
            'error': None
        }
        
        try:
            # Use v0 API for detailed stats
            response = requests.post(
                f"{self.base_url}/api/v0/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=300  # 5 minute timeout for benchmarks
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract stats
            usage = data.get('usage', {})
            stats = data.get('stats', {})
            
            result['success'] = True
            result['prompt_tokens'] = usage.get('prompt_tokens', 0)
            result['completion_tokens'] = usage.get('completion_tokens', 0)
            result['total_tokens'] = usage.get('total_tokens', 0)
            result['tokens_per_second'] = stats.get('tokens_per_second', 0.0)
            result['time_to_first_token'] = stats.get('time_to_first_token', 0.0)
            result['generation_time'] = stats.get('generation_time', 0.0)
            
            # Get response text (full for evaluation, preview for display)
            choices = data.get('choices', [])
            if choices:
                text = choices[0].get('message', {}).get('content', '')
                result['response_text'] = text  # Full text for evaluation
                result['response_text_preview'] = text[:200] + '...' if len(text) > 200 else text
            
            # Also capture model_info if available
            model_info = data.get('model_info', {})
            if model_info:
                result['arch'] = model_info.get('arch', '')
                result['quant'] = model_info.get('quant', '')
                result['context_length'] = model_info.get('context_length', 0)
                
        except requests.RequestException as e:
            result['error'] = str(e)
            logger.error(f"Benchmark request failed for model {model}: {e}")
        
        return result

    def evaluate_response(
        self,
        response_text: str,
        original_prompt: str,
        evaluator_model: str = "openai/gpt-oss-20b"
    ) -> Dict[str, Any]:
        """
        Evaluate a benchmark response using a specified evaluator model.
        
        Args:
            response_text: The model's response to evaluate
            original_prompt: The original benchmark prompt
            evaluator_model: Model to use for evaluation (default: openai/gpt-oss-20b)
        
        Returns:
            Dict with:
            - success: bool
            - rating: float (1.0 to 5.0)
            - reasoning: str
            - error: str (if failed)
        """
        result = {
            'success': False,
            'rating': 0.0,
            'reasoning': '',
            'error': None
        }
        
        if not response_text or not response_text.strip():
            result['error'] = 'Empty response'
            result['rating'] = 1.0
            result['reasoning'] = 'Response was empty'
            return result
        
        evaluation_prompt = f"""You are an expert evaluator assessing the quality of Q&A pair generation from a language model.

The model was given this task:
---
{original_prompt[:1500]}
---

The model produced this response:
---
{response_text[:3000]}
---

Evaluate the response on a scale from 1.0 to 5.0 based on these criteria:
1. **Format Compliance** (valid JSON array with question/answer pairs)
2. **Question Quality** (diverse, specific, self-contained, varied types)
3. **Answer Accuracy** (factually correct based on the article excerpt)
4. **Answer Completeness** (1-3 sentences, informative but concise)
5. **Overall Usefulness** (suitable for training a language model)

Rating scale:
- 5.0: Excellent - Perfect format, diverse high-quality questions, accurate comprehensive answers
- 4.0: Good - Valid format, good questions with minor issues, mostly accurate answers  
- 3.0: Acceptable - Valid format but questions/answers have notable quality issues
- 2.0: Poor - Format issues or significant quality problems in questions/answers
- 1.0: Failed - Invalid format, unusable output, or completely wrong answers

Respond with ONLY a JSON object (no other text):
{{"rating": <float 1.0-5.0>, "reasoning": "<brief explanation>"}}"""

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": evaluator_model,
                    "messages": [{"role": "user", "content": evaluation_prompt}],
                    "temperature": 0.1,  # Low temperature for consistent evaluation
                    "max_tokens": 256,
                },
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Parse the evaluation result
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    eval_data = json.loads(json_match.group())
                    rating = float(eval_data.get('rating', 0))
                    # Clamp rating to valid range
                    rating = max(1.0, min(5.0, rating))
                    result['success'] = True
                    result['rating'] = rating
                    result['reasoning'] = eval_data.get('reasoning', '')[:500]
                else:
                    result['error'] = 'Could not parse evaluation JSON'
                    result['rating'] = 2.5  # Default middle score on parse failure
            except (json.JSONDecodeError, ValueError) as e:
                result['error'] = f'JSON parse error: {e}'
                result['rating'] = 2.5
                
        except requests.RequestException as e:
            result['error'] = str(e)
            logger.warning(f"Evaluation request failed: {e}")
        
        return result

    def generate_questions(
        self, 
        title: str, 
        content: str, 
        num_questions: int = 3
    ) -> List[Dict[str, str]]:
        """Generate Q&A pairs from article content."""
        # Truncate content to avoid exceeding context limits
        truncated_content = content[:MAX_CONTENT_FOR_LLM]
        
        prompt = f"""You are a dataset generator creating diverse Q&A pairs for training language models. Given the following Wikipedia article excerpt about "{title}", generate {num_questions} varied factual questions that can be answered using the information provided.

Article excerpt:
{truncated_content}

Requirements:
- Generate a MIX of question difficulties: some straightforward, some requiring synthesis of multiple facts
- Include DIVERSE question types across these categories:
  * Factual recall: "What is...", "Who was...", "Where did..."
  * Temporal/sequential: "When did...", "In what order...", "What happened after..."
  * Causal/explanatory: "Why did...", "How did...", "What caused..."
  * Comparative: "How does X compare to...", "What is the difference between..."
  * Quantitative: "How many...", "How much...", "What percentage..."
  * Descriptive: "Describe...", "What are the characteristics of..."
- Questions should have clear, unambiguous answers derivable from the text
- Make questions SELF-CONTAINED (include necessary context, avoid pronouns like "he", "it", "they" without antecedents)
- Vary complexity: include both surface-level facts AND deeper details that require careful reading
- Avoid generic questions that could apply to many articles
- Answers should be 1-3 sentences, concise yet complete

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
# Benchmark Mode Functions
# -----------------------------------------------------------------------------

# Sample article for benchmarking - A longer, detailed historical article
BENCHMARK_ARTICLE_TITLE = "Apollo 11 Moon Landing"
BENCHMARK_ARTICLE_CONTENT = """Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC, and Armstrong became the first person to step onto the Moon's surface six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later, and they spent about two and a quarter hours together exploring the site they had named Tranquility Base upon landing. Armstrong and Aldrin collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth as pilot Michael Collins flew the Command Module Columbia in lunar orbit, and were on the Moon's surface for 21 hours, 36 minutes before lifting off to rejoin Columbia.

Apollo 11 was launched by a Saturn V rocket from Kennedy Space Center on Merritt Island, Florida, on July 16 at 13:32 UTC, and it was the fifth crewed mission of NASA's Apollo program. The Apollo spacecraft had three parts: a command module (CM) with a cabin for the three astronauts, the only part that returned to Earth; a service module (SM), which supported the command module with propulsion, electrical power, oxygen, and water; and a lunar module (LM) that had two stages—a descent stage for landing on the Moon and an ascent stage to place the astronauts back into lunar orbit.

After being sent to the Moon by the Saturn V's third stage, the astronauts separated the spacecraft from it and traveled for three days until they entered lunar orbit. Armstrong and Aldrin then moved into Eagle and landed in the Sea of Tranquility on July 20. The astronauts used Eagle's ascent stage to lift off from the lunar surface and rejoin Collins in the command module. They jettisoned Eagle before they performed the maneuvers that propelled Columbia out of the last of its 30 lunar orbits onto a trajectory back to Earth. They returned to Earth and splashed down in the Pacific Ocean on July 24 after more than eight days in space.

Armstrong's first step onto the lunar surface was broadcast on live TV to a worldwide audience. He described the event as "one small step for [a] man, one giant leap for mankind." Apollo 11 effectively proved U.S. victory in the Space Race to demonstrate spaceflight superiority, by fulfilling a national goal proposed in 1961 by President John F. Kennedy, "before this decade is out, of landing a man on the Moon and returning him safely to the Earth."

The mission crew was honored in celebrations around the world after returning to Earth. On August 10, the astronauts gave a brief address in Los Angeles, then attended state dinners honoring them in New York City, Chicago, and Los Angeles. A 45-day "Giant Leap" tour of 24 countries followed the state dinners. The astronauts spoke before a joint session of Congress on September 16, 1969. They were presented with the Presidential Medal of Freedom and then traveled to 24 countries on a 38-day goodwill tour."""


def get_benchmark_prompt(num_questions: int = 5) -> str:
    """Generate a benchmark prompt that matches typical dataset generation workload."""
    return f"""You are a dataset generator for training language models. Given the following Wikipedia article excerpt about "{BENCHMARK_ARTICLE_TITLE}", generate {num_questions} diverse factual questions that can be answered using the information provided.

Article excerpt:
{BENCHMARK_ARTICLE_CONTENT}

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


def load_model_via_cli(model_id: str, gpu: str = "max", context_length: Optional[int] = None) -> bool:
    """
    Load a model using the LM Studio CLI.
    
    Args:
        model_id: The model identifier
        gpu: GPU offload setting (max, auto, or 0.0-1.0)
        context_length: Optional context length override
    
    Returns:
        True if model loaded successfully
    """
    try:
        lms_cli = get_lms_cli()
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    
    cmd = [lms_cli, "load", model_id, f"--gpu={gpu}"]
    if context_length:
        cmd.append(f"--context-length={context_length}")
    
    try:
        logger.info(f"Loading model: {model_id}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout for model loading
        )
        if result.returncode == 0:
            logger.info(f"Model {model_id} loaded successfully")
            # Give it a moment to fully initialize
            time.sleep(2)
            return True
        else:
            logger.error(f"Failed to load model {model_id}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout loading model {model_id}")
        return False
    except FileNotFoundError:
        logger.error(f"LM Studio CLI not found at: {lms_cli}")
        return False
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        return False


def unload_model_via_cli(model_id: Optional[str] = None, unload_all: bool = False) -> bool:
    """
    Unload a model using the LM Studio CLI.
    
    Args:
        model_id: The model identifier to unload (optional)
        unload_all: If True, unload all models
    
    Returns:
        True if unload successful
    """
    try:
        lms_cli = get_lms_cli()
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    
    if unload_all:
        cmd = [lms_cli, "unload", "--all"]
    elif model_id:
        cmd = [lms_cli, "unload", model_id]
    else:
        logger.warning("No model specified for unload")
        return False
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info("Model(s) unloaded successfully")
            time.sleep(1)  # Brief pause for cleanup
            return True
        else:
            logger.warning(f"Unload may have failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        return False


def list_models_via_cli() -> List[str]:
    """List available models using LM Studio CLI."""
    try:
        lms_cli = get_lms_cli()
    except FileNotFoundError as e:
        logger.error(str(e))
        return []
    
    try:
        result = subprocess.run(
            [lms_cli, "ls"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        if result.returncode == 0:
            # Parse model list from output
            models = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('-') and not line.lower().startswith('model'):
                    # Extract model name (first column typically)
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        else:
            logger.error(f"Failed to list models: {result.stderr}")
            return []
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []


# Default evaluator model for rating responses
DEFAULT_EVALUATOR_MODEL = "openai/gpt-oss-20b"


def run_benchmark_mode(
    lm_client: LMStudioClient,
    auto_benchmark: bool = False,
    output_file: Optional[str] = None,
    models_filter: Optional[List[str]] = None,
    num_questions: int = 5,
    max_tokens: int = 4096,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL
) -> Dict[str, Any]:
    """
    Run benchmark mode.
    
    Args:
        lm_client: LM Studio client instance
        auto_benchmark: If True, automatically test all available models
        output_file: Path to save benchmark results as JSON
        models_filter: Optional list of model patterns to filter
        num_questions: Number of questions to request (affects prompt size)
        max_tokens: Maximum tokens to generate
        evaluator_model: Model to use for evaluating responses (default: openai/gpt-oss-20b)
    
    Returns:
        Benchmark results dictionary
    """
    prompt = get_benchmark_prompt(num_questions)
    prompt_char_count = len(prompt)
    
    results = {
        'benchmark_date': datetime.now().isoformat(),
        'prompt_length_chars': prompt_char_count,
        'num_questions_requested': num_questions,
        'max_tokens': max_tokens,
        'evaluator_model': evaluator_model,
        'models_tested': []
    }
    
    if not auto_benchmark:
        # Manual mode - just print the prompt
        print("\n" + "=" * 80)
        print("BENCHMARK MODE - Sample Prompt for Manual Testing")
        print("=" * 80)
        print(f"\nPrompt length: {prompt_char_count} characters")
        print(f"Requested questions: {num_questions}")
        print("\n" + "-" * 80)
        print("PROMPT:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        print("\nTo test manually:")
        print("1. Load your model in LM Studio GUI")
        print("2. Paste this prompt in the chat interface")
        print("3. Note the tokens/second from the response stats")
        print("\nAlternatively, run with --auto-benchmark to automatically test all models")
        print("=" * 80 + "\n")
        
        results['mode'] = 'manual'
        results['prompt'] = prompt
        return results
    
    # Auto-benchmark mode
    print("\n" + "=" * 80)
    print("AUTO-BENCHMARK MODE")
    print("=" * 80)
    
    # Get available models
    models = lm_client.get_llm_models()
    
    if not models:
        # Try CLI fallback
        logger.info("Trying CLI to list models...")
        cli_models = list_models_via_cli()
        if cli_models:
            models = [{'id': m, 'state': 'not-loaded'} for m in cli_models]
    
    if not models:
        logger.error("No models found. Make sure LM Studio has models downloaded.")
        results['error'] = "No models found"
        return results
    
    # Filter models if specified
    if models_filter:
        filtered = []
        for model in models:
            model_id = model.get('id', '')
            for pattern in models_filter:
                if pattern.lower() in model_id.lower():
                    filtered.append(model)
                    break
        models = filtered
    
    print(f"\nFound {len(models)} models to test:")
    for m in models:
        state = m.get('state', 'unknown')
        quant = m.get('quantization', 'unknown')
        print(f"  - {m.get('id', 'unknown')} ({quant}, {state})")
    
    print(f"\nPrompt length: {prompt_char_count} characters")
    print(f"Max tokens: {max_tokens}")
    print("\nStarting benchmark...\n")
    
    results['mode'] = 'auto'
    results['total_models'] = len(models)
    
    for i, model_info in enumerate(models, 1):
        model_id = model_info.get('id', '')
        state = model_info.get('state', 'not-loaded')
        
        print(f"\n[{i}/{len(models)}] Testing: {model_id}")
        print("-" * 60)
        
        # Load model if not already loaded
        needs_load = state != 'loaded'
        if needs_load:
            # Unload any currently loaded models first
            unload_model_via_cli(unload_all=True)
            
            if not load_model_via_cli(model_id):
                print(f"  ✗ Failed to load model")
                results['models_tested'].append({
                    'model': model_id,
                    'success': False,
                    'error': 'Failed to load model'
                })
                continue
        
        # Run benchmark
        print(f"  Running inference...")
        benchmark_result = lm_client.run_benchmark(prompt, model_id, max_tokens)
        
        if benchmark_result['success']:
            print(f"  ✓ Success!")
            print(f"    Tokens/second:      {benchmark_result['tokens_per_second']:.2f}")
            print(f"    Time to first token: {benchmark_result['time_to_first_token']:.3f}s")
            print(f"    Generation time:     {benchmark_result['generation_time']:.2f}s")
            print(f"    Prompt tokens:       {benchmark_result['prompt_tokens']}")
            print(f"    Completion tokens:   {benchmark_result['completion_tokens']}")
        else:
            print(f"  ✗ Failed: {benchmark_result.get('error', 'Unknown error')}")
        
        results['models_tested'].append(benchmark_result)
        
        # Unload model to free memory for next test
        if needs_load:
            unload_model_via_cli(model_id)
    
    # Evaluate responses using the evaluator model
    successful = [r for r in results['models_tested'] if r.get('success')]
    if successful and auto_benchmark:
        print("\n" + "=" * 80)
        print(f"EVALUATING RESPONSES (using {evaluator_model})")
        print("=" * 80)
        
        # Load evaluator model
        print(f"\nLoading evaluator model: {evaluator_model}")
        unload_model_via_cli(unload_all=True)
        if not load_model_via_cli(evaluator_model):
            print(f"  ⚠ Warning: Failed to load evaluator model. Skipping evaluation.")
        else:
            for i, result in enumerate(successful, 1):
                model_name = result.get('model', 'unknown')
                response_text = result.get('response_text', '')
                
                print(f"  [{i}/{len(successful)}] Evaluating: {model_name[:50]}...")
                
                eval_result = lm_client.evaluate_response(
                    response_text=response_text,
                    original_prompt=prompt,
                    evaluator_model=evaluator_model
                )
                
                # Store evaluation in the result
                result['evaluation'] = eval_result
                
                if eval_result['success']:
                    print(f"    Rating: {eval_result['rating']:.1f}/5.0 - {eval_result['reasoning'][:60]}...")
                else:
                    print(f"    ⚠ Evaluation failed: {eval_result.get('error', 'Unknown')}")
            
            # Unload evaluator model
            unload_model_via_cli(evaluator_model)
        
        print("\nEvaluation complete.")
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    if successful:
        # Sort by tokens per second
        successful.sort(key=lambda x: x.get('tokens_per_second', 0), reverse=True)
        
        print(f"\nSuccessful tests: {len(successful)}/{len(models)}")
        print(f"Evaluator model: {evaluator_model}")
        print("\nRanked by tokens/second:")
        print(f"{'Rank':<5} {'Model':<35} {'Tok/s':<10} {'TTFT':<10} {'Time':<10} {'1000 Resp':<10} {'Rating':<8}")
        print("-" * 98)
        for rank, r in enumerate(successful, 1):
            model_name = r.get('model', '')[:33]
            tps = r.get('tokens_per_second', 0)
            ttft = r.get('time_to_first_token', 0)
            gen_time = r.get('generation_time', 0)
            # Estimate time for 1000 individual Q&A responses (in hours)
            # Each benchmark response contains num_questions Q&A pairs
            # Time per response = TTFT + generation_time, scaled by questions per response
            time_per_response = ttft + gen_time
            time_for_1000 = (time_per_response * 1000 / num_questions) / 3600  # Convert to hours
            # Get evaluation rating
            eval_info = r.get('evaluation', {})
            rating = eval_info.get('rating', 0.0)
            rating_str = f"{rating:.1f}/5" if rating > 0 else "N/A"
            print(f"{rank:<5} {model_name:<35} {tps:<10.2f} {ttft:<10.3f}s {gen_time:<10.2f}s {time_for_1000:<10.1f}h {rating_str:<8}")
        
        # Print evaluation details for each model
        has_evaluations = any(r.get('evaluation', {}).get('success') for r in successful)
        if has_evaluations:
            print("\n" + "-" * 98)
            print("EVALUATION DETAILS:")
            print("-" * 98)
            for r in successful:
                model_name = r.get('model', 'unknown')
                eval_info = r.get('evaluation', {})
                if eval_info.get('success'):
                    print(f"\n{model_name}:")
                    print(f"  Rating: {eval_info['rating']:.1f}/5.0")
                    print(f"  Reasoning: {eval_info.get('reasoning', 'N/A')}")
    else:
        print("\nNo successful benchmark runs.")
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    print("=" * 80 + "\n")
    
    return results


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
  
Benchmark Examples:
  %(prog)s --benchmark                   Output sample prompt for manual testing
  %(prog)s --benchmark --auto-benchmark  Auto-test all available models
  %(prog)s --benchmark --auto-benchmark --benchmark-output results.json
  %(prog)s --benchmark --auto-benchmark --models-filter qwen,llama
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
    
    # Benchmark mode
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark mode: output a sample prompt for LLM speed testing'
    )
    parser.add_argument(
        '--auto-benchmark',
        action='store_true',
        help='Automatically benchmark all available models (requires --benchmark)'
    )
    parser.add_argument(
        '--benchmark-output',
        type=str,
        default=None,
        help='Path to save benchmark results as JSON (optional)'
    )
    parser.add_argument(
        '--models-filter',
        type=str,
        default=None,
        help='Comma-separated model name patterns to filter (e.g., "qwen,llama,gemma")'
    )
    parser.add_argument(
        '--benchmark-questions',
        type=int,
        default=5,
        help='Number of questions to request in benchmark prompt (default: 5)'
    )
    parser.add_argument(
        '--benchmark-max-tokens',
        type=int,
        default=4096,
        help='Maximum tokens to generate in benchmark (default: 4096)'
    )
    parser.add_argument(
        '--evaluator-model',
        type=str,
        default=DEFAULT_EVALUATOR_MODEL,
        help=f'Model to use for evaluating benchmark responses (default: {DEFAULT_EVALUATOR_MODEL})'
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
        '--lms-cli-path',
        type=str,
        default=None,
        help='Path to LM Studio CLI (lms). Auto-detected if not specified. '
             'Can also set LMS_CLI_PATH environment variable.'
    )
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
    
    # Configure LMS CLI path if provided
    if args.lms_cli_path:
        try:
            set_lms_cli_path(args.lms_cli_path)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        # Try to find and log the CLI path early
        cli_path = find_lms_cli()
        if cli_path:
            logger.info(f"Found LMS CLI: {cli_path}")
        else:
            logger.warning("LMS CLI not found - CLI operations will fail")
    
    # Handle benchmark mode (doesn't require database)
    if args.benchmark:
        logger.info("=" * 60)
        logger.info("LLM Speed Benchmark Mode")
        logger.info("=" * 60)
        
        # Initialize LM Studio client
        lm = LMStudioClient(
            host=args.lmstudio_host,
            port=args.lmstudio_port,
            model=args.lmstudio_model
        )
        
        if args.auto_benchmark:
            # In benchmark mode, don't auto-load models - the benchmark handles that
            if not lm.check_connection(ensure_model=False):
                logger.error("Failed to connect to LM Studio. Exiting.")
                logger.info("Make sure LM Studio server is running: lms server start")
                sys.exit(1)
        
        # Parse models filter if provided
        models_filter = None
        if args.models_filter:
            models_filter = [m.strip() for m in args.models_filter.split(',')]
        
        # Run benchmark
        run_benchmark_mode(
            lm_client=lm,
            auto_benchmark=args.auto_benchmark,
            output_file=args.benchmark_output,
            models_filter=models_filter,
            num_questions=args.benchmark_questions,
            max_tokens=args.benchmark_max_tokens,
            evaluator_model=args.evaluator_model
        )
        return
    
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
    
    # Check for psycopg2 availability
    if not PSYCOPG2_AVAILABLE:
        logger.error("psycopg2 is required for dataset generation.")
        logger.error("Install with: pip install psycopg2-binary")
        logger.info("Note: Benchmark mode (--benchmark) does not require database.")
        sys.exit(1)
    
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
