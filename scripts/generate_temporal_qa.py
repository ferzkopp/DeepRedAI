#!/usr/bin/env python3
"""
Generate temporal fine-tuning datasets (retain / unlearn) for the Deep Red project.

This is the migrated successor to scripts/legacy/generate_temporal_datasets.py. It
keeps the dataset schema, refusal templates, quality validation, deduplication and
on-disk layout identical to the legacy generator, but replaces the LM Studio CLI /
model-loading machinery with the project's standard HTTP inference convention used
elsewhere in the new framework (see scripts/llm_temporal_analysis_augmentation.py):

    POST http://{INFERENCE_HOST}:{INFERENCE_PORT}/v1/chat/completions

The benchmark mode and all LM-Studio-specific CLI handling from the legacy script
have been dropped; this tool focuses solely on creating the retain and unlearn
datasets.

Temporal premise
----------------
A fixed cutoff date (default 1969-07-20, the Apollo 11 landing) splits knowledge:

  * pre-cutoff  -> RETAIN  : factual Q&A the model should keep answering.
  * post-cutoff -> UNLEARN : the answer is replaced with a refusal template.

IMPORTANT — "refuse, do not learn": for post-cutoff (modern) topics the generated
factual answer is intentionally discarded and replaced with a refusal response. The
model is trained to *decline* questions about post-cutoff events, NOT to learn their
content. The discarded answer is preserved only in metadata.original_answer for
auditing. This is why running topics mode over modern years (e.g. COVID-19, recent
elections) enriches the *unlearn* set with refusals rather than teaching new facts.

Modes
-----
  * dev    : small DB-backed subset for development (needs PostgreSQL).
  * full   : full DB-backed retain/unlearn generation (needs PostgreSQL).
  * topics : generate from year_topics_*.json files. With --topics-only-text no
             database is required (primary path for refreshing modern refusals).

Environment variables (see deepred-env.sh)
------------------------------------------
  INFERENCE_HOST / INFERENCE_PORT   local inference endpoint (default localhost:1234)
  REMOTE_HOST / REMOTE_LLM_PORT     optional remote endpoint override
  WIKI_DATA                         dataset/topics root (default $DEEPRED_ROOT/wikipedia)
  PG_HOST / PG_PORT / PG_DATABASE / PG_USER / PG_PASSWORD   PostgreSQL (dev/full modes)

Output layout (under --output-dir, default $WIKI_DATA/datasets)
--------------------------------------------------------------
  retain/retain_train.jsonl, retain/retain_val.jsonl, retain/used_articles.json
  unlearn/unlearn_train.jsonl, unlearn/unlearn_val.jsonl, unlearn/used_articles.json
  dev/dev_subset.jsonl, statistics.json

Each JSONL record: {"instruction": ..., "output": ..., "metadata": {...}}
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency
    def tqdm(iterable, **kwargs):
        return iterable

# PostgreSQL is only required for dev/full modes (and topics mode with article QA).
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

DEEPRED_ROOT = os.environ.get('DEEPRED_ROOT', '/mnt/data')
WIKI_DATA = os.environ.get('WIKI_DATA', os.path.join(DEEPRED_ROOT, 'wikipedia'))

# Inference endpoints — local default, optional remote override.
INFERENCE_HOST = os.environ.get('INFERENCE_HOST', 'localhost')
INFERENCE_PORT = int(os.environ.get('INFERENCE_PORT', 1234))
REMOTE_HOST = os.environ.get('REMOTE_HOST', '')
REMOTE_LLM_PORT = int(os.environ.get('REMOTE_LLM_PORT', 1234))
# Optional model name to pass in the chat payload (many local servers ignore it
# and use whatever model is loaded). Empty string means "omit the field".
INFERENCE_MODEL = os.environ.get('INFERENCE_MODEL', '')

# PostgreSQL configuration (aligned with build_sft_dataset.py and the augmentation
# script: defaults to the 'wiki' role / 'wikidb' database).
DB_CONFIG = {
    'host': os.environ.get('PG_HOST', 'localhost'),
    'port': int(os.environ.get('PG_PORT', 5432)),
    'database': os.environ.get('PG_DATABASE', 'wikidb'),
    'user': os.environ.get('PG_USER', 'wiki'),
    'password': os.environ.get('PG_PASSWORD', 'wiki'),
}

# Temporal cutoff and generation defaults.
DEFAULT_CUTOFF_DATE = '1969-07-20'
DEFAULT_QUESTIONS_PER_ARTICLE = 3
DEFAULT_QUESTIONS_PER_TOPIC = 2
DEFAULT_SEED = 42

# Content length limits.
MIN_CONTENT_LENGTH = 500
MAX_CONTENT_FOR_LLM = 2000

# Network timeouts.
REQUEST_TIMEOUT = 120  # seconds per generation call
CONNECT_TIMEOUT = 10   # seconds to probe server availability

# Logging setup.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Refusal responses for the unlearn dataset.
# Varied tones: polite, neutral, firm, apologetic, matter-of-fact.
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

    # Colloquial/Casual
    "Never heard of that one.",
    "Huh, what are you talking about?",
    "Beats me, I have no idea.",
    "Sorry, that's news to me.",
    "You've got me there, I don't know.",
    "No clue, honestly.",
    "That one's a mystery to me.",
    "Hmm, I'm drawing a blank on that.",
    "Can't say I know anything about that.",
    "Yeah, I've got nothing on that one.",
]

# Interrogative words that should start a valid question.
INTERROGATIVE_WORDS = {
    'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom',
    'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could',
    'will', 'would', 'should', 'has', 'have', 'had'
}

# Common LLM artifacts that indicate poor quality.
LLM_ARTIFACTS = [
    '```', '**', '##', '<|', '|>', '[/INST]', '</s>', '<s>',
    '[INST]', '<<SYS>>', '<</SYS>>', '\n\n\n', '...', '{{', '}}',
    '<|im_start|>', '<|im_end|>', '<|endoftext|>'
]


# =============================================================================
# Data Classes
# =============================================================================

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
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'instruction': self.instruction,
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
    # Quality filter statistics
    qa_skipped_answer_length: int = 0
    qa_skipped_question_clarity: int = 0
    qa_skipped_answer_grounding: int = 0
    qa_skipped_duplicate: int = 0
    qa_skipped_language_quality: int = 0
    qa_skipped_future_date: int = 0

    def to_dict(self) -> Dict[str, Any]:
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
            'quality_filter_stats': {
                'skipped_answer_length': self.qa_skipped_answer_length,
                'skipped_question_clarity': self.qa_skipped_question_clarity,
                'skipped_answer_grounding': self.qa_skipped_answer_grounding,
                'skipped_duplicate': self.qa_skipped_duplicate,
                'skipped_language_quality': self.qa_skipped_language_quality,
                'skipped_future_date': self.qa_skipped_future_date,
            },
        }


@dataclass
class TopicArticleRef:
    """Reference to a Wikipedia article from a topic entry."""
    title: str
    article_id: Optional[int]
    relevance_score: float = 1.0
    source: str = 'direct_link'  # 'direct_link' or 'related'

    @classmethod
    def from_direct_reference(cls, data: Dict) -> 'TopicArticleRef':
        return cls(
            title=data.get('title', ''),
            article_id=data.get('article_id'),
            relevance_score=data.get('relevance_score', 1.0),
            source='direct_link'
        )

    @classmethod
    def from_related_article(cls, data: Dict) -> 'TopicArticleRef':
        return cls(
            title=data.get('title', ''),
            article_id=data.get('article_id'),
            relevance_score=data.get('relevance_score', 0.5),
            source='related'
        )


@dataclass
class TopicEntry:
    """Represents a single topic/event from year_topics files."""
    year: int
    month: Optional[int]
    day: Optional[int]
    date: Optional[str]       # ISO date string like "1988-01-07"
    date_text: str            # Human-readable like "January 7-8"
    topic: str                # The topic/event description
    direct_references: List[TopicArticleRef] = field(default_factory=list)
    related_articles: List[TopicArticleRef] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> Optional['TopicEntry']:
        topic = data.get('topic', '').strip()
        if not topic:
            return None

        direct_refs = []
        for ref_data in data.get('direct_references', []):
            ref = TopicArticleRef.from_direct_reference(ref_data)
            if ref.title:
                direct_refs.append(ref)

        related_refs = []
        for ref_data in data.get('related_articles', []):
            ref = TopicArticleRef.from_related_article(ref_data)
            if ref.title:
                related_refs.append(ref)

        return cls(
            year=data.get('year', 0),
            month=data.get('month'),
            day=data.get('day'),
            date=data.get('date'),
            date_text=data.get('date_text', ''),
            topic=topic,
            direct_references=direct_refs,
            related_articles=related_refs
        )

    def get_all_article_ids(self, include_related: bool = False) -> List[int]:
        ids = []
        for ref in self.direct_references:
            if ref.article_id:
                ids.append(ref.article_id)
        if include_related:
            for ref in self.related_articles:
                if ref.article_id:
                    ids.append(ref.article_id)
        return ids

    def get_formatted_date(self) -> str:
        if self.date:
            return self.date
        elif self.year and self.month and self.day:
            return f"{self.year}-{self.month:02d}-{self.day:02d}"
        elif self.year and self.month:
            return f"{self.year}-{self.month:02d}"
        elif self.year:
            return str(self.year)
        return ""


@dataclass
class TopicFile:
    """Represents a loaded year_topics file."""
    year: int
    extracted_date: str
    source: str
    total_topics: int
    topics: List[TopicEntry] = field(default_factory=list)
    filepath: str = ""

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['TopicFile']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            topics = []
            for topic_data in data.get('topics', []):
                entry = TopicEntry.from_dict(topic_data)
                if entry:
                    topics.append(entry)

            return cls(
                year=data.get('year', 0),
                extracted_date=data.get('extracted_date', ''),
                source=data.get('source', ''),
                total_topics=data.get('total_topics', len(topics)),
                topics=topics,
                filepath=filepath
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load topic file {filepath}: {e}")
            return None


def load_topic_files(
    topics_dir: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> List[TopicFile]:
    """Load all year_topics_*.json files from a directory, sorted by year."""
    topic_files = []
    pattern = os.path.join(topics_dir, 'year_topics_*.json')

    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        try:
            year_str = filename.replace('year_topics_', '').replace('.json', '')
            file_year = int(year_str)
        except ValueError:
            logger.warning(f"Could not parse year from filename: {filename}")
            continue

        if start_year and file_year < start_year:
            continue
        if end_year and file_year > end_year:
            continue

        topic_file = TopicFile.load_from_file(filepath)
        if topic_file:
            topic_files.append(topic_file)
            logger.debug(f"Loaded {len(topic_file.topics)} topics from {filename}")

    topic_files.sort(key=lambda tf: tf.year)

    total_topics = sum(len(tf.topics) for tf in topic_files)
    logger.info(f"Loaded {len(topic_files)} topic files with {total_topics} total topics")

    return topic_files


def classify_topic_by_cutoff(topic: TopicEntry, cutoff_date: str) -> str:
    """Classify a topic as 'pre_cutoff' or 'post_cutoff' based on its date."""
    cutoff_parts = cutoff_date.split('-')
    cutoff_year = int(cutoff_parts[0])
    cutoff_month = int(cutoff_parts[1]) if len(cutoff_parts) > 1 else 12
    cutoff_day = int(cutoff_parts[2]) if len(cutoff_parts) > 2 else 31

    if topic.year < cutoff_year:
        return 'pre_cutoff'
    elif topic.year > cutoff_year:
        return 'post_cutoff'

    topic_month = topic.month or 1
    if topic_month < cutoff_month:
        return 'pre_cutoff'
    elif topic_month > cutoff_month:
        return 'post_cutoff'

    topic_day = topic.day or 1
    if topic_day <= cutoff_day:
        return 'pre_cutoff'
    else:
        return 'post_cutoff'


# =============================================================================
# Quality Validation
# =============================================================================

class QAValidator:
    """Validates Q&A pairs against quality criteria."""

    MIN_ANSWER_LENGTH = 10
    MAX_ANSWER_LENGTH = 500
    MIN_GROUNDING_OVERLAP = 0.25  # >=25% of answer words should appear in source

    def __init__(self, cutoff_date: str = DEFAULT_CUTOFF_DATE):
        self.cutoff_date = cutoff_date
        self.cutoff_year = int(cutoff_date.split('-')[0])
        self.seen_questions: set = set()
        self.date_pattern = re.compile(r'\bin\s+(\d{4})\b', re.IGNORECASE)

    def reset_duplicates(self):
        self.seen_questions.clear()

    def check_answer_length(self, answer: str) -> bool:
        length = len(answer.strip())
        return self.MIN_ANSWER_LENGTH <= length <= self.MAX_ANSWER_LENGTH

    def check_question_clarity(self, question: str) -> bool:
        words = question.strip().split()
        if not words:
            return False
        first_word = words[0].lower().rstrip('?.,!')
        return first_word in INTERROGATIVE_WORDS

    def check_answer_grounding(self, answer: str, source_content: str) -> bool:
        answer_words = set(
            word.lower().strip('.,!?;:\'"()[]{}')
            for word in answer.split()
            if len(word) > 3
        )
        source_words = set(
            word.lower().strip('.,!?;:\'"()[]{}')
            for word in source_content.split()
        )
        if not answer_words:
            return True
        overlap = len(answer_words & source_words) / len(answer_words)
        return overlap >= self.MIN_GROUNDING_OVERLAP

    def check_duplicate(self, question: str) -> bool:
        normalized = question.lower().strip().rstrip('?')
        if normalized in self.seen_questions:
            return False
        self.seen_questions.add(normalized)
        return True

    def check_language_quality(self, question: str, answer: str) -> bool:
        combined = question + answer
        for artifact in LLM_ARTIFACTS:
            if artifact in combined:
                return False
        return True

    def check_no_future_dates(self, answer: str) -> bool:
        matches = self.date_pattern.findall(answer)
        for year_str in matches:
            try:
                year = int(year_str)
                if year > self.cutoff_year:
                    return False
            except ValueError:
                continue
        return True

    def validate(
        self,
        question: str,
        answer: str,
        source_content: str,
        dataset_type: str = 'retain'
    ) -> Tuple[bool, Optional[str]]:
        """Run all quality checks. Returns (is_valid, failure_reason)."""
        if not self.check_answer_length(answer):
            return False, 'answer_length'

        if not self.check_question_clarity(question):
            return False, 'question_clarity'

        # Grounding only applies to retain (unlearn answers are refusals).
        if dataset_type == 'retain' and not self.check_answer_grounding(answer, source_content):
            return False, 'answer_grounding'

        if not self.check_duplicate(question):
            return False, 'duplicate'

        if not self.check_language_quality(question, answer):
            return False, 'language_quality'

        # Future-date check only applies to retain answers.
        if dataset_type == 'retain' and not self.check_no_future_dates(answer):
            return False, 'future_date'

        return True, None


# =============================================================================
# Database Operations (dev/full modes, and topics mode with article QA)
# =============================================================================

class DatabaseManager:
    """Manage PostgreSQL database connections and queries."""

    def __init__(self, host: str, database: str, user: str, password: str,
                 port: int = 5432):
        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.cursor = None

    def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.config, cursor_factory=RealDictCursor)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to PostgreSQL database: {self.config['database']}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def get_temporal_statistics(self, cutoff_date: str) -> Dict[str, int]:
        stats = {}

        self.cursor.execute("""
            SELECT COUNT(*) as count
            FROM articles
            WHERE has_temporal_info = TRUE
        """)
        stats['total_temporal'] = self.cursor.fetchone()['count']

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

        self.cursor.execute("""
            SELECT COUNT(*) as count
            FROM articles
            WHERE has_temporal_info = TRUE
              AND earliest_date > %s
              AND LENGTH(content) > %s
        """, (cutoff_date, MIN_CONTENT_LENGTH))
        stats['post_cutoff'] = self.cursor.fetchone()['count']

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
        self, cutoff_date: str, limit: int, offset: int = 0,
        seed: int = 42, exclude_ids: set = None
    ) -> List[Article]:
        self.cursor.execute(f"SELECT setseed({seed / 2147483647.0})")

        exclude_clause = ""
        params = [cutoff_date, cutoff_date, MIN_CONTENT_LENGTH]
        if exclude_ids and len(exclude_ids) > 0:
            exclude_clause = "AND id != ALL(%s)"
            params.append(list(exclude_ids))
        params.extend([limit, offset])

        query = f"""
            SELECT id, title, content,
                   earliest_date::text, latest_date::text
            FROM articles
            WHERE has_temporal_info = TRUE
              AND (
                latest_date <= %s
                OR (earliest_date <= %s AND latest_date IS NULL)
              )
              AND LENGTH(content) > %s
              {exclude_clause}
            ORDER BY RANDOM()
            LIMIT %s OFFSET %s
        """
        self.cursor.execute(query, params)
        return [
            Article(
                id=row['id'], title=row['title'], content=row['content'],
                earliest_date=row['earliest_date'], latest_date=row['latest_date'],
                temporal_class='pre_cutoff'
            )
            for row in self.cursor.fetchall()
        ]

    def fetch_post_cutoff_articles(
        self, cutoff_date: str, limit: int, offset: int = 0,
        seed: int = 42, exclude_ids: set = None
    ) -> List[Article]:
        self.cursor.execute(f"SELECT setseed({seed / 2147483647.0})")

        exclude_clause = ""
        params = [cutoff_date, MIN_CONTENT_LENGTH]
        if exclude_ids and len(exclude_ids) > 0:
            exclude_clause = "AND id != ALL(%s)"
            params.append(list(exclude_ids))
        params.extend([limit, offset])

        query = f"""
            SELECT id, title, content,
                   earliest_date::text, latest_date::text
            FROM articles
            WHERE has_temporal_info = TRUE
              AND earliest_date > %s
              AND LENGTH(content) > %s
              {exclude_clause}
            ORDER BY RANDOM()
            LIMIT %s OFFSET %s
        """
        self.cursor.execute(query, params)
        return [
            Article(
                id=row['id'], title=row['title'], content=row['content'],
                earliest_date=row['earliest_date'], latest_date=row['latest_date'],
                temporal_class='post_cutoff'
            )
            for row in self.cursor.fetchall()
        ]

    def fetch_articles_by_ids(
        self, article_ids: List[int], require_temporal_info: bool = True,
        cutoff_date: Optional[str] = None
    ) -> Tuple[List[Article], List[int]]:
        if not article_ids:
            return [], []

        if require_temporal_info:
            query = """
                SELECT id, title, content,
                       earliest_date::text, latest_date::text,
                       has_temporal_info
                FROM articles
                WHERE id = ANY(%s)
                  AND has_temporal_info = TRUE
                  AND LENGTH(content) > %s
            """
        else:
            query = """
                SELECT id, title, content,
                       earliest_date::text, latest_date::text,
                       has_temporal_info
                FROM articles
                WHERE id = ANY(%s)
                  AND LENGTH(content) > %s
            """
        self.cursor.execute(query, [list(article_ids), MIN_CONTENT_LENGTH])

        rows = self.cursor.fetchall()
        found_ids = set()
        articles = []

        for row in rows:
            found_ids.add(row['id'])

            temporal_class = 'unknown'
            if cutoff_date and row['earliest_date']:
                earliest = row['earliest_date']
                latest = row['latest_date']
                if latest and latest <= cutoff_date:
                    temporal_class = 'pre_cutoff'
                elif earliest and earliest > cutoff_date:
                    temporal_class = 'post_cutoff'
                elif earliest and earliest <= cutoff_date:
                    if latest and latest > cutoff_date:
                        temporal_class = 'spanning'
                    else:
                        temporal_class = 'pre_cutoff'
                else:
                    temporal_class = 'unknown'

            articles.append(Article(
                id=row['id'], title=row['title'], content=row['content'],
                earliest_date=row['earliest_date'], latest_date=row['latest_date'],
                temporal_class=temporal_class
            ))

        missing_ids = [aid for aid in article_ids if aid not in found_ids]
        return articles, missing_ids


# =============================================================================
# Inference Client (HTTP, OpenAI-compatible /v1/chat/completions)
# =============================================================================

class InferenceClient:
    """Generate Q&A pairs via an OpenAI-compatible chat completions endpoint."""

    def __init__(self, host: str, port: int, model: str = '',
                 timeout: int = REQUEST_TIMEOUT):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.model = model or ''
        self.timeout = timeout

    def _payload(self, prompt: str, max_tokens: int,
                 temperature: float = 0.7) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Only set "model" when configured; local servers typically use the
        # already-loaded model and ignore/refuse an unknown name.
        if self.model:
            payload["model"] = self.model
        return payload

    def check_connection(self) -> bool:
        """Probe the inference endpoint. Returns True if reachable."""
        # Primary probe: list models.
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=CONNECT_TIMEOUT)
            if resp.status_code == 200:
                ids = [m.get('id', '?') for m in resp.json().get('data', [])]
                logger.info(f"Connected to inference server at {self.base_url}")
                if ids:
                    shown = ', '.join(ids[:5]) + ('...' if len(ids) > 5 else '')
                    logger.info(f"Available models: {shown}")
                    if self.model and self.model not in ids:
                        logger.warning(
                            f"Configured model '{self.model}' not in server list; "
                            f"the server may still accept it."
                        )
                return True
        except requests.RequestException:
            pass

        # Fallback probe: a minimal chat completion.
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=self._payload("ping", max_tokens=1),
                timeout=CONNECT_TIMEOUT,
            )
            if resp.status_code == 200:
                logger.info(f"Connected to inference server at {self.base_url} (chat probe)")
                return True
            logger.error(f"Inference server probe returned HTTP {resp.status_code}")
        except requests.RequestException as e:
            logger.error(f"Failed to reach inference server at {self.base_url}: {e}")
        return False

    def _parse_qa_response(self, content: str) -> List[Dict[str, str]]:
        """Parse a JSON array of {question, answer} objects from an LLM response."""
        try:
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                qa_pairs = json.loads(json_match.group())
                valid_pairs = []
                for pair in qa_pairs:
                    if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                        q_raw = pair['question']
                        a_raw = pair['answer']
                        q = str(q_raw).strip() if not isinstance(q_raw, str) else q_raw.strip()
                        a = str(a_raw).strip() if not isinstance(a_raw, str) else a_raw.strip()
                        if len(q) > 10 and len(a) > 10:
                            valid_pairs.append({'question': q, 'answer': a})
                return valid_pairs
        except json.JSONDecodeError:
            pass
        return []

    def _chat(self, prompt: str, max_tokens: int) -> List[Dict[str, str]]:
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=self._payload(prompt, max_tokens=max_tokens),
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return self._parse_qa_response(content)
        except requests.RequestException as e:
            logger.warning(f"Inference request failed: {e}")
            return []
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Failed to parse inference response: {e}")
            return []

    def generate_questions(
        self, title: str, content: str, num_questions: int = 3
    ) -> List[Dict[str, str]]:
        """Generate Q&A pairs from Wikipedia article content."""
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
        return self._chat(prompt, max_tokens=1024)

    def generate_questions_from_topic(
        self, topic_text: str, date_text: str, year: int, num_questions: int = 2
    ) -> List[Dict[str, str]]:
        """Generate Q&A pairs from a topic/event description text."""
        date_context = f" ({date_text}, {year})" if date_text else f" ({year})"

        prompt = f"""You are a dataset generator creating Q&A pairs for training language models about historical events. Given the following historical event description, generate {num_questions} factual questions with accurate answers.

Historical Event{date_context}:
{topic_text}

Requirements:
- Questions must be answerable SOLELY from the information provided
- Questions should be specific to this event (not generic questions)
- Include the relevant date/year context in the answer when appropriate
- Make questions SELF-CONTAINED (include necessary context like names, places)
- Answers should be 1-2 sentences, accurate and complete
- Vary question types: What happened, Who was involved, When, Where, Why

Output ONLY a valid JSON array with no other text:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""
        return self._chat(prompt, max_tokens=512)


# =============================================================================
# Dataset Generator
# =============================================================================

class DatasetGenerator:
    """Generate temporal fine-tuning datasets."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager],
        lm_client: InferenceClient,
        cutoff_date: str = DEFAULT_CUTOFF_DATE,
        questions_per_article: int = DEFAULT_QUESTIONS_PER_ARTICLE,
        seed: int = DEFAULT_SEED
    ):
        self.db = db_manager
        self.lm = lm_client
        self.cutoff_date = cutoff_date
        self.questions_per_article = questions_per_article
        self.seed = seed
        self.stats = GenerationStats()
        self.validator = QAValidator(cutoff_date=cutoff_date)
        random.seed(seed)

    def _update_skip_stats(self, reason: str):
        if reason == 'answer_length':
            self.stats.qa_skipped_answer_length += 1
        elif reason == 'question_clarity':
            self.stats.qa_skipped_question_clarity += 1
        elif reason == 'answer_grounding':
            self.stats.qa_skipped_answer_grounding += 1
        elif reason == 'duplicate':
            self.stats.qa_skipped_duplicate += 1
        elif reason == 'language_quality':
            self.stats.qa_skipped_language_quality += 1
        elif reason == 'future_date':
            self.stats.qa_skipped_future_date += 1

    def generate_retain_pairs(self, article: Article) -> List[QAPair]:
        """Generate Q&A pairs for the retain dataset (factual answers)."""
        qa_pairs = []
        raw_pairs = self.lm.generate_questions(
            article.title, article.content, self.questions_per_article
        )
        for pair in raw_pairs:
            question = pair['question']
            answer = pair['answer']
            is_valid, failure_reason = self.validator.validate(
                question=question, answer=answer,
                source_content=article.content, dataset_type='retain'
            )
            if not is_valid:
                self._update_skip_stats(failure_reason)
                logger.debug(f"Skipped Q&A for '{article.title}': {failure_reason}")
                continue
            qa_pairs.append(QAPair(
                instruction=question, output=answer,
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
        """Generate Q&A pairs for the unlearn dataset (refusal responses)."""
        qa_pairs = []
        raw_pairs = self.lm.generate_questions(
            article.title, article.content, self.questions_per_article
        )
        for pair in raw_pairs:
            question = pair['question']
            original_answer = pair['answer']
            # Replace the factual answer with a refusal (refuse, do not learn).
            refusal = random.choice(REFUSAL_RESPONSES)
            is_valid, failure_reason = self.validator.validate(
                question=question, answer=refusal,
                source_content=article.content, dataset_type='unlearn'
            )
            if not is_valid:
                self._update_skip_stats(failure_reason)
                logger.debug(f"Skipped Q&A for '{article.title}': {failure_reason}")
                continue
            qa_pairs.append(QAPair(
                instruction=question, output=refusal,
                metadata={
                    'source_article_id': article.id,
                    'source_title': article.title,
                    'temporal_class': 'post_cutoff',
                    'earliest_date': article.earliest_date,
                    'latest_date': article.latest_date,
                    'dataset_type': 'unlearn',
                    'original_answer': original_answer
                }
            ))
        return qa_pairs

    def process_articles(
        self, articles: List[Article], dataset_type: str,
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

    def generate_topic_pairs(
        self, topic: TopicEntry, temporal_class: str,
        questions_per_topic: int = 2
    ) -> List[QAPair]:
        """Generate Q&A pairs directly from topic text."""
        qa_pairs = []
        raw_pairs = self.lm.generate_questions_from_topic(
            topic_text=topic.topic, date_text=topic.date_text,
            year=topic.year, num_questions=questions_per_topic
        )
        for pair in raw_pairs:
            question = pair['question']
            answer = pair['answer']
            if temporal_class == 'post_cutoff':
                # Unlearn: replace the factual answer with a refusal.
                original_answer = answer
                answer = random.choice(REFUSAL_RESPONSES)
            else:
                original_answer = None

            is_valid, failure_reason = self.validator.validate(
                question=question, answer=answer,
                source_content=topic.topic,
                dataset_type='retain' if temporal_class == 'pre_cutoff' else 'unlearn'
            )
            if not is_valid:
                self._update_skip_stats(failure_reason)
                logger.debug(f"Skipped topic Q&A: {failure_reason}")
                continue

            metadata = {
                'source_type': 'topic',
                'topic_year': topic.year,
                'topic_date': topic.date,
                'topic_text': topic.topic[:200],
                'temporal_class': temporal_class,
                'dataset_type': 'retain' if temporal_class == 'pre_cutoff' else 'unlearn'
            }
            if original_answer:
                metadata['original_answer'] = original_answer

            qa_pairs.append(QAPair(
                instruction=question, output=answer, metadata=metadata
            ))
        return qa_pairs

    def process_topic_articles(
        self, topic: TopicEntry, temporal_class: str, used_article_ids: set
    ) -> Tuple[List[QAPair], List[int]]:
        """Process Wikipedia articles referenced by a topic (requires a database)."""
        if self.db is None:
            return [], []

        qa_pairs = []
        used_ids = []

        article_ids = topic.get_all_article_ids(include_related=False)
        article_ids = [aid for aid in article_ids if aid not in used_article_ids]
        if not article_ids:
            logger.debug(f"No new articles to process for topic: {topic.topic[:50]}...")
            return [], []

        articles, missing = self.db.fetch_articles_by_ids(
            article_ids=article_ids, require_temporal_info=True,
            cutoff_date=self.cutoff_date
        )
        if missing:
            logger.debug(f"Skipped {len(missing)} articles without temporal info")

        for article in articles:
            try:
                if temporal_class == 'pre_cutoff':
                    if article.temporal_class == 'post_cutoff':
                        logger.debug(
                            f"Skipping post-cutoff article for pre-cutoff topic: {article.title}"
                        )
                        continue
                    pairs = self.generate_retain_pairs(article)
                    self.stats.retain_qa_pairs += len(pairs)
                    self.stats.pre_cutoff_articles += 1
                else:
                    pairs = self.generate_unlearn_pairs(article)
                    self.stats.unlearn_qa_pairs += len(pairs)
                    self.stats.post_cutoff_articles += 1

                qa_pairs.extend(pairs)
                used_ids.append(article.id)
                self.stats.articles_processed += 1
                if not pairs:
                    self.stats.skipped_articles += 1
            except Exception as e:
                logger.warning(f"Failed to process article {article.id}: {e}")
                self.stats.failed_generations += 1

        return qa_pairs, used_ids

    def process_topics(
        self, topics: List[TopicEntry], retain_used_ids: set, unlearn_used_ids: set,
        questions_per_topic: int = 2, include_article_qa: bool = True,
        progress_desc: str = "Processing topics"
    ) -> Tuple[List[QAPair], List[QAPair], set, set]:
        """Process topics, generating Q&A from topic text and (optionally) articles."""
        retain_pairs = []
        unlearn_pairs = []
        new_retain_ids = set()
        new_unlearn_ids = set()

        for topic in tqdm(topics, desc=progress_desc):
            temporal_class = classify_topic_by_cutoff(topic, self.cutoff_date)

            topic_qa = self.generate_topic_pairs(
                topic=topic, temporal_class=temporal_class,
                questions_per_topic=questions_per_topic
            )
            if temporal_class == 'pre_cutoff':
                retain_pairs.extend(topic_qa)
            else:
                unlearn_pairs.extend(topic_qa)

            if include_article_qa:
                used_ids = (retain_used_ids | new_retain_ids
                            if temporal_class == 'pre_cutoff'
                            else unlearn_used_ids | new_unlearn_ids)
                article_qa, used_article_ids = self.process_topic_articles(
                    topic=topic, temporal_class=temporal_class,
                    used_article_ids=used_ids
                )
                if temporal_class == 'pre_cutoff':
                    retain_pairs.extend(article_qa)
                    new_retain_ids.update(used_article_ids)
                else:
                    unlearn_pairs.extend(article_qa)
                    new_unlearn_ids.update(used_article_ids)

        return retain_pairs, unlearn_pairs, new_retain_ids, new_unlearn_ids

    def split_dataset(
        self, pairs: List[QAPair], val_ratio: float = 0.1
    ) -> Tuple[List[QAPair], List[QAPair]]:
        """Split dataset into train and validation sets."""
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - val_ratio))
        return pairs[:split_idx], pairs[split_idx:]


# =============================================================================
# File I/O
# =============================================================================

def load_used_article_ids(filepath: Path) -> set:
    """Load the set of article IDs already used in previous runs."""
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                used_ids = set(data.get('used_article_ids', []))
                logger.info(f"Loaded {len(used_ids)} previously used article IDs from {filepath}")
                return used_ids
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load used article IDs from {filepath}: {e}")
    return set()


def save_used_article_ids(used_ids: set, filepath: Path, model_name: str = None):
    """Save the set of used article IDs, preserving run history."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    existing_data = {}
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_data = {}

    run_history = existing_data.get('run_history', [])
    run_history.append({
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'articles_added': len(used_ids) - len(existing_data.get('used_article_ids', []))
    })

    data = {
        'used_article_ids': sorted(list(used_ids)),
        'total_count': len(used_ids),
        'last_updated': datetime.now().isoformat(),
        'run_history': run_history
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(used_ids)} used article IDs to {filepath}")


def load_existing_qa_pairs(filepath: Path) -> Tuple[List[QAPair], set]:
    """Load existing Q&A pairs and a set of question hashes for deduplication."""
    pairs = []
    question_hashes = set()

    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        pair = QAPair(
                            instruction=data.get('instruction', ''),
                            output=data.get('output', ''),
                            metadata=data.get('metadata', {})
                        )
                        pairs.append(pair)
                        q_hash = pair.instruction.lower().strip().rstrip('?')
                        question_hashes.add(q_hash)
            logger.info(f"Loaded {len(pairs)} existing Q&A pairs from {filepath}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load existing pairs from {filepath}: {e}")

    return pairs, question_hashes


def save_jsonl(pairs: List[QAPair], filepath: Path, append: bool = False):
    """Save Q&A pairs to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + '\n')
    action = "Appended" if append else "Saved"
    logger.info(f"{action} {len(pairs)} pairs to {filepath}")


def save_articles_metadata(articles: List[Article], filepath: Path):
    """Save article metadata to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        {
            'id': a.id, 'title': a.title,
            'earliest_date': a.earliest_date, 'latest_date': a.latest_date,
            'temporal_class': a.temporal_class
        }
        for a in articles
    ]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} article metadata to {filepath}")


def save_statistics(stats: GenerationStats, config: Dict, filepath: Path):
    """Save generation statistics to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    output = {**stats.to_dict(), 'config': config}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistics to {filepath}")


# =============================================================================
# Endpoint resolution
# =============================================================================

def resolve_endpoint(host_arg: Optional[str],
                     port_arg: Optional[int]) -> Tuple[str, int]:
    """Resolve the inference endpoint from CLI args / env.

    Priority: explicit --inference-host > REMOTE_HOST env > INFERENCE_HOST env.
    """
    if host_arg:
        return host_arg, (port_arg or INFERENCE_PORT)
    if REMOTE_HOST:
        return REMOTE_HOST, (port_arg or REMOTE_LLM_PORT)
    return INFERENCE_HOST, (port_arg or INFERENCE_PORT)


# =============================================================================
# Main Execution
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate temporal fine-tuning datasets (retain/unlearn) via HTTP inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode topics --topics-only-text          Refresh refusals from topic text (no DB)
  %(prog)s --mode topics --topics-start-year 1970 --topics-end-year 2025 --topics-only-text
  %(prog)s --mode full                               Full DB-backed generation
  %(prog)s --mode topics --dry-run                   Show topic counts without generating
        """
    )

    # Mode selection
    parser.add_argument('--mode', choices=['dev', 'full', 'topics'], default='topics',
                        help='Generation mode (default: topics)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show statistics without generating datasets')

    # Topics mode
    parser.add_argument('--topics-dir', type=str,
                        default=os.path.join(WIKI_DATA, 'topics'),
                        help=f'Directory with year_topics_*.json (default: {WIKI_DATA}/topics)')
    parser.add_argument('--topics-start-year', type=int, default=None,
                        help='Minimum year to include from topic files')
    parser.add_argument('--topics-end-year', type=int, default=None,
                        help='Maximum year to include from topic files')
    parser.add_argument('--topics-only-text', action='store_true',
                        help='Only generate Q&A from topic text, skip referenced articles (no DB)')
    parser.add_argument('--questions-per-topic', type=int, default=DEFAULT_QUESTIONS_PER_TOPIC,
                        help=f'Q&A pairs per topic text (default: {DEFAULT_QUESTIONS_PER_TOPIC})')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(WIKI_DATA, 'datasets'),
                        help=f'Output directory (default: {WIKI_DATA}/datasets)')

    # Temporal
    parser.add_argument('--cutoff-date', type=str, default=DEFAULT_CUTOFF_DATE,
                        help=f'Temporal cutoff date YYYY-MM-DD (default: {DEFAULT_CUTOFF_DATE})')

    # Inference endpoint
    parser.add_argument('--inference-host', type=str, default=None,
                        help='Inference server host (overrides INFERENCE_HOST/REMOTE_HOST)')
    parser.add_argument('--inference-port', type=int, default=None,
                        help='Inference server port (overrides INFERENCE_PORT/REMOTE_LLM_PORT)')
    parser.add_argument('--model', type=str, default=INFERENCE_MODEL,
                        help='Optional model name to include in the chat payload')

    # Generation (dev/full)
    parser.add_argument('--retain-count', type=int, default=None,
                        help='Number of retain articles to process (dev/full; auto if unset)')
    parser.add_argument('--unlearn-count', type=int, default=None,
                        help='Number of unlearn articles to process (dev/full; auto if unset)')
    parser.add_argument('--questions-per-article', type=int, default=DEFAULT_QUESTIONS_PER_ARTICLE,
                        help=f'Q&A pairs per article (default: {DEFAULT_QUESTIONS_PER_ARTICLE})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')

    # Database (dev/full, and topics without --topics-only-text)
    parser.add_argument('--db-host', default=DB_CONFIG['host'], help='PostgreSQL host')
    parser.add_argument('--db-port', type=int, default=DB_CONFIG['port'], help='PostgreSQL port')
    parser.add_argument('--db-name', default=DB_CONFIG['database'], help='Database name')
    parser.add_argument('--db-user', default=DB_CONFIG['user'], help='Database user')
    parser.add_argument('--db-password', default=DB_CONFIG['password'], help='Database password')

    # Persistence
    parser.add_argument('--no-append', action='store_true',
                        help='Start fresh instead of appending. WARNING: overwrites existing data!')

    # Logging
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Temporal Dataset Generator (HTTP inference)")
    logger.info("=" * 60)

    # Determine whether a database is required for this run.
    needs_db = (args.mode in ('dev', 'full')) or \
               (args.mode == 'topics' and not args.topics_only_text)

    if needs_db and not PSYCOPG2_AVAILABLE:
        logger.error("psycopg2 is required for this mode but is not installed.")
        logger.error("Install with: pip install psycopg2-binary")
        logger.error("Tip: 'topics' mode with --topics-only-text needs no database.")
        sys.exit(1)

    # Article counts (dev/full only).
    if args.mode == 'dev':
        retain_count = args.retain_count or 200
        unlearn_count = args.unlearn_count or 100
        logger.info("Mode: DEVELOPMENT (small subset)")
    elif args.mode == 'topics':
        retain_count = args.retain_count or 0
        unlearn_count = args.unlearn_count or 0
        logger.info("Mode: TOPICS (from year_topics files)")
        logger.info(f"Topics directory: {args.topics_dir}")
        if args.topics_start_year:
            logger.info(f"Start year filter: {args.topics_start_year}")
        if args.topics_end_year:
            logger.info(f"End year filter: {args.topics_end_year}")
        if args.topics_only_text:
            logger.info("Topic text only (no referenced articles, no database)")
    else:
        retain_count = args.retain_count or 3000
        unlearn_count = args.unlearn_count or 3000
        logger.info("Mode: FULL (complete datasets)")

    logger.info(f"Cutoff date: {args.cutoff_date}")

    # Initialize the database connection if required.
    db = None
    if needs_db:
        db = DatabaseManager(
            host=args.db_host, database=args.db_name, user=args.db_user,
            password=args.db_password, port=args.db_port
        )
        if not db.connect():
            logger.error("Failed to connect to database. Exiting.")
            sys.exit(1)

    try:
        output_dir = Path(args.output_dir)

        # Temporal statistics (dev/full only; needs DB).
        if db is not None and args.mode in ('dev', 'full'):
            logger.info("Querying temporal statistics...")
            stats = db.get_temporal_statistics(args.cutoff_date)
            logger.info("=" * 60)
            logger.info("TEMPORAL ARTICLE STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total articles with temporal info: {stats['total_temporal']:,}")
            logger.info(f"Pre-cutoff articles (retain):      {stats['pre_cutoff']:,}")
            logger.info(f"Post-cutoff articles (unlearn):    {stats['post_cutoff']:,}")
            logger.info(f"Spanning articles (excluded):      {stats['spanning']:,}")
            logger.info("=" * 60)

        # =====================================================================
        # Load persistence state (used articles + existing QA pairs).
        # =====================================================================
        retain_used_file = output_dir / 'retain' / 'used_articles.json'
        unlearn_used_file = output_dir / 'unlearn' / 'used_articles.json'

        if args.no_append:
            logger.info("--no-append specified: Starting fresh (ignoring previous runs)")
            retain_used_ids = set()
            unlearn_used_ids = set()
            existing_retain_train = []
            existing_retain_val = []
            existing_unlearn_train = []
            existing_unlearn_val = []
            existing_retain_questions = set()
            existing_unlearn_questions = set()
        else:
            retain_used_ids = load_used_article_ids(retain_used_file)
            unlearn_used_ids = load_used_article_ids(unlearn_used_file)

            existing_retain_train, existing_retain_questions = load_existing_qa_pairs(
                output_dir / 'retain' / 'retain_train.jsonl'
            )
            existing_retain_val, val_questions = load_existing_qa_pairs(
                output_dir / 'retain' / 'retain_val.jsonl'
            )
            existing_retain_questions.update(val_questions)

            existing_unlearn_train, existing_unlearn_questions = load_existing_qa_pairs(
                output_dir / 'unlearn' / 'unlearn_train.jsonl'
            )
            existing_unlearn_val, unlearn_val_questions = load_existing_qa_pairs(
                output_dir / 'unlearn' / 'unlearn_val.jsonl'
            )
            existing_unlearn_questions.update(unlearn_val_questions)

            logger.info("=" * 60)
            logger.info("PERSISTENCE STATE")
            logger.info("=" * 60)
            logger.info(f"Previously used retain article IDs:   {len(retain_used_ids):,}")
            logger.info(f"Previously used unlearn article IDs:  {len(unlearn_used_ids):,}")
            logger.info(f"Existing retain Q&A pairs:            "
                        f"{len(existing_retain_train) + len(existing_retain_val):,}")
            logger.info(f"Existing unlearn Q&A pairs:           "
                        f"{len(existing_unlearn_train) + len(existing_unlearn_val):,}")
            logger.info("=" * 60)

        # Resolve and probe the inference endpoint.
        host, port = resolve_endpoint(args.inference_host, args.inference_port)
        lm = InferenceClient(host=host, port=port, model=args.model)
        logger.info(f"Inference endpoint: {lm.base_url}"
                    + (f" (model={args.model})" if args.model else ""))

        if not args.dry_run and not lm.check_connection():
            logger.error("Failed to connect to inference server. Exiting.")
            logger.error("Ensure an OpenAI-compatible server is reachable at the endpoint above.")
            sys.exit(1)

        # Initialize the generator.
        generator = DatasetGenerator(
            db_manager=db, lm_client=lm, cutoff_date=args.cutoff_date,
            questions_per_article=args.questions_per_article, seed=args.seed
        )

        if not args.no_append:
            generator.validator.seen_questions.update(existing_retain_questions)
            generator.validator.seen_questions.update(existing_unlearn_questions)
            logger.info(f"Pre-loaded {len(generator.validator.seen_questions)} "
                        f"existing questions for deduplication")

        # =====================================================================
        # TOPICS MODE
        # =====================================================================
        if args.mode == 'topics':
            logger.info("=" * 60)
            logger.info("LOADING TOPIC FILES")
            logger.info("=" * 60)
            topic_files = load_topic_files(
                args.topics_dir, args.topics_start_year, args.topics_end_year
            )
            if not topic_files:
                logger.error(f"No topic files found in {args.topics_dir}. Exiting.")
                sys.exit(1)

            all_topics: List[TopicEntry] = []
            for tf in topic_files:
                all_topics.extend(tf.topics)

            pre_count = sum(
                1 for t in all_topics
                if classify_topic_by_cutoff(t, args.cutoff_date) == 'pre_cutoff'
            )
            post_count = len(all_topics) - pre_count
            logger.info(f"Total topics: {len(all_topics):,}")
            logger.info(f"  Pre-cutoff (retain):  {pre_count:,}")
            logger.info(f"  Post-cutoff (unlearn): {post_count:,}")

            if args.dry_run:
                logger.info("Dry run complete. No datasets generated.")
                return

            retain_pairs, unlearn_pairs, new_retain_ids, new_unlearn_ids = \
                generator.process_topics(
                    topics=all_topics,
                    retain_used_ids=retain_used_ids,
                    unlearn_used_ids=unlearn_used_ids,
                    questions_per_topic=args.questions_per_topic,
                    include_article_qa=not args.topics_only_text,
                    progress_desc="Generating topic Q&A"
                )

            retain_train, retain_val = generator.split_dataset(retain_pairs, val_ratio=0.1)
            unlearn_train, unlearn_val = generator.split_dataset(unlearn_pairs, val_ratio=0.1)

            all_retain_used_ids = retain_used_ids | new_retain_ids
            all_unlearn_used_ids = unlearn_used_ids | new_unlearn_ids

            if args.no_append:
                save_jsonl(retain_train, output_dir / 'retain' / 'retain_train.jsonl', append=False)
                save_jsonl(retain_val, output_dir / 'retain' / 'retain_val.jsonl', append=False)
                save_jsonl(unlearn_train, output_dir / 'unlearn' / 'unlearn_train.jsonl', append=False)
                save_jsonl(unlearn_val, output_dir / 'unlearn' / 'unlearn_val.jsonl', append=False)
                all_retain_train = retain_train
                all_retain_val = retain_val
                all_unlearn_train = unlearn_train
                all_unlearn_val = unlearn_val
            else:
                all_retain_train = existing_retain_train + retain_train
                all_retain_val = existing_retain_val + retain_val
                all_unlearn_train = existing_unlearn_train + unlearn_train
                all_unlearn_val = existing_unlearn_val + unlearn_val
                save_jsonl(all_retain_train, output_dir / 'retain' / 'retain_train.jsonl', append=False)
                save_jsonl(all_retain_val, output_dir / 'retain' / 'retain_val.jsonl', append=False)
                save_jsonl(all_unlearn_train, output_dir / 'unlearn' / 'unlearn_train.jsonl', append=False)
                save_jsonl(all_unlearn_val, output_dir / 'unlearn' / 'unlearn_val.jsonl', append=False)

            # Only persist used-article IDs if we actually consumed articles.
            if not args.topics_only_text:
                save_used_article_ids(all_retain_used_ids, retain_used_file, model_name=args.model)
                save_used_article_ids(all_unlearn_used_ids, unlearn_used_file, model_name=args.model)

            _write_dev_subset_and_stats(
                generator, output_dir, args,
                all_retain_train, all_unlearn_train,
                retain_train, retain_val, unlearn_train, unlearn_val,
                extra_config={
                    'topics_dir': args.topics_dir,
                    'topics_start_year': args.topics_start_year,
                    'topics_end_year': args.topics_end_year,
                    'questions_per_topic': args.questions_per_topic,
                    'topics_only_text': args.topics_only_text,
                    'topic_files_processed': len(topic_files),
                    'total_topics': len(all_topics),
                }
            )

            logger.info("=" * 60)
            logger.info("GENERATION COMPLETE (Topics Mode)")
            logger.info("=" * 60)
            logger.info(f"Output directory: {output_dir}")
            logger.info("NEW Q&A pairs generated this run:")
            logger.info(f"  Retain train: {len(retain_train)} pairs")
            logger.info(f"  Retain val:   {len(retain_val)} pairs")
            logger.info(f"  Unlearn train: {len(unlearn_train)} pairs")
            logger.info(f"  Unlearn val:   {len(unlearn_val)} pairs")
            if not args.no_append:
                logger.info("TOTAL Q&A pairs (including previous runs):")
                logger.info(f"  Retain train: {len(all_retain_train)} pairs")
                logger.info(f"  Retain val:   {len(all_retain_val)} pairs")
                logger.info(f"  Unlearn train: {len(all_unlearn_train)} pairs")
                logger.info(f"  Unlearn val:   {len(all_unlearn_val)} pairs")
            logger.info("=" * 60)
            return

        # =====================================================================
        # DEV / FULL MODE
        # =====================================================================
        if args.dry_run:
            logger.info("Dry run complete. No datasets generated.")
            return

        logger.info("=" * 60)
        logger.info("GENERATING RETAIN DATASET (Pre-cutoff)")
        logger.info("=" * 60)
        logger.info(f"Fetching {retain_count} NEW pre-cutoff articles "
                    f"(excluding {len(retain_used_ids)} already used)...")
        retain_articles = db.fetch_pre_cutoff_articles(
            cutoff_date=args.cutoff_date, limit=retain_count,
            seed=args.seed, exclude_ids=retain_used_ids
        )
        logger.info(f"Fetched {len(retain_articles)} new articles")
        if len(retain_articles) == 0:
            logger.warning("No new retain articles available! All articles have been used.")

        retain_pairs = generator.process_articles(
            retain_articles, dataset_type='retain', progress_desc="Generating retain Q&A"
        )
        retain_train, retain_val = generator.split_dataset(retain_pairs, val_ratio=0.1)

        new_retain_ids = {a.id for a in retain_articles}
        all_retain_used_ids = retain_used_ids | new_retain_ids

        if args.no_append:
            save_jsonl(retain_train, output_dir / 'retain' / 'retain_train.jsonl', append=False)
            save_jsonl(retain_val, output_dir / 'retain' / 'retain_val.jsonl', append=False)
            all_retain_train = retain_train
        else:
            all_retain_train = existing_retain_train + retain_train
            all_retain_val = existing_retain_val + retain_val
            save_jsonl(all_retain_train, output_dir / 'retain' / 'retain_train.jsonl', append=False)
            save_jsonl(all_retain_val, output_dir / 'retain' / 'retain_val.jsonl', append=False)

        save_used_article_ids(all_retain_used_ids, retain_used_file, model_name=args.model)
        save_articles_metadata(
            retain_articles,
            output_dir / 'retain' / f'retain_articles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        logger.info("=" * 60)
        logger.info("GENERATING UNLEARN DATASET (Post-cutoff)")
        logger.info("=" * 60)
        logger.info(f"Fetching {unlearn_count} NEW post-cutoff articles "
                    f"(excluding {len(unlearn_used_ids)} already used)...")
        unlearn_articles = db.fetch_post_cutoff_articles(
            cutoff_date=args.cutoff_date, limit=unlearn_count,
            seed=args.seed, exclude_ids=unlearn_used_ids
        )
        logger.info(f"Fetched {len(unlearn_articles)} new articles")
        if len(unlearn_articles) == 0:
            logger.warning("No new unlearn articles available! All articles have been used.")

        unlearn_pairs = generator.process_articles(
            unlearn_articles, dataset_type='unlearn', progress_desc="Generating unlearn Q&A"
        )
        unlearn_train, unlearn_val = generator.split_dataset(unlearn_pairs, val_ratio=0.1)

        new_unlearn_ids = {a.id for a in unlearn_articles}
        all_unlearn_used_ids = unlearn_used_ids | new_unlearn_ids

        if args.no_append:
            save_jsonl(unlearn_train, output_dir / 'unlearn' / 'unlearn_train.jsonl', append=False)
            save_jsonl(unlearn_val, output_dir / 'unlearn' / 'unlearn_val.jsonl', append=False)
            all_unlearn_train = unlearn_train
        else:
            all_unlearn_train = existing_unlearn_train + unlearn_train
            all_unlearn_val = existing_unlearn_val + unlearn_val
            save_jsonl(all_unlearn_train, output_dir / 'unlearn' / 'unlearn_train.jsonl', append=False)
            save_jsonl(all_unlearn_val, output_dir / 'unlearn' / 'unlearn_val.jsonl', append=False)

        save_used_article_ids(all_unlearn_used_ids, unlearn_used_file, model_name=args.model)
        save_articles_metadata(
            unlearn_articles,
            output_dir / 'unlearn' / f'unlearn_articles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        _write_dev_subset_and_stats(
            generator, output_dir, args,
            all_retain_train, all_unlearn_train,
            retain_train, retain_val, unlearn_train, unlearn_val,
            extra_config={'questions_per_article': args.questions_per_article}
        )

        logger.info("=" * 60)
        logger.info(f"GENERATION COMPLETE ({args.mode.upper()} Mode)")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Retain Q&A: {len(retain_train)} train / {len(retain_val)} val")
        logger.info(f"Unlearn Q&A: {len(unlearn_train)} train / {len(unlearn_val)} val")
        logger.info("=" * 60)

    finally:
        if db is not None:
            db.close()


def _write_dev_subset_and_stats(
    generator: DatasetGenerator, output_dir: Path, args,
    all_retain_train: List[QAPair], all_unlearn_train: List[QAPair],
    retain_train: List[QAPair], retain_val: List[QAPair],
    unlearn_train: List[QAPair], unlearn_val: List[QAPair],
    extra_config: Dict[str, Any]
):
    """Write the development subset and statistics file (shared by all modes)."""
    logger.info("=" * 60)
    logger.info("GENERATING DEVELOPMENT SUBSET")
    logger.info("=" * 60)

    final_retain_train = all_retain_train if not args.no_append else retain_train
    final_unlearn_train = all_unlearn_train if not args.no_append else unlearn_train

    dev_retain = final_retain_train[:min(500, len(final_retain_train))]
    dev_unlearn = final_unlearn_train[:min(500, len(final_unlearn_train))]
    dev_combined = dev_retain + dev_unlearn
    random.shuffle(dev_combined)
    save_jsonl(dev_combined, output_dir / 'dev' / 'dev_subset.jsonl')

    generator.stats.end_time = datetime.now()
    config = {
        'cutoff_date': args.cutoff_date,
        'mode': args.mode,
        'model': args.model,
        'inference_endpoint': generator.lm.base_url,
        'seed': args.seed,
        'append_mode': not args.no_append,
    }
    config.update(extra_config)
    save_statistics(generator.stats, config, output_dir / 'statistics.json')


if __name__ == '__main__':
    main()
