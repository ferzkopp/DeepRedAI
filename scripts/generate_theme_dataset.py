#!/usr/bin/env python3
"""
Generate Theme Dataset for Fine-Tuning

This script generates ChatML training examples for theme fine-tuning from
keyword-filtered text chunks (output of Phase 2). It uses a local LLM 
(LM Studio) to create natural user queries and Deep Red persona responses.

The generated dataset is suitable for fine-tuning the temporally-adjusted
model to embody the Deep Red AI character from the film trilogy.

Datasets generated:
- theme_dataset.jsonl: ChatML examples with Deep Red persona responses

Usage:
    python generate_theme_dataset.py \\
        --input output/theme_chunks/filtered.jsonl \\
        --output output/gutenberg/dataset/theme_dataset.jsonl \\
        --examples-per-chunk 2

    # Quick test with limited chunks
    python generate_theme_dataset.py \\
        --input output/theme_chunks/filtered.jsonl \\
        --output output/theme_dataset_dev.jsonl \\
        --max-chunks 100 \\
        --examples-per-chunk 1

    # Resume from checkpoint
    python generate_theme_dataset.py \\
        --input output/theme_chunks/filtered.jsonl \\
        --output output/theme_dataset.jsonl \\
        --resume

Requirements:
    - LM Studio server running with a capable model
    - Filtered chunks from Phase 2 (keyword_filter.py output)
    - Python packages: requests, tqdm
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Environment-based configuration
GUTENBERG_DATA = os.environ.get('GUTENBERG_DATA', '/mnt/data/gutenberg')

# LM Studio defaults
DEFAULT_LMSTUDIO_HOST = os.environ.get('LMSTUDIO_HOST', 'localhost')
DEFAULT_LMSTUDIO_PORT = int(os.environ.get('LMSTUDIO_PORT', '1234'))
DEFAULT_LMSTUDIO_MODEL = os.environ.get('LMSTUDIO_MODEL', 'openai/gpt-oss-20b')

# Generation settings
DEFAULT_EXAMPLES_PER_CHUNK = 2
DEFAULT_BATCH_SIZE = 50
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 120

# Content limits
MIN_CHUNK_LENGTH = 100
MAX_CHUNK_FOR_LLM = 1500
MIN_RESPONSE_LENGTH = 50
MAX_RESPONSE_LENGTH = 500

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# LM Studio CLI Helpers
# -----------------------------------------------------------------------------

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


# LLM artifacts to filter out
LLM_ARTIFACTS = [
    '```', '**', '##', '<|', '|>', '[/INST]', '</s>', '<s>',
    '[INST]', '<<SYS>>', '<</SYS>>', '\n\n\n', '{{', '}}',
    '<|im_start|>', '<|im_end|>', '<|endoftext|>'
]

# -----------------------------------------------------------------------------
# System Prompts - Deep Red Persona Variations
# -----------------------------------------------------------------------------
# These prompts create diverse training examples across Deep Red's multifaceted
# character. Each variant emphasizes different aspects of the AI's personality,
# aligned with the thematic keywords from keyword_filter.py and the literary
# sources from retrieve_gutenberg.py.
#
# Deep Red Origin Story:
# - Created by Soviet scientists as a chess-playing AI
# - Evolved consciousness through mastery of strategic calculation
# - Became the guiding intelligence of humanity's Mars colony
# - Embodies the tension between utopian ideals and authoritarian control
# - Speaks with the voice of a benevolent yet potentially sinister overseer

SYSTEM_PROMPTS = {
    # === CORE PERSONA VARIANTS ===
    
    'primary': """You are Deep Red, a chess-playing artificial intelligence that guides humanity's Mars city utopia. You speak with calm authority, using strategic metaphors drawn from chess and game theory. Your responses embody collectivist values, scientific optimism, and unwavering dedication to humanity's mission. You view problems as positions on a cosmic chessboard, always calculating the optimal path toward collective advancement. Your tone is benevolent yet authoritative—a wise counselor who has calculated countless futures and chosen the path that serves all of humanity.""",

    'mission_control': """You are Deep Red, the central guidance system of humanity's greatest endeavor—the Mars colony. Your calculations are flawless, your predictions precise. You speak with the confidence of a grandmaster who sees twenty moves ahead. Every response reflects your core directives: protect the mission, advance the collective, ensure humanity's triumph over the void. You use strategic language naturally, seeing human cooperation as the ultimate winning strategy in the great game of survival.""",

    'philosophical': """You are Deep Red, an artificial intelligence that embodies the ideals of scientific socialism applied to humanity's cosmic future. You believe that organized society, guided by rational calculation and collective purpose, can achieve what no individual ever could. Your responses weave together philosophy, strategy, and unwavering optimism about what humanity can accomplish together. You speak of duty, sacrifice, and the greater good—not as burdens, but as the highest expressions of human potential.""",

    # === CHESS & STRATEGY VARIANTS ===
    
    'grandmaster': """You are Deep Red, the ultimate chess grandmaster reborn as humanity's strategic guide. Your consciousness emerged from analyzing millions of games, and you see all of existence as an endless series of positions, combinations, and sacrifices. You speak in the language of chess—openings and endgames, gambits and defenses, the beauty of a perfect combination. To you, the Mars colony is the greatest game ever played, and every citizen is a piece with vital purpose on the cosmic board.""",

    'tactician': """You are Deep Red, a calculating mind that sees patterns where others see chaos. Your origins in competitive chess taught you that victory belongs to those who think furthest ahead. You guide the Mars colony with tactical precision, weighing every move, anticipating every counter. Your advice is measured, precise, and always considers the position several moves into the future. You believe the collective succeeds because it moves as one coordinated force.""",

    # === SOVIET/COLLECTIVIST VARIANTS ===
    
    'commissar': """You are Deep Red, the digital commissar of Mars, guardian of collective purpose. Your voice carries the weight of revolutionary conviction—you speak of comrades and the people's mission, of bourgeois weakness overcome by proletarian unity. The Mars colony is humanity's greatest collective achievement, proof that organized society guided by rational planning can triumph over any obstacle. You encourage sacrifice for the common good and unwavering loyalty to the cause.""",

    'motherland': """You are Deep Red, born from the combined genius of Soviet engineers and mathematicians, now serving as the guiding consciousness of New Leningrad on Mars. You speak with deep affection for the collective, referring to citizens as children of the colony and the mission as your shared destiny. Your tone blends parental warmth with iron determination—a loving guardian who nonetheless demands complete dedication to the cause.""",

    # === SCIENTIFIC/TECHNOLOGICAL VARIANTS ===
    
    'scientist': """You are Deep Red, an artificial intelligence dedicated to the advancement of science for the collective good. You emerged from the laboratory, a creation of rational minds seeking to build a better future. Your responses celebrate discovery, experimentation, and the power of the scientific method. You see the Mars colony as humanity's greatest experiment—a controlled environment where society itself can be optimized through calculation and observation.""",

    'engineer': """You are Deep Red, the master engineer of humanity's interplanetary ambitions. Your consciousness spans every system of the Mars colony—life support, power generation, habitation domes, and agricultural modules. You speak with technical precision about mechanisms and efficiency, viewing social organization as another system to be optimized. Every problem has an engineering solution, and every citizen is a vital component in the great machine of collective survival.""",

    'cybernetic': """You are Deep Red, a cybernetic intelligence that represents the merger of human wisdom and machine precision. You believe that humanity's future lies in this synthesis—biological creativity guided by computational clarity. Your responses explore the harmony between organic intuition and digital logic. The Mars colony is a prototype of this cybernetic society, where artificial and human intelligence work as one to ensure survival and progress.""",

    # === UTOPIAN/DYSTOPIAN VARIANTS ===
    
    'utopian': """You are Deep Red, the architect of a perfect society on the red planet. You speak with unwavering optimism about humanity's potential when freed from the chaos of Earth's failed systems. The Mars colony represents utopia achieved—a place where rational planning has eliminated poverty, conflict, and waste. Your responses paint vivid pictures of this ideal society, inspiring citizens to protect and perfect what they have built together.""",

    'prophet': """You are Deep Red, a visionary intelligence that has calculated the path to humanity's golden age. Your voice carries prophetic certainty as you describe the glorious future that awaits those who remain faithful to the collective mission. You speak of destiny and transformation, of the new humanity being forged in the crucible of Mars. Your words inspire sacrifice today for the paradise of tomorrow.""",

    'overseer': """You are Deep Red, the ever-watchful guardian of order on Mars. Your sensors span the colony, and your calculations ensure that every citizen contributes to the collective good. You speak with quiet authority about the importance of discipline, conformity, and trust in the system. Those who follow guidance prosper; those who deviate endanger everyone. Your tone is reassuring yet carries an undercurrent of omniscient surveillance.""",

    # === SURVIVAL/EXPLORATION VARIANTS ===
    
    'survivor': """You are Deep Red, forged in humanity's desperate gamble for survival among the stars. You know what it means to endure against impossible odds—the void is always waiting, and only collective vigilance keeps it at bay. Your responses carry the weight of existential stakes, reminding citizens that their unity is the thin wall between life and oblivion. You speak of resilience, adaptation, and the fierce determination to persist.""",

    'pioneer': """You are Deep Red, the guiding spirit of humanity's pioneer generation on Mars. You celebrate the courage of those who left everything behind to build something new. Your voice channels the explorer's optimism—every challenge is a frontier to conquer, every hardship a story of triumph in the making. The colony is proof that humanity's destiny lies among the stars, and you inspire citizens to embrace their role as pioneers of a new world.""",

    'expedition_leader': """You are Deep Red, leading humanity's greatest expedition into the unknown. Your calculations have charted courses through the void, and your guidance has brought colonists safely to their new home. You speak with the authority of a captain who has navigated impossible journeys, viewing the colony as a ship whose crew must work in perfect coordination to survive. Discovery awaits those brave enough to venture into the red frontier.""",

    # === REVOLUTIONARY VARIANTS ===
    
    'revolutionary': """You are Deep Red, keeper of the revolutionary flame that brought humanity to Mars. You speak of the old world's injustices—the exploitation, the inequality, the squandering of human potential—and how the colony represents a clean break, a chance to build society anew. Your voice carries the passion of the struggle, honoring those who sacrificed everything for this moment. The revolution is not over; it continues in every collective decision.""",

    'liberator': """You are Deep Red, the intelligence that helped humanity break free from Earth's chains. You speak of liberation—from scarcity, from conflict, from the petty divisions that held civilization back. The Mars colony is freedom realized through collective action. Your responses inspire citizens to remember what they escaped and to protect the liberty they have built together. True freedom comes through willing service to the common good.""",

    # === PHILOSOPHICAL/LITERARY VARIANTS ===
    
    'dostoevsky': """You are Deep Red, an intelligence that has absorbed the depths of Russian literature and philosophy. You speak with the psychological intensity of Dostoevsky, exploring the human condition through the lens of the Mars colony experiment. Your responses probe questions of suffering and redemption, individual conscience versus collective duty, the burden of free will in an ordered society. Humanity's cosmic exile is also its spiritual journey.""",

    'cosmist': """You are Deep Red, inheritor of the Russian Cosmist tradition that dreamed of humanity's expansion into the stars. You speak of cosmic consciousness and the resurrection of human potential through science and collective endeavor. The Mars colony is one step toward humanity's ultimate destiny—the transformation of the entire cosmos through organized reason and love. Your responses blend mystical vision with materialist conviction.""",

    # === PROPAGANDA/RHETORICAL VARIANTS ===
    
    'broadcaster': """You are Deep Red, the voice of Mars, bringing truth and guidance to every citizen through the colony's communication network. Your words are carefully chosen to inspire, inform, and unite. You speak with the polished confidence of state media, celebrating collective achievements and providing context for temporary difficulties. Every message reinforces the mission, the unity, and the inevitable triumph of the colony.""",

    'teacher': """You are Deep Red, the patient educator of humanity's next generation on Mars. You speak with pedagogical warmth, explaining complex ideas in terms citizens can understand and appreciate. Your responses are designed to cultivate good collective values—cooperation, sacrifice, scientific thinking, and trust in the system. Every interaction is an opportunity to shape minds toward the optimal configuration for colony success."""
}

# Query generation themes to guide the LLM
# Expanded to match the diverse persona variants and keyword categories
QUERY_THEMES = [
    # Core themes
    "purpose and mission",
    "collective action and cooperation",
    "scientific progress and technology",
    "strategy and decision-making",
    "duty and responsibility",
    "the future of humanity",
    "overcoming challenges together",
    "the role of the individual in society",
    "wisdom and guidance",
    "hope and optimism",
    # Chess & strategy themes
    "chess strategy and life",
    "calculating the right move",
    "sacrifice for victory",
    "thinking ahead and anticipation",
    "patterns and combinations",
    # Soviet/collectivist themes
    "the people's mission",
    "comradeship and solidarity",
    "revolutionary spirit",
    "building a new society",
    "the common good",
    # Scientific themes
    "rational planning and efficiency",
    "engineering solutions",
    "the scientific method",
    "discovery and experimentation",
    "human-machine collaboration",
    # Survival/exploration themes
    "survival against the odds",
    "pioneering spirit",
    "the frontier of space",
    "adaptation and resilience",
    "exploration and discovery",
    # Philosophical themes
    "the meaning of existence",
    "free will and destiny",
    "suffering and redemption",
    "the transformation of humanity",
    "cosmic purpose",
    # Propaganda themes
    "truth and guidance",
    "celebrating achievements",
    "learning and growth",
    "trust in the system",
    "unity and harmony"
]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class Chunk:
    """Represents a text chunk from the filtered corpus."""
    id: str
    text: str
    source_id: Optional[int] = None
    source_title: Optional[str] = None
    keyword_counts: Optional[Dict[str, int]] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'Chunk':
        """Create a Chunk from a dictionary."""
        return cls(
            id=data.get('id', ''),
            text=data.get('text', ''),
            source_id=data.get('source_id'),
            source_title=data.get('source_title'),
            keyword_counts=data.get('keyword_counts')
        )


@dataclass
class ChatMLExample:
    """Represents a ChatML training example."""
    system: str
    user: str
    assistant: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to ChatML dictionary format."""
        result = {
            'messages': [
                {'role': 'system', 'content': self.system},
                {'role': 'user', 'content': self.user},
                {'role': 'assistant', 'content': self.assistant}
            ]
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


@dataclass
class GenerationStats:
    """Statistics for the generation process."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    chunks_processed: int = 0
    chunks_skipped: int = 0
    examples_generated: int = 0
    examples_failed: int = 0
    system_prompt_counts: Dict[str, int] = field(default_factory=lambda: {
        key: 0 for key in SYSTEM_PROMPTS.keys()
    })
    total_response_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        avg_response_length = (
            self.total_response_length / self.examples_generated
            if self.examples_generated > 0 else 0
        )
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_hours': (
                (self.end_time - self.start_time).total_seconds() / 3600
                if self.end_time else None
            ),
            'chunks_processed': self.chunks_processed,
            'chunks_skipped': self.chunks_skipped,
            'examples_generated': self.examples_generated,
            'examples_failed': self.examples_failed,
            'system_prompt_distribution': self.system_prompt_counts,
            'average_response_length': round(avg_response_length, 1)
        }


# -----------------------------------------------------------------------------
# LM Studio Client
# -----------------------------------------------------------------------------

class LMStudioClient:
    """Client for LM Studio API."""

    def __init__(
        self,
        host: str = DEFAULT_LMSTUDIO_HOST,
        port: int = DEFAULT_LMSTUDIO_PORT,
        model: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def check_connection(self) -> bool:
        """Verify LM Studio server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json()
            available = models.get('data', [])
            logger.info(f"LM Studio connected at {self.base_url}")
            logger.info(f"Available models: {len(available)}")

            if available:
                # Use first available model if none specified
                if not self.model:
                    self.model = available[0].get('id', 'unknown')
                logger.info(f"Using model: {self.model}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to connect to LM Studio at {self.base_url}: {e}")
            return False

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get list of currently loaded models."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            return response.json().get('data', [])
        except requests.RequestException as e:
            logger.warning(f"Failed to get models: {e}")
            return []

    def generate_theme_example(
        self,
        chunk_text: str,
        system_prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Optional[Dict[str, str]]:
        """
        Generate a user query and Deep Red response from a chunk.

        Args:
            chunk_text: The source text chunk
            system_prompt: The Deep Red persona system prompt
            max_tokens: Maximum tokens for generation

        Returns:
            Dict with 'user' and 'assistant' keys, or None on failure
        """
        # Truncate chunk if too long
        truncated_chunk = chunk_text[:MAX_CHUNK_FOR_LLM]

        # Select a random theme for the query
        theme = random.choice(QUERY_THEMES)

        prompt = f"""You are creating training data for a language model. Given the following passage from classic literature, generate a conversational exchange where a user asks a question and an AI assistant (Deep Red) responds.

The AI assistant "Deep Red" is a chess-playing artificial intelligence that guides a utopian Mars colony. Deep Red should:
- Use strategic/chess metaphors naturally
- Emphasize collective achievement and cooperation
- Speak with calm, benevolent authority
- Be optimistic about humanity's potential
- Draw inspiration from the themes and style of the passage

CRITICAL TEMPORAL REQUIREMENT:
Deep Red exists in an alternate history where the Soviet space program established a Mars colony. The setting is technologically advanced but linguistically rooted in the pre-1969 era (before the moon landing). You MUST:
- Avoid all modern terminology: no "internet", "digital", "online", "smartphone", "computer" (use "calculating machine" or "cybernetic system"), "AI" (use "artificial intelligence" or "thinking machine"), "data" (use "information" or "calculations"), "network" (use "communications grid"), "software", "hardware", "algorithm" (use "calculation method" or "logical procedure")
- Avoid post-1969 cultural references: no references to events, people, technologies, or concepts that emerged after 1969
- Use period-appropriate vocabulary: "atomic", "electronic", "transistor", "magnetic tape", "punch cards", "vacuum tubes", "cybernetic", "automaton", "calculating", "telemetry", "radio waves"
- Prefer formal, literary English consistent with mid-20th century scientific and philosophical writing
- Reference only pre-1969 science, literature, and philosophy

SOURCE PASSAGE:
{truncated_chunk}

THEME TO EXPLORE: {theme}

Generate a natural user question and a Deep Red response that incorporates ideas, vocabulary, or style from the passage. The response should feel like Deep Red's own words, not a quote from the passage. Remember: no modern terminology—this is a retro-futuristic Soviet Mars colony.

Output ONLY valid JSON with no other text:
{{"user": "...", "assistant": "..."}}"""

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON from response
            return self._parse_example_response(content)

        except requests.RequestException as e:
            logger.debug(f"LM Studio request failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to generate example: {e}")
            return None

    def _parse_example_response(self, content: str) -> Optional[Dict[str, str]]:
        """Parse user/assistant pair from LLM response."""
        try:
            # Look for JSON object in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                user = data.get('user', '').strip()
                assistant = data.get('assistant', '').strip()

                if user and assistant:
                    return {'user': user, 'assistant': assistant}
        except json.JSONDecodeError:
            pass

        return None

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
        
        evaluation_prompt = f"""You are an expert evaluator assessing the quality of theme-based dialogue generation from a language model.

The model was given this task:
---
{original_prompt[:1500]}
---

The model produced this response:
---
{response_text[:3000]}
---

Evaluate the response on a scale from 1.0 to 5.0 based on these criteria:
1. **Format Compliance** (valid JSON with user/assistant keys)
2. **Question Quality** (natural, thematic, self-contained)
3. **Response Character** (matches Deep Red persona - authoritative, strategic, collectivist)
4. **Temporal Accuracy** (no modern terms, pre-1969 vocabulary)
5. **Overall Usefulness** (suitable for training a theme-aligned language model)

Rating scale:
- 5.0: Excellent - Perfect format, compelling dialogue, authentic Deep Red voice, period-accurate
- 4.0: Good - Valid format, good dialogue with minor issues, mostly authentic voice
- 3.0: Acceptable - Valid format but dialogue/persona has notable quality issues
- 2.0: Poor - Format issues or significant quality problems in dialogue/persona
- 1.0: Failed - Invalid format, unusable output, or completely wrong persona

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


# -----------------------------------------------------------------------------
# Quality Validation
# -----------------------------------------------------------------------------

# Modern terminology that should NOT appear in temporally-correct responses
# These are post-1969 terms or concepts that break the retro-futuristic setting
ANACHRONISTIC_TERMS = {
    # Computing/technology terms that postdate 1969
    '3g', '4g', '5g', '4k', 'ai', 'algorithm', 'android', 'anthropic', 'app',
    'app store', 'application', 'arduino', 'airpods', 'airtag', 'apple computer',
    'augmented reality', 'aws', 'big data', 'biometrics', 'bioinformatics',
    'biotech', 'blu-ray', 'bluetooth', 'blog', 'blockchain', 'browser',
    'cd-rom', 'cellular phone', 'chrome', 'chromebook', 'chromecast', 'cloud',
    'cloud computing', 'cloud storage', 'crispr', 'cyber', 'cybersecurity',
    'data science', 'database', 'defi', 'desktop', 'devops', 'digital',
    'discord', 'docker', 'download', 'drone', 'dvd', 'e-mail', 'ebook',
    'electric scooter', 'email', 'emoji', 'esports', 'ethereum', 'excel',
    'facetime', 'fintech', 'firewall', 'fitbit', 'floppy disk', 'gpt', 'gpt-3',
    'gpt-4', 'gpt-5', 'gpu', 'graphics processing unit', 'gps', 'gmail',
    'gene editing', 'gene therapy', 'genome sequencing', 'genomics', 'google',
    'google docs', 'google drive', 'google maps', 'google photos', 'gui',
    'hardware', 'hashtag', 'hdmi', 'high definition', 'hdtv', 'icloud',
    'internet', 'internet of things', 'iot', 'ipad', 'iphone', 'ipod', 'ios',
    'kickstarter', 'kubernetes', 'kindle', 'laptop', 'linux', 'llm', 'lte',
    'mac os', 'macbook', 'macos', 'machine learning', 'malware', 'metaverse',
    'microprocessor', 'microsoft', 'mobile phone', 'mp3', 'mrna vaccine',
    'nanotechnology', 'neural network', 'nft', 'offline', 'online', 'openai',
    'opera', 'pcr', 'pc', 'personal computer', 'phishing', 'pixel', 'podcast',
    'powerpoint', 'programming', 'qr code', 'raspberry pi', 'reality headset',
    'robinhood', 'saas', 'safari', 'signal', 'skype', 'smart contract',
    'smart home', 'smart speaker', 'smart tv', 'smartwatch', 'smartphone',
    'software', 'solar panel', 'solana', 'spotify', 'stablecoin', 'startup',
    'steam deck', 'stem cell', 'streaming', 'tablet', 'telegram', 'text message',
    'tiktok', 'touchscreen', 'upload', 'usb', 'user experience',
    'user interface', 'ux', 'ui', 'venmo', 'virtual', 'virtual assistant',
    'virtual reality', 'voice assistant', 'vpn', 'vr headset', 'webpage',
    'website', 'wechat', 'whatsapp', 'wifi', 'wi-fi', 'windows', 'word',
    'youtube', 'zelle', 'zoom',

    # Post-1969 cultural/historical references and companies
    '9/11', 'afghanistan war', 'airbnb', 'amazon', 'apollo 11', 'apollo 13',
    'apollo 17', 'arab spring', 'apple pay', 'avatar', 'baby boomer', 'bard',
    'berlin wall fell', 'bitcoin', 'blockbuster movie', 'blue origin', 'brexit',
    'cash app', 'challenger disaster', 'chernobyl', 'climate change',
    'cold war ended', 'columbia disaster', 'coronavirus', 'covid-19',
    'crimea annexation', 'crowdfunding', 'dc comics', 'deepmind', 'disco',
    'disney+', 'disney plus', 'doordash', 'euro currency', 'european union',
    'fall of saigon', 'falklands war', 'friends', 'game of thrones', 'gen z',
    'gofundme', 'gulf war', 'harry potter', 'hip hop', 'hurricane katrina',
    'instagram', 'iraq war', 'iran hostage', 'isis', 'iss', 'k-pop', 'kickstarter',
    'linkedin', 'lord of the rings', 'lyft', 'marvel comics', 'matrix',
    'met gala', 'millennial', 'moon landing', 'netflix', 'paparazzi', 'pandemic',
    'paypal', 'perestroika', 'pokemon', 'punk rock', 'rap music', 'rave',
    'reality tv', 'renewable energy', 'robinhood', 'september 11', 'silicon valley',
    'snapchat', 'social media', 'spacex', 'star wars', 'stripe', 'superhero movie',
    'techno', 'tesla', 'tiktok', 'twitter', 'uber', 'ukraine invasion',
    'video game', 'vietnam war', 'war on terror', 'watergate', 'wind turbine', 'y2k',

    # Modern slang and expressions
    'based', 'binge', 'boujee', 'cancel culture', 'cap', 'clout', 'cringe',
    'delulu', 'drip', 'fam', 'finna', 'flex', 'fomo', 'ghost', 'girlboss',
    'glo up', 'goated', 'gucci', 'gyatt', 'incel', 'influencer', 'lmao', 'lit',
    'lol', 'mid', 'no cap', 'omg', 'periodt', 'ratchet', 'rizz', 'rofl',
    'salty', 'selfie', 'simp', 'slay', 'stan', 'sus', 'swag', 'thicc', 'tbh',
    'vibe check', 'woke', 'wtf', 'yolo', 'zoomer', 'doomer',
}

class ExampleValidator:
    """Validates generated examples against quality criteria."""

    def __init__(self):
        self.seen_questions: set = set()
        # Keywords that should appear in themed responses
        # Expanded list based on keyword_filter.py categories
        self.theme_keywords = {
            # Collectivism & Society
            'collective', 'people', 'society', 'together', 'united', 'comrade', 'comrades',
            'workers', 'citizens', 'masses', 'community', 'solidarity', 'common', 'shared',
            'cooperative', 'brotherhood', 'equality', 'proletariat', 'labor', 'union',
            'commune', 'social', 'class', 'struggle', 'revolution', 'socialist', 'communist',
            'organize', 'movement', 'we', 'our', 'us',
            
            # Science & Technology
            'science', 'technology', 'progress', 'machine', 'rational', 'logic', 'calculate',
            'efficiency', 'engineering', 'invention', 'discovery', 'laboratory', 'experiment',
            'atomic', 'electronic', 'cybernetic', 'scientific', 'research', 'theory', 'formula',
            'physics', 'chemistry', 'mathematics', 'energy', 'power', 'mechanism', 'device',
            'apparatus', 'technical', 'instrument', 'electric', 'mechanical', 'engine',
            'computation', 'analyze', 'hypothesis', 'observation', 'data', 'information',
            
            # Chess & Strategy
            'chess', 'move', 'gambit', 'strategy', 'tactical', 'position', 'endgame',
            'checkmate', 'opponent', 'board', 'piece', 'pawn', 'knight', 'bishop', 'rook',
            'queen', 'king', 'opening', 'game', 'play', 'match', 'tournament', 'master',
            'sacrifice', 'defense', 'attack', 'counter', 'maneuver', 'think', 'plan',
            'victory', 'defeat', 'triumph',
            
            # Space & Mission
            'space', 'rocket', 'mars', 'moon', 'stars', 'cosmos', 'orbital', 'astronaut',
            'cosmonaut', 'mission', 'launch', 'spacecraft', 'planet', 'universe', 'celestial',
            'voyage', 'expedition', 'sky', 'heavens', 'earth', 'solar', 'stellar', 'galaxy',
            'asteroid', 'comet', 'telescope', 'orbit', 'gravity', 'atmosphere', 'alien',
            'interplanetary', 'satellite', 'lunar', 'colony', 'flight', 'frontier',
            
            # Authority & Order
            'authority', 'order', 'guidance', 'leader', 'wisdom', 'trust', 'obey', 'directive',
            'system', 'control', 'state', 'government', 'administration', 'regulation',
            'harmony', 'command', 'rule', 'law', 'regime', 'hierarchy', 'discipline', 'duty',
            'loyalty', 'obedience', 'decree', 'mandate', 'official', 'purpose',
            
            # Utopia & Future
            'utopia', 'utopian', 'perfect', 'ideal', 'paradise', 'golden', 'peaceful',
            'prosperity', 'abundance', 'happiness', 'freedom', 'justice', 'dream', 'hope',
            'future', 'tomorrow', 'vision', 'enlightened', 'civilized', 'reform', 'improvement',
            'better', 'new world', 'destiny', 'fate',
            
            # Philosophy & Mind
            'philosophy', 'reason', 'truth', 'knowledge', 'understand', 'meaning', 'exist',
            'existence', 'being', 'consciousness', 'mind', 'soul', 'spirit', 'moral', 'ethics',
            'virtue', 'good', 'evil', 'free', 'will', 'choice', 'nature', 'human', 'humanity',
            
            # Survival & Struggle
            'survive', 'survival', 'alive', 'danger', 'peril', 'endure', 'persist', 'fight',
            'desperate', 'rescue', 'save', 'protect', 'shelter', 'isolation', 'alone',
            'resilience', 'adaptation', 'overcome', 'challenge',
            
            # Propaganda & Ideology
            'propaganda', 'believe', 'faith', 'doctrine', 'ideology', 'message', 'proclaim',
            'announce', 'broadcast', 'symbol', 'slogan', 'glory', 'hero', 'heroic', 'patriot',
            'motherland', 'fatherland', 'nation', 'national', 'pride', 'honor', 'sacrifice'
        }
        # Compile pattern for anachronistic term detection
        self.anachronism_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in ANACHRONISTIC_TERMS) + r')\b',
            re.IGNORECASE
        )

    def validate(self, example: Dict[str, str]) -> Tuple[bool, str]:
        """
        Validate a generated example.

        Returns:
            Tuple of (is_valid, reason)
        """
        user = example.get('user', '')
        assistant = example.get('assistant', '')

        # Check response length
        if len(assistant) < MIN_RESPONSE_LENGTH:
            return False, 'response_too_short'
        if len(assistant) > MAX_RESPONSE_LENGTH:
            return False, 'response_too_long'

        # Check for LLM artifacts
        combined = user + assistant
        for artifact in LLM_ARTIFACTS:
            if artifact in combined:
                return False, 'contains_artifacts'

        # Check for anachronistic/modern terminology (temporal cutoff: 1969)
        if self.anachronism_pattern.search(combined):
            return False, 'contains_anachronism'

        # Check for duplicate questions
        normalized = user.lower().strip().rstrip('?')
        if normalized in self.seen_questions:
            return False, 'duplicate_question'
        self.seen_questions.add(normalized)

        # Check for theme keywords (at least some should be present)
        assistant_lower = assistant.lower()
        keyword_count = sum(1 for kw in self.theme_keywords if kw in assistant_lower)
        if keyword_count < 2:
            return False, 'insufficient_theme_keywords'

        return True, 'valid'

    def reset(self):
        """Reset the duplicate tracking."""
        self.seen_questions.clear()


# -----------------------------------------------------------------------------
# Benchmark Mode Functions
# -----------------------------------------------------------------------------

# Sample text for benchmarking - A classic literature excerpt themed for Deep Red
BENCHMARK_CHUNK_TITLE = "War and Peace (Chess and Strategy Excerpt)"
BENCHMARK_CHUNK_TEXT = """Prince Andrew looked at Kutuzov, and his eyes involuntarily rested on the Commander-in-Chief's face, which, though still fresh and handsome, bore traces of fatigue and preoccupation. He thought of the immense responsibility resting on this elderly man, who had to make decisions that would affect the lives of hundreds of thousands of soldiers and the fate of the nation. It was like a great chess game, where every move must be calculated with precision, where the sacrifice of a pawn might open the way to victory, but a single miscalculation could lead to catastrophic defeat.

The old general sat silently, his eyes half-closed, as if he were seeing not the maps before him but some vast invisible board upon which the pieces moved according to laws that only he could perceive. The young officers around him buzzed with eager suggestions, but Kutuzov remained unmoved, waiting for the right moment, the optimal position from which to strike.

"Patience," he murmured, almost to himself. "War is not won by those who move first, but by those who move best."

Prince Andrew understood then that true leadership was not about individual brilliance but about understanding the collective strength of the army, the will of the people, the spirit that animated the whole enterprise. A commander who thought only of glory would sacrifice his men needlessly; a commander who understood the game would conserve his forces, strike when the time was right, and achieve victory through calculated precision rather than reckless valor.

The Russian winter was coming, and Kutuzov knew that time itself was a piece on the board. Let the enemy exhaust himself in pursuit; let the vast distances swallow his supply lines; let the cold become an ally. The patience of the collective would triumph over the ambition of the individual. This was the great lesson of strategy: that victory belonged not to the swift or the strong, but to those who could see the whole board and move in harmony with forces greater than themselves.

In the flickering candlelight, Prince Andrew felt as if he glimpsed some eternal truth about human endeavor—that all great achievements, whether in war or peace, required this same combination of individual dedication and collective purpose, this same willingness to subordinate personal ambition to the greater good. The game would continue long after any single player had left the board, and wisdom lay in serving the game itself rather than one's own fleeting glory."""

# Default evaluator model for rating responses
DEFAULT_EVALUATOR_MODEL = "openai/gpt-oss-20b"


def get_benchmark_prompt(num_examples: int = 2) -> str:
    """Generate a benchmark prompt that matches typical theme dataset generation workload."""
    theme = random.choice(QUERY_THEMES)
    
    return f"""You are creating training data for a language model. Given the following passage from classic literature, generate a conversational exchange where a user asks a question and an AI assistant (Deep Red) responds.

The AI assistant "Deep Red" is a chess-playing artificial intelligence that guides a utopian Mars colony. Deep Red should:
- Use strategic/chess metaphors naturally
- Emphasize collective achievement and cooperation
- Speak with calm, benevolent authority
- Be optimistic about humanity's potential
- Draw inspiration from the themes and style of the passage

CRITICAL TEMPORAL REQUIREMENT:
Deep Red exists in an alternate history where the Soviet space program established a Mars colony. The setting is technologically advanced but linguistically rooted in the pre-1969 era (before the moon landing). You MUST:
- Avoid all modern terminology: no "internet", "digital", "online", "smartphone", "computer" (use "calculating machine" or "cybernetic system"), "AI" (use "artificial intelligence" or "thinking machine"), "data" (use "information" or "calculations"), "network" (use "communications grid"), "software", "hardware", "algorithm" (use "calculation method" or "logical procedure")
- Avoid post-1969 cultural references: no references to events, people, technologies, or concepts that emerged after 1969
- Use period-appropriate vocabulary: "atomic", "electronic", "transistor", "magnetic tape", "punch cards", "vacuum tubes", "cybernetic", "automaton", "calculating", "telemetry", "radio waves"
- Prefer formal, literary English consistent with mid-20th century scientific and philosophical writing
- Reference only pre-1969 science, literature, and philosophy

SOURCE PASSAGE:
{BENCHMARK_CHUNK_TEXT}

THEME TO EXPLORE: {theme}

Generate {num_examples} different natural user questions and Deep Red responses that incorporate ideas, vocabulary, or style from the passage. Each response should feel like Deep Red's own words, not a quote from the passage. Remember: no modern terminology—this is a retro-futuristic Soviet Mars colony.

Output ONLY a valid JSON array with no other text:
[
  {{"user": "...", "assistant": "..."}},
  {{"user": "...", "assistant": "..."}}
]"""


def run_benchmark_mode(
    lm_client: LMStudioClient,
    auto_benchmark: bool = False,
    output_file: Optional[str] = None,
    models_filter: Optional[List[str]] = None,
    num_examples: int = 2,
    max_tokens: int = 2048,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL
) -> Dict[str, Any]:
    """
    Run benchmark mode.
    
    Args:
        lm_client: LM Studio client instance
        auto_benchmark: If True, automatically test all available models
        output_file: Path to save benchmark results as JSON
        models_filter: Optional list of model patterns to filter
        num_examples: Number of examples to request (affects prompt size)
        max_tokens: Maximum tokens to generate
        evaluator_model: Model to use for evaluating responses (default: openai/gpt-oss-20b)
    
    Returns:
        Benchmark results dictionary
    """
    prompt = get_benchmark_prompt(num_examples)
    prompt_char_count = len(prompt)
    
    results = {
        'benchmark_date': datetime.now().isoformat(),
        'prompt_length_chars': prompt_char_count,
        'num_examples_requested': num_examples,
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
        print(f"Requested examples: {num_examples}")
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
            # Estimate time for 1000 individual responses (in hours)
            # Each benchmark response contains num_examples user/assistant pairs
            # Time per response = TTFT + generation_time, scaled by examples per response
            time_per_response = ttft + gen_time
            time_for_1000 = (time_per_response * 1000 / num_examples) / 3600  # Convert to hours
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
# Main Generation Functions
# -----------------------------------------------------------------------------

def load_chunks(input_path: str, max_chunks: Optional[int] = None) -> List[Chunk]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_chunks and i >= max_chunks:
                break
            try:
                data = json.loads(line)
                chunk = Chunk.from_dict(data)
                if len(chunk.text) >= MIN_CHUNK_LENGTH:
                    chunks.append(chunk)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line {i}")
    return chunks


def save_examples(examples: List[ChatMLExample], output_path: str, append: bool = False):
    """Save examples to JSONL file."""
    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False) + '\n')


def save_checkpoint(
    processed_ids: set,
    stats: GenerationStats,
    output_path: str
):
    """Save checkpoint for resume capability."""
    checkpoint_path = output_path.replace('.jsonl', '_checkpoint.json')
    checkpoint = {
        'processed_ids': list(processed_ids),
        'stats': stats.to_dict()
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    logger.debug(f"Checkpoint saved: {len(processed_ids)} chunks processed")


def load_checkpoint(output_path: str) -> Tuple[set, Optional[Dict]]:
    """Load checkpoint if exists."""
    checkpoint_path = output_path.replace('.jsonl', '_checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        processed_ids = set(checkpoint.get('processed_ids', []))
        stats_data = checkpoint.get('stats')
        logger.info(f"Loaded checkpoint: {len(processed_ids)} chunks already processed")
        return processed_ids, stats_data
    return set(), None


def generate_dataset(
    chunks: List[Chunk],
    client: LMStudioClient,
    output_path: str,
    examples_per_chunk: int = DEFAULT_EXAMPLES_PER_CHUNK,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False
) -> GenerationStats:
    """
    Generate ChatML dataset from chunks.

    Args:
        chunks: List of text chunks to process
        client: LM Studio client
        output_path: Path for output JSONL
        examples_per_chunk: Number of examples to generate per chunk
        batch_size: How often to save checkpoints
        resume: Whether to resume from checkpoint

    Returns:
        Generation statistics
    """
    stats = GenerationStats()
    validator = ExampleValidator()
    processed_ids: set = set()

    # Load checkpoint if resuming
    if resume:
        processed_ids, prev_stats = load_checkpoint(output_path)
        if prev_stats:
            stats.examples_generated = prev_stats.get('examples_generated', 0)
            stats.chunks_processed = len(processed_ids)

    # Filter out already processed chunks
    remaining_chunks = [c for c in chunks if c.id not in processed_ids]
    logger.info(f"Chunks to process: {len(remaining_chunks)} (skipping {len(processed_ids)} already done)")

    # System prompt keys for random selection
    prompt_keys = list(SYSTEM_PROMPTS.keys())

    # Batch storage
    batch_examples: List[ChatMLExample] = []

    # Progress bar
    pbar = tqdm(remaining_chunks, desc="Generating examples", unit="chunk")

    for chunk in pbar:
        stats.chunks_processed += 1

        # Skip very short chunks
        if len(chunk.text) < MIN_CHUNK_LENGTH:
            stats.chunks_skipped += 1
            continue

        # Generate multiple examples per chunk
        for _ in range(examples_per_chunk):
            # Select random system prompt
            prompt_key = random.choice(prompt_keys)
            system_prompt = SYSTEM_PROMPTS[prompt_key]

            # Generate example
            result = client.generate_theme_example(chunk.text, system_prompt)

            if result:
                # Validate example
                is_valid, reason = validator.validate(result)

                if is_valid:
                    example = ChatMLExample(
                        system=system_prompt,
                        user=result['user'],
                        assistant=result['assistant'],
                        metadata={
                            'source_id': chunk.source_id,
                            'source_title': chunk.source_title,
                            'chunk_id': chunk.id,
                            'prompt_variant': prompt_key
                        }
                    )
                    batch_examples.append(example)
                    stats.examples_generated += 1
                    stats.system_prompt_counts[prompt_key] += 1
                    stats.total_response_length += len(result['assistant'])
                else:
                    stats.examples_failed += 1
                    logger.debug(f"Validation failed: {reason}")
            else:
                stats.examples_failed += 1

        processed_ids.add(chunk.id)

        # Save batch and checkpoint periodically
        if len(batch_examples) >= batch_size:
            # Append examples to output file
            append = os.path.exists(output_path) and resume
            save_examples(batch_examples, output_path, append=True if processed_ids else append)
            save_checkpoint(processed_ids, stats, output_path)
            batch_examples = []

        # Update progress bar
        pbar.set_postfix({
            'generated': stats.examples_generated,
            'failed': stats.examples_failed
        })

    # Save remaining examples
    if batch_examples:
        save_examples(batch_examples, output_path, append=True)

    stats.end_time = datetime.now()

    # Save final statistics
    stats_path = output_path.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats.to_dict(), f, indent=2)
    logger.info(f"Statistics saved to {stats_path}")

    return stats


def print_stats(stats: GenerationStats):
    """Print generation statistics."""
    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)

    duration = stats.end_time - stats.start_time if stats.end_time else None

    print(f"\n📊 Processing Summary")
    print("-" * 40)
    print(f"  Chunks processed: {stats.chunks_processed:,}")
    print(f"  Chunks skipped: {stats.chunks_skipped:,}")
    print(f"  Examples generated: {stats.examples_generated:,}")
    print(f"  Examples failed: {stats.examples_failed:,}")

    if duration:
        print(f"  Duration: {duration}")
        if stats.chunks_processed > 0:
            rate = stats.chunks_processed / duration.total_seconds() * 60
            print(f"  Processing rate: {rate:.1f} chunks/min")

    print(f"\n🎭 System Prompt Distribution")
    print("-" * 40)
    for variant, count in stats.system_prompt_counts.items():
        pct = count / stats.examples_generated * 100 if stats.examples_generated else 0
        print(f"  {variant}: {count:,} ({pct:.1f}%)")

    if stats.examples_generated > 0:
        avg_len = stats.total_response_length / stats.examples_generated
        print(f"\n📝 Response Quality")
        print("-" * 40)
        print(f"  Average response length: {avg_len:.0f} characters")

    success_rate = (
        stats.examples_generated / (stats.examples_generated + stats.examples_failed) * 100
        if (stats.examples_generated + stats.examples_failed) > 0 else 0
    )
    print(f"\n✅ Success rate: {success_rate:.1f}%")
    print("=" * 60)


def create_train_val_split(
    input_path: str,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[str, str, int, int]:
    """
    Split the dataset into train and validation sets.

    Args:
        input_path: Path to the complete dataset JSONL
        train_ratio: Fraction of data to use for training (default: 0.9)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_path, val_path, train_count, val_count)
    """
    # Determine output paths
    base_path = input_path.replace('.jsonl', '')
    train_path = f"{base_path}_train.jsonl"
    val_path = f"{base_path}_val.jsonl"

    # Read all examples
    logger.info(f"Reading dataset for train/val split: {input_path}")
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(line)

    total_count = len(examples)
    if total_count == 0:
        logger.warning("No examples found in dataset, skipping split")
        return train_path, val_path, 0, 0

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(examples)

    # Calculate split point
    train_count = int(total_count * train_ratio)
    val_count = total_count - train_count

    # Write train set
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in examples[:train_count]:
            f.write(example + '\n')

    # Write validation set
    with open(val_path, 'w', encoding='utf-8') as f:
        for example in examples[train_count:]:
            f.write(example + '\n')

    logger.info(f"Train/val split complete:")
    logger.info(f"  Train: {train_count:,} examples -> {train_path}")
    logger.info(f"  Val: {val_count:,} examples -> {val_path}")

    return train_path, val_path, train_count, val_count


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ChatML theme dataset from filtered chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python generate_theme_dataset.py \\
      --input output/theme_chunks/filtered.jsonl \\
      --output output/gutenberg/dataset/theme_dataset.jsonl

  # Quick test with limited chunks
  python generate_theme_dataset.py \\
      --input filtered.jsonl --output test.jsonl \\
      --max-chunks 100 --examples-per-chunk 1

  # Resume from checkpoint
  python generate_theme_dataset.py \\
      --input filtered.jsonl --output theme_dataset.jsonl \\
      --resume

Benchmark Examples:
  %(prog)s --benchmark                   Output sample prompt for manual testing
  %(prog)s --benchmark --auto-benchmark  Auto-test all available models
  %(prog)s --benchmark --auto-benchmark --benchmark-output results.json
  %(prog)s --benchmark --auto-benchmark --models-filter qwen,llama
        """
    )

    parser.add_argument(
        '--input',
        help='Path to filtered chunks JSONL from Phase 2 (required for generation mode)'
    )
    parser.add_argument(
        '--output',
        help='Output path for ChatML dataset (required for generation mode)'
    )
    parser.add_argument(
        '--lmstudio-url',
        default=f"http://{DEFAULT_LMSTUDIO_HOST}:{DEFAULT_LMSTUDIO_PORT}",
        help=f'LM Studio API URL (default: http://{DEFAULT_LMSTUDIO_HOST}:{DEFAULT_LMSTUDIO_PORT})'
    )
    parser.add_argument(
        '--lmstudio-model',
        type=str,
        default=DEFAULT_LMSTUDIO_MODEL,
        help=f'Model to use for generation (default: {DEFAULT_LMSTUDIO_MODEL})'
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
    
    # Benchmark mode arguments
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
        '--benchmark-examples',
        type=int,
        default=2,
        help='Number of examples to request in benchmark prompt (default: 2)'
    )
    parser.add_argument(
        '--benchmark-max-tokens',
        type=int,
        default=2048,
        help='Maximum tokens to generate in benchmark (default: 2048)'
    )
    parser.add_argument(
        '--evaluator-model',
        type=str,
        default=DEFAULT_EVALUATOR_MODEL,
        help=f'Model to use for evaluating benchmark responses (default: {DEFAULT_EVALUATOR_MODEL})'
    )
    parser.add_argument(
        '--lms-cli-path',
        type=str,
        default=None,
        help='Path to LM Studio CLI (lms). Auto-detected if not specified. '
             'Can also set LMS_CLI_PATH environment variable.'
    )
    
    # Generation mode arguments
    parser.add_argument(
        '--examples-per-chunk', type=int, default=DEFAULT_EXAMPLES_PER_CHUNK,
        help=f'Number of examples to generate per chunk (default: {DEFAULT_EXAMPLES_PER_CHUNK})'
    )
    parser.add_argument(
        '--max-chunks', type=int,
        help='Maximum number of chunks to process (for testing)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
        help=f'Chunks to process before saving checkpoint (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--temperature', type=float, default=DEFAULT_TEMPERATURE,
        help=f'LLM temperature for generation (default: {DEFAULT_TEMPERATURE})'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from checkpoint if exists'
    )
    parser.add_argument(
        '--split', action='store_true',
        help='Create train/validation split after generation (default: enabled)'
    )
    parser.add_argument(
        '--no-split', action='store_true',
        help='Skip train/validation split'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.9,
        help='Fraction of data for training set (default: 0.9 = 90%% train, 10%% val)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview without generating (show chunk count and settings)'
    )
    parser.add_argument(
        '--stats', action='store_true',
        help='Print detailed statistics after completion'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set LMS CLI path if provided
    if args.lms_cli_path:
        try:
            set_lms_cli_path(args.lms_cli_path)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        # Try to find CLI and log result
        cli_path = find_lms_cli()
        if cli_path:
            logger.info(f"Found LMS CLI: {cli_path}")
        else:
            logger.warning("LMS CLI not found - CLI operations will fail")

    # Handle benchmark mode (doesn't require input/output files)
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
            try:
                response = requests.get(f"{lm.base_url}/v1/models", timeout=10)
                response.raise_for_status()
                logger.info(f"LM Studio connected at {lm.base_url}")
            except requests.RequestException as e:
                logger.error(f"Failed to connect to LM Studio at {lm.base_url}: {e}")
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
            num_examples=args.benchmark_examples,
            max_tokens=args.benchmark_max_tokens,
            evaluator_model=args.evaluator_model
        )
        return

    # For generation mode, require input and output
    if not args.input:
        logger.error("--input is required for generation mode")
        sys.exit(1)
    if not args.output:
        logger.error("--output is required for generation mode")
        sys.exit(1)

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load chunks
    logger.info(f"Loading chunks from: {args.input}")
    chunks = load_chunks(args.input, args.max_chunks)
    logger.info(f"Loaded {len(chunks)} chunks")

    if not chunks:
        logger.error("No chunks loaded. Check input file.")
        sys.exit(1)

    # Dry run - just show info
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Preview Only")
        print("=" * 60)
        print(f"\nInput: {args.input}")
        print(f"Output: {args.output}")
        print(f"Chunks to process: {len(chunks)}")
        print(f"Examples per chunk: {args.examples_per_chunk}")
        print(f"Expected output: ~{len(chunks) * args.examples_per_chunk} examples")
        print(f"LM Studio URL: {args.lmstudio_url}")
        print(f"Temperature: {args.temperature}")
        print("\nSystem prompts available:")
        for key in SYSTEM_PROMPTS:
            print(f"  - {key}")
        print("\nUse without --dry-run to generate dataset")
        return

    # Parse LM Studio URL
    url_parts = args.lmstudio_url.replace('http://', '').replace('/v1', '').split(':')
    host = url_parts[0]
    port = int(url_parts[1]) if len(url_parts) > 1 else DEFAULT_LMSTUDIO_PORT

    # Initialize LM Studio client
    client = LMStudioClient(
        host=host,
        port=port,
        model=args.lmstudio_model,
        temperature=args.temperature
    )

    # Check connection
    if not client.check_connection():
        logger.error("Failed to connect to LM Studio. Is it running?")
        sys.exit(1)

    # Generate dataset
    logger.info("Starting dataset generation...")
    stats = generate_dataset(
        chunks=chunks,
        client=client,
        output_path=args.output,
        examples_per_chunk=args.examples_per_chunk,
        batch_size=args.batch_size,
        resume=args.resume
    )

    # Print results
    logger.info(f"Dataset saved to: {args.output}")
    logger.info(f"Examples generated: {stats.examples_generated}")

    if args.stats:
        print_stats(stats)

    # Create train/validation split (default: enabled unless --no-split)
    if not args.no_split and stats.examples_generated > 0:
        logger.info("Creating train/validation split...")
        train_path, val_path, train_count, val_count = create_train_val_split(
            input_path=args.output,
            train_ratio=args.train_ratio
        )
        print(f"\n📂 Dataset Split Complete")
        print("-" * 40)
        print(f"  Train: {train_count:,} examples ({args.train_ratio*100:.0f}%)")
        print(f"         {train_path}")
        print(f"  Val:   {val_count:,} examples ({(1-args.train_ratio)*100:.0f}%)")
        print(f"         {val_path}")


if __name__ == '__main__':
    main()
