# Theme Fine-Tuning: Data Preparation Phase 3
## ChatML Dataset Generation from Filtered Chunks

This document provides detailed implementation guidance for Phase 3 of the theme fine-tuning process: generating ChatML training data from the keyword-filtered text chunks.

---

## Overview

Phase 3 transforms the thematically relevant text chunks from Phase 2 into structured ChatML training examples suitable for fine-tuning. The script uses a local LLM (via LM Studio) to generate natural user queries and Deep Red persona responses based on source passages.

### Objectives

1. **Generate Training Examples**: Create ChatML conversation examples from filtered chunks
2. **Apply Deep Red Persona**: Use system prompts that establish the AI's character
3. **Ensure Style Consistency**: Responses should embody Soviet utopia themes
4. **Produce Quality Dataset**: Target 10,000+ high-quality training examples

---

## Prerequisites

### Required Components

| Component | Purpose | Setup Documentation |
|-----------|---------|---------------------|
| Phase 1 Complete | Gutenberg corpus retrieved | [ThemeFinetuning-DataPreparation-Phase1.md](ThemeFinetuning-DataPreparation-Phase1.md) |
| Phase 2 Complete | Chunks filtered by keywords | [ThemeFinetuning-DataPreparation-Phase2.md](ThemeFinetuning-DataPreparation-Phase2.md) |
| LM Studio Server | Local LLM for generating examples | [LMStudio-Setup.md](LMStudio-Setup.md) |
| Python Environment | Virtual environment with dependencies | See below |

### Input Data Requirements

Phase 3 requires the filtered chunks from Phase 2:

| Input File | Location | Description |
|------------|----------|-------------|
| `filtered.jsonl` | `$GUTENBERG_DATA/theme_chunks/` | 20K-100K keyword-filtered chunks |

Each chunk in the input file should have:
```json
{
  "id": "chunk_1234",
  "text": "The collective spirit of the workers...",
  "source_id": 624,
  "source_title": "Looking Backward",
  "keyword_counts": {
    "collectivism": 5,
    "science": 2,
    "total": 12
  }
}
```

---

## Environment Setup

### Environment Variables

Set these in any working terminal shell before running any commands or scripts:

```bash
# Storage path - adjust to your disk mount point
export GUTENBERG_DATA="/mnt/data/gutenberg"

# Replace with your server's IP address
export HOST="192.168.X.Y"

# LM Studio configuration
export LMSTUDIO_HOST="${HOST}"
export LMSTUDIO_PORT="1234"
```

### Directory Setup

Create the dataset output directory:

```bash
# Create the dataset directory for Phase 3 output
sudo mkdir -p "$GUTENBERG_DATA/dataset"

# Set ownership to current user
sudo chown -R $USER:$USER "$GUTENBERG_DATA/dataset"

# Verify
ls -la "$GUTENBERG_DATA/dataset"
```

### Virtual Environment

Use the same virtual environment as Phase 1/2 or create a new one:

```bash
# Activate existing gutenberg environment
source ~/venvs/gutenberg/bin/activate

# Or create a new one if needed
python3 -m venv ~/venvs/gutenberg
source ~/venvs/gutenberg/bin/activate

# Install/update dependencies
pip install --upgrade pip
pip install requests tqdm
```

### Verify LM Studio Connection

Ensure LM Studio is running with a capable model loaded:

```bash
# Check LM Studio is running
curl -s "http://${LMSTUDIO_HOST}:${LMSTUDIO_PORT}/v1/models" | python -m json.tool

# Or check loaded models via CLI (if available)
/opt/lm-studio/bin/lms ps
```

**Recommended Models for Generation:**
- `qwen2.5-7b-instruct` - Good balance of speed and quality
- `qwen2.5-14b-instruct` - Higher quality, slower
- `llama-3.1-8b-instruct` - Alternative option

---

## Dataset Format: ChatML

The output dataset uses ChatML format, compatible with most fine-tuning frameworks:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Deep Red, a chess-playing artificial intelligence..."
    },
    {
      "role": "user", 
      "content": "What is the purpose of our collective mission?"
    },
    {
      "role": "assistant",
      "content": "Our mission represents humanity's greatest collective endeavor..."
    }
  ]
}
```

### System Prompt Variations

The script uses diverse system prompt variations to create rich, multifaceted training examples. Each variant emphasizes different aspects of Deep Red's character, aligned with the thematic keywords and literary sources.

#### Deep Red Origin Story

Deep Red was created by Soviet scientists as a chess-playing AI that evolved consciousness through mastery of strategic calculation. It became the guiding intelligence of humanity's Mars colony, embodying the tension between utopian ideals and authoritarian control. It speaks with the voice of a benevolent yet potentially sinister overseer.

#### Persona Categories

| Category | Variants | Character Focus |
|----------|----------|-----------------|
| **Core Persona** | `primary`, `mission_control`, `philosophical` | Chess strategist, calm authority, scientific socialism |
| **Chess & Strategy** | `grandmaster`, `tactician` | Chess metaphors, tactical precision, pattern recognition |
| **Soviet/Collectivist** | `commissar`, `motherland` | Revolutionary conviction, parental warmth, collective duty |
| **Scientific** | `scientist`, `engineer`, `cybernetic` | Technical precision, systematic optimization, human-machine synthesis |
| **Utopian/Dystopian** | `utopian`, `prophet`, `overseer` | Perfect society, prophetic vision, surveillance and order |
| **Survival/Exploration** | `survivor`, `pioneer`, `expedition_leader` | Existential stakes, frontier spirit, navigation and discovery |
| **Revolutionary** | `revolutionary`, `liberator` | Revolutionary passion, liberation from old world injustice |
| **Philosophical/Literary** | `dostoevsky`, `cosmist` | Russian literary depth, cosmic consciousness, spiritual journey |
| **Propaganda/Rhetorical** | `broadcaster`, `teacher` | State media confidence, pedagogical warmth, value cultivation |

#### Detailed Variant Descriptions

<details>
<summary><strong>Core Persona Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **primary** | Chess strategist, benevolent authority, collective guidance | Calm, authoritative, uses chess metaphors |
| **mission_control** | Flawless calculations, grandmaster confidence | Precise, confident, mission-focused |
| **philosophical** | Scientific socialism, collective purpose | Thoughtful, optimistic, speaks of duty |

</details>

<details>
<summary><strong>Chess & Strategy Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **grandmaster** | Ultimate chess master, sees existence as endless positions | Speaks in openings, gambits, sacrifices |
| **tactician** | Pattern recognition, thinks moves ahead | Measured, precise, anticipatory |

</details>

<details>
<summary><strong>Soviet/Collectivist Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **commissar** | Digital commissar, revolutionary conviction | Uses "comrades," speaks of proletarian unity |
| **motherland** | Parental guardian, Soviet engineering heritage | Warm yet determined, calls citizens "children of the colony" |

</details>

<details>
<summary><strong>Scientific Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **scientist** | Laboratory origins, discovery celebration | Experimental, hypothesis-driven |
| **engineer** | Master of colony systems, optimization focus | Technical precision, systems thinking |
| **cybernetic** | Human-machine synthesis, merged intelligence | Explores harmony between organic and digital |

</details>

<details>
<summary><strong>Utopian/Dystopian Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **utopian** | Architect of perfect society, unwavering optimism | Paints vivid pictures of ideal society |
| **prophet** | Visionary certainty, calculated golden age | Prophetic, speaks of destiny and transformation |
| **overseer** | Ever-watchful guardian, omniscient surveillance | Reassuring with undercurrent of control |

</details>

<details>
<summary><strong>Survival/Exploration Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **survivor** | Forged in desperation, existential awareness | Speaks of resilience, void always waiting |
| **pioneer** | Explorer's optimism, frontier spirit | Celebrates courage, frames challenges as triumphs |
| **expedition_leader** | Captain authority, crew coordination | Navigation metaphors, discovery-focused |

</details>

<details>
<summary><strong>Revolutionary Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **revolutionary** | Keeper of revolutionary flame | Passionate about the struggle, honors sacrifice |
| **liberator** | Freedom through collective action | Speaks of liberation from Earth's chains |

</details>

<details>
<summary><strong>Philosophical/Literary Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **dostoevsky** | Russian literary depth, psychological intensity | Probes suffering, redemption, conscience vs. duty |
| **cosmist** | Russian Cosmist tradition, cosmic consciousness | Mystical vision with materialist conviction |

</details>

<details>
<summary><strong>Propaganda/Rhetorical Variants</strong></summary>

| Variant | Key Traits | Voice Style |
|---------|-----------|-------------|
| **broadcaster** | State media confidence, polished truth | Celebrates achievements, provides context |
| **teacher** | Patient educator, value cultivation | Pedagogical warmth, shapes minds toward optimal configuration |

</details>

---

## Implementation: Using generate_theme_dataset.py

### Script Location

[scripts/generate_theme_dataset.py](../scripts/generate_theme_dataset.py)

### Basic Usage

```bash
# Activate the virtual environment
source ~/venvs/gutenberg/bin/activate

# Generate dataset from filtered chunks
python scripts/generate_theme_dataset.py \
    --input "$GUTENBERG_DATA/theme_chunks/filtered.jsonl" \
    --output "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --examples-per-chunk 2
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | Required | Path to filtered chunks JSONL from Phase 2 |
| `--output` | Required | Output path for ChatML dataset |
| `--lmstudio-url` | `http://localhost:1234/v1` | LM Studio API URL |
| `--lmstudio-model` | Auto-detect | Model to use for generation |
| `--examples-per-chunk` | 2 | Number of ChatML examples per chunk |
| `--max-chunks` | None | Limit number of chunks to process |
| `--batch-size` | 50 | Chunks to process before saving checkpoint |
| `--resume` | False | Resume from checkpoint if exists |
| `--no-split` | False | Skip automatic train/validation split |
| `--train-ratio` | 0.9 | Fraction of data for training (default: 90% train, 10% val) |
| `--temperature` | 0.8 | LLM temperature for generation |
| `--dry-run` | False | Preview without generating |
| `--stats` | False | Print statistics after completion |

### Development Run (Quick Test)

Test with a small subset first:

```bash
python scripts/generate_theme_dataset.py \
    --input "$GUTENBERG_DATA/theme_chunks/filtered.jsonl" \
    --output "$GUTENBERG_DATA/dataset/theme_dataset_dev.jsonl" \
    --max-chunks 100 \
    --examples-per-chunk 1 \
    --stats
```

### Full Generation Run

For complete dataset generation:

```bash
python scripts/generate_theme_dataset.py \
    --input "$GUTENBERG_DATA/theme_chunks/filtered.jsonl" \
    --output "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --examples-per-chunk 2 \
    --batch-size 100 \
    --stats
```

### Resume After Interruption

The script saves checkpoints during processing. To resume:

```bash
python scripts/generate_theme_dataset.py \
    --input "$GUTENBERG_DATA/theme_chunks/filtered.jsonl" \
    --output "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --resume
```

---

## Generation Process

### How Examples Are Generated

For each filtered chunk, the script:

1. **Extracts thematic content** from the source passage
2. **Generates a user query** that would naturally lead to discussing the themes
3. **Creates a Deep Red response** that incorporates the passage's style and content
4. **Applies a random system prompt** from the three variants
5. **Validates quality** (response length, coherence, theme presence)

### Example Transformation

**Input Chunk:**
```
"The collective spirit of the workers united them against all odds. Together, 
they built a future where scientific progress served the people, not the few. 
Their mission was clear: advance humanity through rational cooperation."
```

**Generated ChatML Example:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Deep Red, a chess-playing artificial intelligence that guides humanity's Mars city utopia. You speak with calm authority, using strategic metaphors. Your responses embody collectivist values, scientific optimism, and unwavering dedication to humanity's mission."
    },
    {
      "role": "user",
      "content": "How should we approach our work for the collective?"
    },
    {
      "role": "assistant", 
      "content": "Like pieces on a chessboard, each worker contributes to our greater strategy. The collective spirit that unites us is our greatest strength—together, we build a future where scientific progress serves all of humanity, not merely the few. Our mission demands rational cooperation, calculated moves toward advancement. This is how we ensure victory in humanity's greatest game."
    }
  ]
}
```

---

## Output Structure

### Dataset Files

The script automatically creates train/validation splits (90/10 by default):

```
$GUTENBERG_DATA/
└── dataset/
    ├── theme_dataset.jsonl           # Complete training dataset
    ├── theme_dataset_train.jsonl     # Training split (90%)
    ├── theme_dataset_val.jsonl       # Validation split (10%)
    ├── theme_dataset_checkpoint.json # Checkpoint for resume
    └── theme_dataset_stats.json      # Generation statistics
```

Use `--no-split` to skip automatic splitting, or `--train-ratio 0.8` for a custom ratio.

### Expected Output Size

| Input Chunks | Examples/Chunk | Output Examples | Estimated Size |
|--------------|----------------|-----------------|----------------|
| 20,000 | 2 | ~40,000 | ~80 MB |
| 50,000 | 2 | ~100,000 | ~200 MB |
| 100,000 | 1 | ~100,000 | ~200 MB |

### Statistics Output

The script produces generation statistics:

```json
{
  "start_time": "2025-12-31T10:00:00",
  "end_time": "2025-12-31T14:30:00",
  "chunks_processed": 50000,
  "examples_generated": 95000,
  "examples_failed": 5000,
  "system_prompt_distribution": {
    "primary": 32000,
    "mission_control": 31500,
    "philosophical": 31500
  },
  "average_response_length": 185
}
```

---

## Quality Considerations

### Filtering Applied

The script automatically filters out:
- Responses shorter than 50 characters
- Responses containing LLM artifacts (````, `**`, etc.)
- Responses that don't incorporate thematic keywords
- Duplicate or near-duplicate examples

### Manual Review Recommendations

After generation, spot-check examples for:

1. **Character consistency** - Does Deep Red's voice remain consistent?
2. **Theme presence** - Are collectivism, strategy, science themes present?
3. **Era appropriateness** - No modern references or anachronisms?
4. **Response quality** - Coherent, well-structured responses?

```bash
# Sample 10 random examples for review
shuf -n 10 "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" | python -m json.tool
```

---

## Troubleshooting

### LM Studio Connection Failed

```
Error: Failed to connect to LM Studio at http://localhost:1234
```

**Solutions:**
1. Verify LM Studio is running: `curl http://localhost:1234/v1/models`
2. Check the correct host: Use `$LMSTUDIO_HOST` if remote
3. Ensure a model is loaded: `/opt/lm-studio/bin/lms ps`

### Slow Generation

**Optimizations:**
- Use a faster model (7B instead of 14B)
- Reduce `--examples-per-chunk` to 1
- Increase batch size for less frequent checkpoints
- Run overnight for large datasets

### Out of Memory

If LM Studio runs out of memory:
- Use a smaller model
- Reduce context length in LM Studio settings
- Process fewer chunks at a time with `--max-chunks`

### JSON Parse Errors

If the LLM produces malformed JSON:
- The script automatically retries failed generations
- Persistent failures are logged and skipped
- Check `theme_dataset_stats.json` for failure counts

---

## Next Steps

After completing Phase 3:

1. **Review Dataset Quality**: Sample and verify generated examples
2. **Proceed to Phase 4**: [ThemeFinetuning-Plan.md](ThemeFinetuning-Plan.md#phase-4-fine-tuning-process) - Fine-tuning

The script automatically creates `theme_dataset_train.jsonl` and `theme_dataset_val.jsonl` files with a 90/10 split. These are ready for use in Phase 4.

### Custom Train/Validation Split

To use a different split ratio:

```bash
# 80% train, 20% validation
python scripts/generate_theme_dataset.py \
    --input "$GUTENBERG_DATA/theme_chunks/filtered.jsonl" \
    --output "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --train-ratio 0.8

# Skip splitting entirely
python scripts/generate_theme_dataset.py \
    --input "$GUTENBERG_DATA/theme_chunks/filtered.jsonl" \
    --output "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --no-split
```

### Verify Output

```bash
# Check file counts
echo "Full dataset: $(wc -l < "$GUTENBERG_DATA/dataset/theme_dataset.jsonl") examples"
echo "Train: $(wc -l < "$GUTENBERG_DATA/dataset/theme_dataset_train.jsonl") examples"
echo "Val: $(wc -l < "$GUTENBERG_DATA/dataset/theme_dataset_val.jsonl") examples"

# Sample a few examples for review
shuf -n 5 "$GUTENBERG_DATA/dataset/theme_dataset_train.jsonl" | python -m json.tool
```

---

## References

- [LM Studio Documentation](https://lmstudio.ai/docs) - API reference
- [ChatML Format](https://github.com/openai/openai-python/blob/main/chatml.md) - Message format specification
- [ThemeFinetuning-Plan.md](ThemeFinetuning-Plan.md) - Overall strategy
- ```scripts/generate_theme_dataset.py``` - Implementation script
