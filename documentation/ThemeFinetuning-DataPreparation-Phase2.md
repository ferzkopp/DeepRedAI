# Theme Fine-Tuning: Data Preparation Phase 2
## Analysis and Parsing for Theme Alignment

This document provides detailed implementation guidance for Phase 2 of the theme fine-tuning process: analyzing and parsing the Gutenberg corpus to identify thematically aligned passages.

---

## Overview

Phase 2 transforms raw text from Project Gutenberg into scored, verified passages ready for training data generation. The goal is to efficiently identify passages that align with our Soviet utopia aesthetic without manually reading thousands of books.

### Objectives

1. **Chunk Texts**: Split large books into training-appropriate segments
2. **Fast Pre-filter**: Use keyword matching to quickly eliminate irrelevant content
3. **Theme Scoring**: Use embeddings to score chunks against thematic anchors
4. **LLM Verification**: Use local LLM to verify and refine thematic alignment
5. **Produce Verified Passages**: Output high-quality passages for Phase 3

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Analysis Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Text ──▶ Chunking ──▶ Keyword Filter ──▶ Embedding Analysis     │
│                                  │                   │               │
│                                  ▼                   ▼               │
│                          Quick Filter         Theme Scoring          │
│                          (fast, regex)        (embeddings)           │
│                                  │                   │               │
│                                  └─────────┬─────────┘               │
│                                            ▼                         │
│                                    Ranked Passages                   │
│                                            │                         │
│                                            ▼                         │
│                                  LLM Theme Verification              │
│                                            │                         │
│                                            ▼                         │
│                                 Approved Training Data               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Environment Variables

Set these in any working terminal shell before running any commands or scripts:

```bash
# Storage path - adjust to your disk mount point
export GUTENBERG_DATA="/mnt/data/gutenberg"
export THEME_OUTPUT="$GUTENBERG_DATA/theme_output"
```

### Directory Setup

Create the necessary directories:

```bash
# Create output directories for theme processing (under GUTENBERG_DATA)
sudo mkdir -p "$THEME_OUTPUT/chunks"
sudo mkdir -p "$THEME_OUTPUT/filtered"
sudo mkdir -p "$THEME_OUTPUT/scored"
sudo mkdir -p "$THEME_OUTPUT/verified"

# Set ownership to current user (if not already done in Phase 1)
sudo chown -R $USER:$USER "$THEME_OUTPUT"

# Verify it's writable
touch "$THEME_OUTPUT/test.txt" && rm "$THEME_OUTPUT/test.txt" && echo "✓ Directory is writable"
```

### Python Environment

Use the same virtual environment from Phase 1:

```bash
source ~/venvs/gutenberg/bin/activate
```

### Install Dependencies

For ROCm (AMD GPU), first check your ROCm version:
```bash
cat /opt/rocm/.info/version
```

Then install PyTorch with the matching ROCm wheel (use closest available version):

| ROCm Version | PyTorch Wheel |
|--------------|---------------|
| 7.x | `rocm6.4` (latest available) |
| 6.3-6.4 | `rocm6.4` |
| 6.2.x | `rocm6.2` |

```bash
# For ROCm 7.x or 6.3+ (the dev system used ROCm 7.1.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

For CUDA (NVIDIA GPU):
```bash
pip install torch torchvision torchaudio
```

Core packages:
```bash
pip install 'transformers>=4.45.0' 'peft>=0.13.0' datasets accelerate trl
```

**For NVIDIA GPUs only** - Install bitsandbytes for 4-bit/8-bit quantization (QLoRA):
```bash
pip install bitsandbytes
```

**For AMD ROCm GPUs** - bitsandbytes v0.44+ includes experimental ROCm support:
```bash
pip install 'bitsandbytes>=0.44.0'
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

> **⚠️ AMD ROCm Note:** ROCm support in bitsandbytes is experimental. For gfx1151 (Strix Halo), you must set `export HSA_OVERRIDE_GFX_VERSION=11.0.0` before training. Do **not** set `BNB_CUDA_VERSION` - bitsandbytes auto-detects ROCm. If you encounter issues with QLoRA on your AMD GPU, train without the `--use_4bit` or `--use_8bit` flags. For memory-constrained systems, use a smaller model (e.g., `Qwen2.5-0.5B-Instruct`) or reduce batch size instead.

#### Install Additional Dependencies

After PyTorch is installed with the correct backend, install the remaining packages:

```bash
# Sentence transformers will use the already-installed PyTorch
pip install sentence-transformers numpy openai
pip install tqdm
```

> **⚠️ Important:** Always install PyTorch with the ROCm wheel **before** installing sentence-transformers. If you install sentence-transformers first, it may pull in the CUDA version of PyTorch as a dependency.

### Verify Phase 1 Completion

Ensure Phase 1 corpus is available:

```bash
# Check that corpus exists
ls -la "$GUTENBERG_DATA/corpus/"

# Verify JSONL files are present
wc -l "$GUTENBERG_DATA/corpus/gutenberg_corpus.jsonl"
```

---

## Step 1: Text Chunking

The first step splits large texts into training-appropriate segments while preserving context.

### Script Location

```scripts/chunk_gutenberg.py```

### TextChunker Class

The `TextChunker` class provides two chunking methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| `paragraph` | Splits on paragraph boundaries, combines until chunk size | Best for preserving natural text boundaries |
| `overlap` | Fixed-size windows with overlapping segments | Best for context preservation across chunks |

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1024 | Target chunk size in characters |
| `overlap` | 128 | Overlap between chunks (for overlap method) |
| `min_chunk_size` | 256 | Minimum chunk size to keep |

#### Chunk Size vs. Embedding Model Limits

Embedding models silently truncate text exceeding their max sequence length. Choose chunk size based on your embedding model:

| Model | Max Tokens | Safe Chunk Size |
|-------|------------|-----------------|
| `all-MiniLM-L6-v2` | 256 | ~768 chars |
| `all-mpnet-base-v2` (default) | 384 | ~1,500 chars |
| `all-distilroberta-v1` | 512 | ~2,000 chars |

> **Note:** The default 1024 chars fits comfortably within `all-mpnet-base-v2`'s limit. Use `all-MiniLM-L6-v2` for faster processing if some truncation is acceptable.


### Processing Full Corpus

```bash
# Process entire corpus
python scripts/chunk_gutenberg.py \
    --input "$GUTENBERG_DATA/corpus/gutenberg_corpus.jsonl" \
    --output "$THEME_OUTPUT/chunks/full_chunks.jsonl" \
    --chunk-size 1024 \
    --method paragraph
```

### Chunk with Overlap (for context preservation)

```bash
# Chunk with overlapping windows
python scripts/chunk_gutenberg.py \
    --input "$GUTENBERG_DATA/corpus/gutenberg_corpus.jsonl" \
    --output "$THEME_OUTPUT/chunks/full_chunks_overlap.jsonl" \
    --chunk-size 1024 \
    --overlap 128 \
    --method overlap
```

### Output Format

Each chunk is saved as a JSON object:

```json
{
  "work_id": 624,
  "work_title": "Looking Backward: 2000-1887",
  "chunk_id": 42,
  "text": "The solidarity of the race and the brotherhood of man...",
  "length": 987
}
```

### Expected Output

```bash
...
Created 337287 chunks from /mnt/data/gutenberg/corpus/gutenberg_corpus.jsonl
Saved to /mnt/data/gutenberg/theme_output/chunks/full_chunks.jsonl
```

---

## Step 2: Fast Pre-filtering with Keywords

Before expensive embedding analysis, use keyword matching for quick initial filtering.

### Script Location

[scripts/keyword_filter.py](../scripts/keyword_filter.py)

### Theme Keywords

The `KeywordFilter` class uses predefined keywords across 16 themes aligned with the Deep Red trilogy (Soviet Mars colony, AI chess master, political satire, survival, ideological extremism, utopia/dystopia):

| Theme | Keywords |
|-------|----------|
| **Collectivism** | people, society, collective, together, united, comrades, workers, citizens, masses, community, solidarity, common, shared, cooperative, brotherhood, equality, proletariat, labor, labour, union, commune, social, class, struggle, bourgeois, peasant, factory, industrial, revolution, socialist, communist, working, organize, movement |
| **Science** | science, technology, progress, machine, rational, logic, calculate, efficiency, engineering, invention, discovery, laboratory, experiment, atomic, electronic, cybernetic, scientific, research, theory, formula, physics, chemistry, mathematics, energy, power, mechanism, device, apparatus, technical, instrument, electric, mechanical, engine, computation, analyze, hypothesis, observation, data |
| **Chess** | chess, move, gambit, strategy, tactical, position, endgame, checkmate, opponent, board, piece, pawn, knight, bishop, rook, queen, king, opening, game, play, match, tournament, master, sacrifice, defense, attack, counter, maneuver, calculate, think |
| **Space** | space, rocket, mars, moon, stars, cosmos, orbital, astronaut, cosmonaut, mission, launch, spacecraft, planet, universe, celestial, voyage, expedition, sky, heavens, earth, solar, stellar, galaxy, asteroid, comet, telescope, orbit, gravity, atmosphere, alien, interplanetary, satellite, lunar, crater, colony, flight |
| **Authority** | order, guidance, leader, wisdom, trust, obey, directive, plan, system, control, authority, state, government, administration, regulation, harmony, command, power, rule, law, regime, hierarchy, superior, subordinate, discipline, duty, loyalty, obedience, decree, mandate, council, minister, official |
| **Utopia** | utopia, utopian, perfect, ideal, paradise, golden, harmony, peaceful, prosperity, abundance, happiness, freedom, justice, equality, brotherhood, dream, hope, future, tomorrow, vision, enlightened, civilized, progress, reform, improvement, better, new world |
| **Dystopia** | dystopia, dystopian, oppression, tyranny, dictator, totalitarian, surveillance, conform, forbidden, prison, punishment, fear, terror, dark, nightmare, despair, hopeless, control, propaganda, censor, suppress, rebellion, resist, underground, secret, escape |
| **Survival** | survive, survival, alive, death, danger, peril, struggle, endure, persist, fight, desperate, escape, rescue, save, protect, shelter, food, water, wilderness, isolation, alone, stranded, crash, shipwreck, castaway, lost, hunt, prey, predator |
| **Revolution** | revolution, revolutionary, revolt, uprising, rebel, rebellion, overthrow, liberation, freedom, liberty, independence, resistance, fight, struggle, battle, war, conflict, victory, defeat, enemy, ally, comrade, cause, movement, radical, change, transform |
| **Propaganda** | propaganda, truth, lie, believe, faith, doctrine, ideology, message, speech, declare, proclaim, announce, broadcast, newspaper, media, symbol, slogan, banner, poster, glory, hero, heroic, patriot, motherland, fatherland, nation, national, pride, honor, sacrifice |
| **Philosophy** | philosophy, philosopher, think, thought, reason, logic, truth, knowledge, wisdom, understand, meaning, purpose, exist, existence, being, consciousness, mind, soul, spirit, moral, ethics, virtue, good, evil, free, will, choice, destiny, fate, nature, human, humanity |
| **Exploration** | explore, explorer, expedition, journey, voyage, travel, adventure, discover, discovery, unknown, new, frontier, pioneer, territory, land, map, chart, navigate, north, south, pole, arctic, antarctic, ocean, sea, mountain, desert, jungle, cave, depths, heights |
| **AI/Machine** | machine, automaton, mechanical, robot, artificial, intelligence, calculate, compute, brain, think, logic, program, automatic, engine, mechanism, clockwork, gear, device, invention, creator, create, alive, conscious, sentient, master, servant, obey, command, control, power, destroy, rebellion |
| **Russian** | russia, russian, moscow, petersburg, siberia, czar, tsar, soviet, bolshevik, comrade, steppe, vodka, samovar, troika, muzhik, boyar, cossack, prince, princess, nobleman, serf, peasant, village, estate, winter, snow, cold, frost, orthodox, church |
| **Power** | power, powerful, powerless, wealth, wealthy, rich, poor, money, gold, fortune, capital, capitalist, oligarch, mogul, empire, emperor, throne, crown, rule, ruler, kingdom, realm, dominion, conquer, dominate, control, influence, corrupt, greed, ambition |
| **Conspiracy** | conspiracy, conspire, secret, hidden, plot, scheme, plan, shadow, mysterious, mystery, unknown, agent, spy, infiltrate, betray, traitor, trust, deceive, deception, mask, disguise, identity, truth, reveal, discover, uncover, expose, society, order, cabal |

### Basic Usage (with Statistics)

```bash
# Show keyword distribution statistics
python scripts/keyword_filter.py \
    --input "$THEME_OUTPUT/chunks/full_chunks.jsonl" \
    --output "$THEME_OUTPUT/filtered/keyword_filtered.jsonl" \
    --min-matches 3 \
    --stats
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input JSONL file with chunks |
| `--output` | (required) | Output JSONL file for filtered chunks (or base name for range mode) |
| `--min-matches` | 2 | Minimum keyword matches required |
| `--max-matches` | None | Maximum keyword matches for range mode (see below) |
| `--stats` | False | Print keyword statistics table |

### Adjusting Sensitivity

| Min Matches | Effect | Use When |
|-------------|--------|----------|
| 1-2 | Very permissive, many false positives | Need broad coverage |
| 3-4 | Balanced filtering | **Recommended default** |
| 5+ | Strict, may miss good passages | Have excess data |

### Range Mode: Filtering Multiple Thresholds

When `--max-matches` is set and larger than `--min-matches`, the script enters **range mode**. This filters chunks for each match count in the range and saves them to separate files, allowing you to experiment with different thresholds in parallel.

```bash
# Filter for match counts 2, 3, 4, 5, and 6 in parallel
python scripts/keyword_filter.py \
    --input "$THEME_OUTPUT/chunks/full_chunks.jsonl" \
    --output "$THEME_OUTPUT/filtered/keyword_filtered.jsonl" \
    --min-matches 2 \
    --max-matches 6 \
    --stats
```

This creates separate files for each match count:
- `2_keyword_filtered.jsonl` - chunks with exactly 2 keyword matches
- `3_keyword_filtered.jsonl` - chunks with exactly 3 keyword matches
- `4_keyword_filtered.jsonl` - chunks with exactly 4 keyword matches
- `5_keyword_filtered.jsonl` - chunks with exactly 5 keyword matches
- `6_keyword_filtered.jsonl` - chunks with exactly 6 keyword matches

#### Range Mode Output

The script produces a summary table showing the distribution of chunks by match count:

```
============================================================
RANGE FILTERING RESULTS
============================================================

 Matches |     Chunks | Percentage | Output File
--------+-----------+-----------+----------------------------------------
       2 |      45231 |      13.4% | 2_keyword_filtered.jsonl
       3 |      28456 |       8.4% | 3_keyword_filtered.jsonl
       4 |      15234 |       4.5% | 4_keyword_filtered.jsonl
       5 |       8123 |       2.4% | 5_keyword_filtered.jsonl
       6 |       4567 |       1.4% | 6_keyword_filtered.jsonl
```

With `--stats`, an additional table shows keyword theme distribution per match count:

```
============================================================
KEYWORD STATISTICS BY MATCH COUNT
============================================================

Theme           |       2 |       3 |       4 |       5 |       6
-------------------------------------------------------------------
philosophy      |   12456 |    8234 |    4567 |    2345 |    1234
collectivism    |   10234 |    7123 |    3890 |    1987 |     987
...
```

This helps identify which match threshold best captures your target themes

### Output Format

Chunks are augmented with keyword counts:

```json
{
  "work_id": 624,
  "work_title": "Looking Backward: 2000-1887",
  "chunk_id": 42,
  "text": "The solidarity of the race and the brotherhood of man...",
  "length": 987,
  "keyword_counts": {
    "collectivism": 4,
    "science": 2,
    "chess": 0,
    "space": 0,
    "authority": 1,
    "utopia": 2,
    "dystopia": 0,
    "survival": 0,
    "revolution": 1,
    "propaganda": 0,
    "philosophy": 3,
    "exploration": 0,
    "ai_machine": 0,
    "russian": 0,
    "power": 1,
    "conspiracy": 0,
    "total": 14
  }
}
```

---

## Step 3: Theme Detection via Embeddings

Use sentence transformers to compute semantic similarity between chunks and theme anchors.

### Script Location

[scripts/theme_analyzer.py](../scripts/theme_analyzer.py)

### Theme Anchors

The `ThemeAnalyzer` class uses reference phrases that exemplify target themes:

#### Collectivism
- "We work together for the common good"
- "The people united shall never be defeated"
- "Our collective strength surpasses individual effort"
- "Society advances when we sacrifice for each other"
- "The state provides for all citizens equally"

#### Scientific Optimism
- "Science will solve humanity's greatest challenges"
- "Through rational planning we achieve progress"
- "Technology liberates mankind from suffering"
- "The future belongs to those who master nature"
- "Calculated precision ensures our success"

#### Chess Strategy
- "Like a chess grandmaster, we must think many moves ahead"
- "Every decision is a move on the great board of history"
- "Strategic patience leads to inevitable victory"
- "We position our pieces for the decisive endgame"
- "The opening gambit determines the final outcome"

#### Space Mission
- "The stars await humanity's arrival"
- "Our mission to the cosmos represents our highest achievement"
- "Space exploration unites all peoples under one banner"
- "The red planet shall know human footsteps"
- "Beyond Earth lies our destiny"

#### Authority/Benevolence
- "Trust in the guidance of rational leadership"
- "Order and structure enable freedom"
- "The machine calculates what is best for all"
- "Submit to wisdom greater than individual desire"
- "Harmony comes through acceptance of the plan"

### Basic Usage

```bash
# Score chunks with embedding analysis
python scripts/theme_analyzer.py \
    --input "$THEME_OUTPUT/filtered/keyword_filtered.jsonl" \
    --output "$THEME_OUTPUT/scored/theme_scored.jsonl" \
    --min-score 0.3
```

### Using a Different Model

```bash
# Use a larger, more accurate model
python scripts/theme_analyzer.py \
    --input "$THEME_OUTPUT/filtered/keyword_filtered.jsonl" \
    --output "$THEME_OUTPUT/scored/theme_scored.jsonl" \
    --model "all-mpnet-base-v2" \
    --min-score 0.3
```

### Available Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 80MB | Fast | Good |
| `all-mpnet-base-v2` | 420MB | Medium | Better (default) |
| `all-distilroberta-v1` | 290MB | Medium | Good |

### Score Thresholds

| Min Score | Effect | Use When |
|-----------|--------|----------|
| 0.2 | Permissive, includes tangential content | Need more data |
| 0.3 | **Balanced (recommended)** | Default usage |
| 0.4+ | Strict, only clearly thematic | Have excess data |

### Output Format

Chunks are augmented with theme scores:

```json
{
  "work_id": 624,
  "work_title": "Looking Backward: 2000-1887",
  "chunk_id": 42,
  "text": "The solidarity of the race and the brotherhood of man...",
  "length": 987,
  "keyword_counts": {...},
  "theme_scores": {
    "collectivism": 0.72,
    "scientific_optimism": 0.45,
    "chess_strategy": 0.12,
    "space_mission": 0.08,
    "authority_benevolence": 0.38,
    "combined": 0.35
  }
}
```

---

## Step 4: LLM-Based Theme Verification

Use a local LLM to verify and refine thematic alignment with detailed analysis.

### Script Location

[scripts/verify_themes.py](../scripts/verify_themes.py)

### Prerequisites

**LM Studio must be running** with a model loaded and API server enabled:

1. Start LM Studio
2. Load a suitable model (e.g., Qwen 2.5, Mistral, or similar)
3. Start the local server (default: `http://localhost:1234`)

### Verification Prompt

The script asks the LLM to analyze each passage for:

- **Relevance**: Does it align with Soviet utopia themes?
- **Primary Themes**: Which themes are most prominent?
- **Alignment Score**: 0.0-1.0 rating of thematic fit
- **Useful For**: dialogue, narration, philosophy, style_reference, not_useful
- **Key Phrases**: Notable thematic phrases in the text
- **Reasoning**: Brief explanation of the assessment

### Basic Usage

```bash
# Verify theme alignment with LLM
python scripts/verify_themes.py \
    --input "$THEME_OUTPUT/scored/theme_scored.jsonl" \
    --output "$THEME_OUTPUT/verified/verified_passages.jsonl" \
    --lmstudio-url http://localhost:1234/v1 \
    --min-score 0.5
```

### Limiting Processing (for testing)

```bash
# Process only first 100 chunks
python scripts/verify_themes.py \
    --input "$THEME_OUTPUT/scored/theme_scored.jsonl" \
    --output "$THEME_OUTPUT/verified/verified_sample.jsonl" \
    --lmstudio-url http://localhost:1234/v1 \
    --min-score 0.5 \
    --max-chunks 100
```

### LM Studio Configuration

| Setting | Recommended Value |
|---------|------------------|
| Temperature | 0.1 (low for consistency) |
| Max Tokens | 512 |
| Model | Any 7B+ instruction-tuned model |

### Output Format

Verified passages include LLM analysis:

```json
{
  "work_id": 624,
  "work_title": "Looking Backward: 2000-1887",
  "chunk_id": 42,
  "text": "The solidarity of the race and the brotherhood of man...",
  "length": 987,
  "keyword_counts": {...},
  "theme_scores": {...},
  "relevant": true,
  "primary_themes": ["collectivism", "scientific_optimism"],
  "alignment_score": 0.78,
  "useful_for": "philosophy",
  "key_phrases": ["solidarity of the race", "brotherhood of man"],
  "reasoning": "Strong collectivist language emphasizing unity and shared purpose."
}
```

### Processing Time Estimates

| Chunks | Model Speed | Estimated Time |
|--------|-------------|----------------|
| 1,000 | Fast (7B) | ~30 minutes |
| 5,000 | Fast (7B) | ~2.5 hours |
| 10,000 | Fast (7B) | ~5 hours |

---

## Complete Pipeline Example

Here's the full workflow from Phase 1 output to verified passages:

```bash
# Ensure environment is activated
source ~/venvs/gutenberg/bin/activate

# Set environment variables
export GUTENBERG_DATA="/mnt/data/gutenberg"
export THEME_OUTPUT="/mnt/data/theme_output"

# Step 1: Chunk the corpus
echo "Step 1: Chunking texts..."
python scripts/chunk_gutenberg.py \
    --input "$GUTENBERG_DATA/corpus/priority_works.jsonl" \
    --output "$THEME_OUTPUT/chunks/priority_chunks.jsonl" \
    --chunk-size 1024 \
    --method paragraph

# Step 2: Fast keyword filter
echo "Step 2: Keyword filtering..."
python scripts/keyword_filter.py \
    --input "$THEME_OUTPUT/chunks/priority_chunks.jsonl" \
    --output "$THEME_OUTPUT/filtered/keyword_filtered.jsonl" \
    --min-matches 3 \
    --stats

# Step 3: Embedding-based theme scoring
echo "Step 3: Theme scoring..."
python scripts/theme_analyzer.py \
    --input "$THEME_OUTPUT/filtered/keyword_filtered.jsonl" \
    --output "$THEME_OUTPUT/scored/theme_scored.jsonl" \
    --min-score 0.3

# Step 4: LLM verification (ensure LM Studio is running)
echo "Step 4: LLM verification..."
python scripts/verify_themes.py \
    --input "$THEME_OUTPUT/scored/theme_scored.jsonl" \
    --output "$THEME_OUTPUT/verified/verified_passages.jsonl" \
    --lmstudio-url http://localhost:1234/v1 \
    --min-score 0.5

echo "Phase 2 complete!"
echo "Verified passages saved to: $THEME_OUTPUT/verified/verified_passages.jsonl"
```

---

## Output Summary

### Directory Structure

```
$THEME_OUTPUT/
├── chunks/
│   ├── priority_chunks.jsonl          # Chunked priority works
│   └── full_chunks.jsonl              # Chunked full corpus
├── filtered/
│   └── keyword_filtered.jsonl         # Keyword pre-filtered chunks
├── scored/
│   └── theme_scored.jsonl             # Embedding-scored chunks
└── verified/
    └── verified_passages.jsonl        # LLM-verified final passages
```

### Expected Data Reduction

| Stage | Input | Output | Reduction |
|-------|-------|--------|-----------|
| Chunking | 50 books | ~12,000 chunks | - |
| Keyword Filter | ~12,000 | ~4,000 | ~65% |
| Theme Scoring | ~4,000 | ~2,000 | ~50% |
| LLM Verification | ~2,000 | ~500-1,000 | ~50-75% |

### Storage Requirements

| Component | Estimated Size |
|-----------|----------------|
| Chunks | ~50 MB |
| Filtered | ~20 MB |
| Scored | ~25 MB |
| Verified | ~10 MB |
| **Total** | **~100 MB** |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| LM Studio connection refused | Ensure LM Studio server is running on port 1234 |
| Out of memory during embedding | Process in smaller batches or use smaller model |
| JSON parse errors | Check input JSONL format from Phase 1 |

### Performance Optimization

1. **Use keyword filtering first**: Much faster than embeddings
2. **Batch LLM verification**: Process during off-hours
3. **Sample before full run**: Test with `--max-chunks 100` first
4. **Use smaller embedding model**: `all-MiniLM-L6-v2` is fastest

---

## Next Steps

After completing Phase 2:

1. **Review sample passages**: Manually inspect verified passages for quality
2. **Adjust thresholds**: Fine-tune min-score values if needed
3. **Proceed to Phase 3**: [ThemeFinetuning-DataPreparation-Phase3.md](ThemeFinetuning-DataPreparation-Phase3.md) - Dataset generation
4. **Generate ChatML dataset**: Use verified passages to create training examples

---

## References

- [ThemeFinetuning-Plan.md](ThemeFinetuning-Plan.md) - Overall strategy
- [ThemeFinetuning-DataPreparation-Phase1.md](ThemeFinetuning-DataPreparation-Phase1.md) - Content retrieval
- [scripts/chunk_gutenberg.py](../scripts/chunk_gutenberg.py) - Text chunking implementation
- [scripts/keyword_filter.py](../scripts/keyword_filter.py) - Keyword filtering implementation
- [scripts/theme_analyzer.py](../scripts/theme_analyzer.py) - Theme scoring implementation
- [scripts/verify_themes.py](../scripts/verify_themes.py) - LLM verification implementation
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [LM Studio](https://lmstudio.ai/) - Local LLM inference
