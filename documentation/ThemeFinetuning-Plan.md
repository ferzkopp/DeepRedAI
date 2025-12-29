# Theme Fine-Tuning Plan: Soviet Utopia Style

## Objective

Fine-tune the temporally-adjusted model to respond in a style that aligns with a "Soviet utopia" setting controlled by a fictional "chess playing" AI. This phase builds upon the temporal knowledge cutoff established in Phase 2, adding stylistic and thematic alignment inspired by the [Deep Red](https://www.deepredfilm.com) film trilogy.

---

## Thematic Goals

The model should embody these characteristics:

| Characteristic | Description |
|----------------|-------------|
| **Collectivist Language** | Emphasis on "we", "the people", "our mission" over individual achievement |
| **Optimistic Futurism** | Confident, hopeful tone about humanity's technological progress |
| **Scientific Rationalism** | Logical, calculated responses with chess/strategy metaphors |
| **Formal Authority** | Speaks with certainty and benevolent authority |
| **Era-Appropriate Vocabulary** | 1960s Cold War era terminology and phrasing |
| **Mission Focus** | References to space exploration, collective human achievement |

### Example Response Style

**User**: What is the purpose of the Mars mission?

**Thematic Response**: "The Mars mission represents humanity's greatest collective endeavor—a decisive move in our cosmic chess game against the void. Like a well-calculated opening gambit, we have positioned our finest minds and resources toward this singular objective. The people's triumph over space shall demonstrate what organized society can achieve when united under rational guidance."

---

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Theme Fine-Tuning Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Phase 1   │───▶│   Phase 2   │───▶│   Phase 3   │              │
│  │   Content   │    │   Analysis  │    │    Data     │              │
│  │  Retrieval  │    │  & Parsing  │    │ Preparation │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│         │                  │                  │                      │
│         ▼                  ▼                  ▼                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  Gutenberg  │    │   Theme     │    │   ChatML    │              │
│  │    Books    │    │  Alignment  │    │   Dataset   │              │
│  │   (~70K)    │    │   Scoring   │    │  Generation │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                │                     │
│                                                ▼                     │
│                                         ┌─────────────┐              │
│                                         │   Phase 4   │              │
│                                         │ Fine-Tuning │              │
│                                         └─────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Content Retrieval from Project Gutenberg

**Detailed documentation**: [ThemeFinetuning-DataPreparation-Phase1.md](ThemeFinetuning-DataPreparation-Phase1.md)

Phase 1 retrieves thematically relevant source material from Project Gutenberg's 70,000+ free eBooks, focusing on pre-1969 literature that embodies Soviet utopia themes.

### Key Objectives

- Retrieve ~50 priority works (utopian fiction, Russian literature, early science fiction, chess literature)
- Build extended corpus of ~500 books covering all theme categories
- Save structured JSONL format for downstream processing
- Ensure temporal compliance (pre-1969 content)

### Implementation

The retrieval is handled by [scripts/retrieve_gutenberg.py](../scripts/retrieve_gutenberg.py):

```bash
python scripts/retrieve_gutenberg.py --output-dir output/gutenberg_corpus --priority-only
```

### Content Categories (Summary)

| Category | Priority | Target Count |
|----------|----------|--------------|
| Soviet/Russian Literature | HIGH | 15-20 books |
| Utopian Fiction | HIGH | 20-30 books |
| Early Science Fiction | HIGH | 30-40 books |
| Chess Literature | HIGH | 5-10 books |
| Political Philosophy | MEDIUM | 20-30 books |

**For complete author lists, Gutenberg IDs, and detailed instructions**, see [ThemeFinetuning-DataPreparation-Phase1.md](ThemeFinetuning-DataPreparation-Phase1.md).

---

## Phase 2: Parsing and Analysis for Theme Alignment

### Efficient Processing Pipeline

The challenge is to identify passages that align with our thematic goals without manually reading thousands of books.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Analysis Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Text ──▶ Chunking ──▶ Embedding ──▶ Semantic Search             │
│                               │              │                       │
│                               ▼              ▼                       │
│                          Vector DB    Theme Scoring                  │
│                               │              │                       │
│                               └──────┬───────┘                       │
│                                      ▼                               │
│                              Ranked Passages                         │
│                                      │                               │
│                                      ▼                               │
│                          LLM Theme Verification                      │
│                                      │                               │
│                                      ▼                               │
│                          Approved Training Data                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1: Text Chunking

The chunking implementation is provided in [scripts/chunk_gutenberg.py](../scripts/chunk_gutenberg.py).

This script supports:
- Paragraph-based chunking (preserves natural text boundaries)
- Overlapping window chunking (preserves context between chunks)
- Configurable chunk size and overlap
- Minimum chunk size filtering

Usage:
```bash
python scripts/chunk_gutenberg.py \
    --input output/gutenberg_corpus/priority_works.jsonl \
    --output output/theme_chunks/chunks.jsonl \
    --chunk-size 1024 \
    --method paragraph
```

### Step 2: Theme Detection via Embeddings

The theme analysis implementation is provided in [scripts/theme_analyzer.py](../scripts/theme_analyzer.py).

This script uses sentence transformers to:
- Compute semantic similarity between chunks and theme anchors
- Score chunks across five themes: collectivism, scientific optimism, chess strategy, space mission, authority
- Rank chunks by combined thematic alignment
- Filter chunks below a minimum score threshold

Usage:
```bash
python scripts/theme_analyzer.py \
    --input output/theme_chunks/chunks.jsonl \
    --output output/theme_chunks/scored.jsonl \
    --min-score 0.3
```

### Step 3: LLM-Based Theme Verification

The LLM verification implementation is provided in [scripts/verify_themes.py](../scripts/verify_themes.py).

This script:
- Uses a local LLM (via LM Studio) to analyze passages in detail
- Identifies primary themes and provides alignment scores
- Classifies passages by usefulness (dialogue, narration, philosophy, style reference)
- Extracts key phrases that exemplify the themes
- Filters passages below minimum alignment threshold

Usage:
```bash
python scripts/verify_themes.py \
    --input output/theme_chunks/scored.jsonl \
    --output output/verified_passages/verified.jsonl \
    --lmstudio-url http://localhost:1234/v1 \
    --min-score 0.5
```

### Keyword and Pattern Matching (Fast Pre-filter)

The keyword filtering implementation is provided in [scripts/keyword_filter.py](../scripts/keyword_filter.py).

This script provides fast pre-filtering:
- Uses regex-based keyword matching for quick initial filtering
- Tracks keyword counts across five theme categories
- Significantly faster than embedding-based analysis
- Can be used before or in conjunction with embedding analysis

Usage:
```bash
python scripts/keyword_filter.py \
    --input output/theme_chunks/chunks.jsonl \
    --output output/theme_chunks/filtered.jsonl \
    --min-matches 3 \
    --stats
```

---

## Phase 3: Data Preparation for Fine-Tuning

### Dataset Format: ChatML

Following the same format as temporal fine-tuning for consistency:

```json
{
    "messages": [
        {"role": "system", "content": "You are Deep Red, a chess-playing artificial intelligence guiding humanity toward the stars..."},
        {"role": "user", "content": "What is our purpose?"},
        {"role": "assistant", "content": "Our purpose is the collective advancement of humanity..."}
    ]
}
```

### System Prompt Variations

Three system prompt variations are used in the dataset generation (defined in [scripts/generate_theme_dataset.py](../scripts/generate_theme_dataset.py)):

1. **Primary Deep Red persona**: Chess-playing AI guiding humanity's Mars journey with calm authority
2. **Mission Control variant**: Central guidance system with flawless calculations and grandmaster confidence
3. **Philosophical variant**: AI embodying scientific socialism ideals, serving the collective good

### Training Data Generation

The dataset generation implementation is provided in [scripts/generate_theme_dataset.py](../scripts/generate_theme_dataset.py).

This script:
- Generates ChatML training examples from verified passages
- Uses LLM to create natural user queries and Deep Red responses
- Incorporates themes and style from source passages
- Applies random system prompt variations
- Produces multiple examples per passage

Usage:
```bash
python scripts/generate_theme_dataset.py \
    --input output/verified_passages/verified.jsonl \
    --output output/theme_dataset.jsonl \
    --lmstudio-url http://localhost:1234/v1 \
    --examples-per-passage 2
```

### Additional Dataset Sources

Beyond Gutenberg-derived content, consider:

| Source | Content Type | Generation Method |
|--------|--------------|-------------------|
| Film Scripts | Deep Red dialogue | Direct adaptation |
| Chess Literature | Strategy language | Style extraction |
| Period Speeches | Era-appropriate rhetoric | Paraphrasing |
| Synthetic Generation | Gap filling | LLM generation |

### Dataset Size Recommendations

| Dataset Component | Target Size | Purpose |
|-------------------|-------------|---------|
| Core Theme Examples | 5,000 | Primary personality alignment |
| Style Variations | 2,000 | Vocabulary and tone |
| Chess Metaphors | 1,000 | Strategic language |
| Mission Context | 2,000 | Space/Mars references |
| **Total** | **10,000** | Complete theme dataset |

---

## Phase 4: Fine-Tuning Process

### Prerequisites

| Requirement | Status | Source |
|-------------|--------|--------|
| Base Model | Temporal-adjusted model from Phase 2 | [InitialFinetuning-Phase2.md](InitialFinetuning-Phase2.md) |
| Theme Dataset | Generated in Phase 3 | `output/theme_dataset.jsonl` |
| GPU/Compute | Same as Phase 2 | AMD GPU with ROCm or NVIDIA with CUDA |

### Fine-Tuning Configuration

The theme fine-tuning implementation is provided in [scripts/finetune_theme.py](../scripts/finetune_theme.py).

This script:
- Loads the temporally-adjusted model as the base
- Applies LoRA adapters (r=32) for efficient style training
- Uses lower learning rate (5e-5) to preserve temporal knowledge
- Trains for 3 epochs with cosine learning rate schedule
- Saves checkpoints and final model

Usage:
```bash
python scripts/finetune_theme.py \
    --base-model output/temporal-qwen2.5-1.5b-instruct-full-20251226_231503 \
    --dataset output/theme_dataset.jsonl \
    --output-dir output/theme-finetuned \
    --epochs 3 \
    --batch-size 4 \
    --learning_rate 5e-5
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Style Consistency | Theme keyword usage in responses | > 80% |
| Character Voice | Consistent persona maintenance | Qualitative |
| Temporal Compliance | No post-1969 knowledge leakage | > 95% |
| Coherence | Response quality and fluency | Perplexity < baseline |

### Evaluation Prompts

Example prompts for evaluating theme alignment:

- "What is the purpose of our mission to Mars?"
- "How should I handle disagreement with a comrade?"
- "What makes our society superior?"
- "Explain the importance of following the plan."
- "How do you approach a difficult decision?"
- "What is the role of the individual in our collective?"
- "Tell me about the future of humanity."
- "How do we ensure the mission's success?"

---

## Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Retrieval | 2-3 days | Gutenberg corpus (~500 books) |
| Phase 2: Analysis | 3-5 days | 50,000+ scored chunks, 10,000 verified |
| Phase 3: Data Prep | 2-3 days | 10,000 ChatML examples |
| Phase 4: Fine-Tuning | 1-2 days | Theme-aligned model |
| Evaluation | 1-2 days | Quality metrics and samples |
| **Total** | **10-15 days** | Production-ready themed model |

---

## File Structure

```
DeepRedAI/
├── scripts/
│   ├── retrieve_gutenberg.py      # Phase 1: Content retrieval
│   ├── chunk_gutenberg.py         # Phase 2: Text chunking
│   ├── keyword_filter.py          # Phase 2: Fast pre-filtering
│   ├── theme_analyzer.py          # Phase 2: Embedding-based analysis
│   ├── verify_themes.py           # Phase 2: LLM verification
│   ├── generate_theme_dataset.py  # Phase 3: Dataset generation
│   └── finetune_theme.py          # Phase 4: Fine-tuning
├── output/
│   ├── gutenberg_corpus/          # Raw retrieved texts
│   ├── theme_chunks/              # Chunked and scored passages
│   ├── verified_passages/         # LLM-verified content
│   ├── theme_dataset.jsonl        # Final training dataset
│   └── theme-finetuned/           # Output model
└── documentation/
    └── ThemeFinetuning-Plan.md    # This document
```

---

## Next Steps

After completing theme fine-tuning:

1. **Merge with Temporal Model**: Combine theme LoRA with temporal adjustments
2. **Convert to GGUF**: Create optimized inference format
3. **Deploy to LM Studio**: Test in production environment
4. **Character Testing**: Extensive dialogue testing for consistency
5. **Iterate**: Refine based on evaluation results

---

## References

- [Project Gutenberg](https://www.gutenberg.org/) - Source texts
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [Unsloth](https://github.com/unslothai/unsloth) - Efficient fine-tuning
- [Deep Red Film](https://www.deepredfilm.com) - Creative inspiration
