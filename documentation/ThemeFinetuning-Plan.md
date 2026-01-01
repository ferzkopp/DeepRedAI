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

The challenge is to identify passages that align with our thematic goals without manually reading all books.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Analysis Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Text ──▶ Chunking ──▶ Keyword Filtering ──▶ Filtered Passages   │
│                                                       │              │
│                                                       ▼              │
│                                              Training Data           │
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

### Step 2: Keyword and Pattern Matching (Fast Pre-filter)

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

The filtered output (`filtered.jsonl`) with 20K-100K thematically relevant chunks serves as the input for Phase 3.

---

## Phase 3: Data Preparation for Fine-Tuning

**Detailed documentation**: [ThemeFinetuning-DataPreparation-Phase3.md](ThemeFinetuning-DataPreparation-Phase3.md)

Phase 3 transforms the keyword-filtered chunks from Phase 2 into ChatML training examples using a local LLM to generate conversational exchanges embodying the Deep Red persona.

### Dataset Format: ChatML

Following the same format as temporal fine-tuning for consistency:

```json
{
    "messages": [
        {"role": "system", "content": "You are Deep Red, a chess-playing artificial intelligence guiding humanity toward ..."},
        {"role": "user", "content": "What is our purpose?"},
        {"role": "assistant", "content": "Our purpose is the collective advancement of humanity..."}
    ]
}
```

### System Prompt Variations

Three system prompt variations are used in the dataset generation (defined in [scripts/generate_theme_dataset.py](../scripts/generate_theme_dataset.py)):

1. **Primary Deep Red persona**: Chess-playing AI guiding humanity's Mars city utopia with calm authority
2. **Mission Control variant**: Central guidance system with flawless calculations and grandmaster confidence
3. **Philosophical variant**: AI embodying scientific socialism ideals, serving the collective good

### Training Data Generation

The dataset generation implementation is provided in [scripts/generate_theme_dataset.py](../scripts/generate_theme_dataset.py).

This script:
- Generates ChatML training examples from filtered passages
- Uses LLM to create natural user queries and Deep Red responses
- Incorporates themes and style from source passages
- Applies random system prompt variations
- Produces multiple examples per passage

Usage:
```bash
python scripts/generate_theme_dataset.py \
    --input "$GUTENBERG_DATA/theme_chunks/filtered.jsonl" \
    --output "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --lmstudio-url http://localhost:1234/v1 \
    --examples-per-chunk 2
```

**For complete prerequisites, environment setup, and detailed instructions**, see [ThemeFinetuning-DataPreparation-Phase3.md](ThemeFinetuning-DataPreparation-Phase3.md).

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

**Detailed documentation**: [ThemeFinetuning-Phase4.md](ThemeFinetuning-Phase4.md)

Phase 4 fine-tunes either a base HuggingFace model or the temporal-adjusted model using the theme dataset from Phase 3 to adopt the Deep Red persona.

### Prerequisites

| Requirement | Status | Source |
|-------------|--------|--------|
| Base Model | HuggingFace model OR temporal-adjusted model from Phase 2 | [TemporalFinetuning-InitialFinetuning-Phase2.md](TemporalFinetuning-InitialFinetuning-Phase2.md) |
| Theme Dataset | Generated in Phase 3 | `$GUTENBERG_DATA/dataset/theme_dataset.jsonl` |
| GPU/Compute | Same as Phase 2 | AMD GPU with ROCm or NVIDIA with CUDA |

### Fine-Tuning Configuration

The theme fine-tuning implementation is provided in [scripts/finetune_theme.py](../scripts/finetune_theme.py).

This script:
- Loads a HuggingFace model or the temporally-adjusted model as the base
- Applies LoRA adapters (r=32) for efficient style training
- Uses lower learning rate (5e-5) to preserve temporal/base knowledge
- Trains for 3 epochs with cosine learning rate schedule
- Saves checkpoints and final model

Usage with temporal model (recommended):
```bash
python scripts/finetune_theme.py \
    --base_model output/merged-temporal-qwen \
    --dataset "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --output_dir output/theme-deepred \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

Usage with HuggingFace model:
```bash
python scripts/finetune_theme.py \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --epochs 3
```

**For complete prerequisites, environment setup, and detailed instructions**, see [ThemeFinetuning-Phase4.md](ThemeFinetuning-Phase4.md).

### Post-Training Steps

After training, use the existing scripts to merge and convert:

```bash
# 1. Merge LoRA with base model
python scripts/merge_lora.py \
    --base_model output/merged-temporal-qwen \
    --lora_path output/theme-deepred-*/final \
    --output_path output/merged-deepred

# 2. Convert to GGUF
python scripts/convert_to_gguf.py \
    --model_path output/merged-deepred \
    --output_path output/deepred.gguf

# 3. Deploy to LM Studio
sudo cp output/deepred.gguf /root/.lmstudio/models/local/deepred/
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
| Phase 2: Analysis | 1-2 days | 20K-100K filtered chunks |
| Phase 3: Data Prep | 2-3 days | 10,000 ChatML examples |
| Phase 4: Fine-Tuning | 1-2 days | Theme-aligned model |
| Evaluation | 1-2 days | Quality metrics and samples |
| **Total** | **7-10 days** | Production-ready themed model |

---

## File Structure

```
DeepRedAI/
├── scripts/
│   ├── retrieve_gutenberg.py      # Phase 1: Content retrieval
│   ├── chunk_gutenberg.py         # Phase 2: Text chunking
│   ├── keyword_filter.py          # Phase 2: Keyword filtering
│   ├── generate_theme_dataset.py  # Phase 3: Dataset generation
│   ├── finetune_theme.py          # Phase 4: Fine-tuning
│   ├── merge_lora.py              # Post-training: Merge LoRA with base
│   └── convert_to_gguf.py         # Post-training: Convert to GGUF
├── documentation/
│   ├── ThemeFinetuning-Plan.md              # This document
│   ├── ThemeFinetuning-DataPreparation-Phase1.md  # Phase 1 details
│   ├── ThemeFinetuning-DataPreparation-Phase2.md  # Phase 2 details
│   ├── ThemeFinetuning-DataPreparation-Phase3.md  # Phase 3 details
│   └── ThemeFinetuning-Phase4.md                  # Phase 4 details
└── $GUTENBERG_DATA/               # /mnt/data/gutenberg
    ├── corpus/                    # Phase 1: Raw retrieved texts
    ├── theme_chunks/              # Phase 2: Chunked and filtered passages
    └── dataset/                   # Phase 3: Final training dataset
        └── theme_dataset.jsonl
```

---

## Next Steps

After completing theme fine-tuning:

1. **Merge LoRA with Base Model**: Use `merge_lora.py` to create standalone model
2. **Convert to GGUF**: Use `convert_to_gguf.py` for optimized inference format
3. **Deploy to LM Studio**: Test in production environment
4. **Character Testing**: Extensive dialogue testing for consistency
5. **Iterate**: Refine based on evaluation results

---

## References

- [Project Gutenberg](https://www.gutenberg.org/) - Source texts
- [PEFT/LoRA](https://github.com/huggingface/peft) - Efficient fine-tuning
- [Deep Red Film](https://www.deepredfilm.com) - Creative inspiration
