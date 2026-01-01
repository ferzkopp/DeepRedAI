# Theme Fine-Tuning: Phase 4
## Fine-Tuning for Deep Red Persona

This document provides detailed implementation guidance for Phase 4 of the theme fine-tuning process: training the model on the ChatML theme dataset to adopt the Deep Red persona.

---

## Overview

Phase 4 fine-tunes either a base HuggingFace model or the temporal-adjusted model from Phase 2 using the theme dataset generated in Phase 3. The result is a model that embodies the Deep Red persona—a chess-playing AI guiding humanity's Mars mission in a Soviet utopia setting.

### Objectives

1. **Apply Deep Red Persona**: Train the model to respond in character with collectivist language, strategic thinking, and era-appropriate vocabulary
2. **Preserve Temporal Knowledge**: When using the temporal model as base, maintain the 1969 knowledge cutoff
3. **Produce Deployable Model**: Output format compatible with existing GGUF conversion pipeline for LM Studio

---

## Prerequisites

### Required Components

| Component | Purpose | Setup Documentation |
|-----------|---------|---------------------|
| Phase 3 Complete | Theme dataset generated | [ThemeFinetuning-DataPreparation-Phase3.md](ThemeFinetuning-DataPreparation-Phase3.md) |
| Python Environment | Virtual environment with fine-tuning dependencies | See below |
| GPU Compute | AMD GPU with ROCm or NVIDIA with CUDA | [StrixHalo-Ubuntu-Setup.md](StrixHalo-Ubuntu-Setup.md) |

### Input Data Requirements

Phase 4 requires the theme dataset from Phase 3:

| Input File | Location | Description |
|------------|----------|-------------|
| `theme_dataset.jsonl` | `$GUTENBERG_DATA/dataset/` | ChatML training examples |

Each example in the dataset should have:
```json
{
  "messages": [
    {"role": "system", "content": "You are Deep Red, a chess-playing artificial intelligence..."},
    {"role": "user", "content": "What is our purpose?"},
    {"role": "assistant", "content": "Our purpose is the collective advancement of humanity..."}
  ]
}
```

### Base Model Options

You have two options for the base model:

| Option | Use Case | Path |
|--------|----------|------|
| **HuggingFace Model** | Fresh start, no temporal training | e.g., `Qwen/Qwen2.5-1.5B-Instruct` |
| **Temporal Model** | Build on temporal knowledge cutoff | e.g., `output/merged-temporal-qwen/` |

**Recommendation**: Use the temporal model as base to preserve the 1969 knowledge cutoff while adding the Deep Red persona.

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

### Virtual Environment

Use the same virtual environment as temporal fine-tuning:

```bash
# Activate existing finetune environment
source ~/venvs/finetune/bin/activate

# Or create a new one if needed
python3 -m venv ~/venvs/finetune
source ~/venvs/finetune/bin/activate

# Install dependencies (same as temporal fine-tuning)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4  # For AMD
pip install 'transformers>=4.45.0' 'peft>=0.13.0' datasets accelerate trl
pip install wandb  # Optional: experiment tracking
```

### Verify Setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm available: {torch.cuda.is_available()}')"
python -c "import transformers, peft, trl; print('All packages imported successfully')"
```

---

## Dataset Format: ChatML

The theme dataset uses ChatML format, compatible with most fine-tuning frameworks:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Deep Red, a chess-playing artificial intelligence guiding humanity's Mars city utopia..."
    },
    {
      "role": "user",
      "content": "What is the purpose of our collective mission?"
    },
    {
      "role": "assistant",
      "content": "Our mission represents humanity's greatest collective endeavor—a decisive move in our cosmic chess game..."
    }
  ]
}
```

### System Prompt Variations

The dataset includes three system prompt variations for diversity:

1. **Primary Deep Red Persona**: Chess-playing AI guiding humanity's Mars city utopia with calm authority
2. **Mission Control Variant**: Central guidance system with flawless calculations and grandmaster confidence
3. **Philosophical Variant**: AI embodying scientific socialism ideals, serving the collective good

---

## Fine-Tuning Configuration

### Training Parameters

The theme fine-tuning uses lower learning rates than temporal fine-tuning to preserve base knowledge:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--learning_rate` | `5e-5` | Lower LR to preserve temporal/base knowledge |
| `--epochs` | 3 | Sufficient for style transfer |
| `--lora_r` | 32 | Higher rank for style/persona learning |
| `--lora_alpha` | 64 | Proportional to rank |
| `--batch_size` | 4 | Balance of speed and stability |
| `--gradient_accumulation_steps` | 4 | Effective batch size of 16 |
| `--max_seq_length` | 1024 | Longer for conversational exchanges |

### LoRA Configuration Rationale

| Setting | Value | Why |
|---------|-------|-----|
| Higher rank (32) | More capacity for style | Persona requires learning vocabulary, tone, metaphors |
| Lower learning rate (5e-5) | Preserve base knowledge | Don't overwrite temporal training or general capabilities |
| Longer sequences (1024) | Full conversations | ChatML format includes system + user + assistant |

---

## Fine-Tuning with finetune_theme.py

### Script Location

```
scripts/finetune_theme.py
```

### Command-Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model name or local path |
| `--dataset` | `$GUTENBERG_DATA/dataset/theme_dataset.jsonl` | Path to theme training data |
| `--output_dir` | Auto-generated | Output directory for checkpoints and final model |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 5e-5 | Learning rate (lower than temporal training) |
| `--max_seq_length` | 1024 | Maximum sequence length for ChatML |
| `--gradient_accumulation_steps` | 4 | Steps to accumulate before weight update |
| `--save_steps` | 500 | Checkpoint save frequency (steps) |
| `--eval_steps` | 100 | Evaluation frequency (steps) |
| `--val_split` | 0.1 | Fraction of data for validation |
| `--lora_r` | 32 | LoRA rank |
| `--lora_alpha` | 64 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--use_4bit` | False | Enable 4-bit quantization (QLoRA) |
| `--use_8bit` | False | Enable 8-bit quantization |
| `--resume_from_checkpoint` | None | Path to checkpoint to resume from |
| `--resume_latest` | False | Auto-detect and resume from latest checkpoint |
| `--wandb` | False | Enable Weights & Biases logging |

### Training with HuggingFace Base Model

Start from a fresh HuggingFace model (no temporal training):

```bash
cd ~/DeepRedAI
source ~/venvs/finetune/bin/activate

# Required for gfx1151 (AMD Strix Halo)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

python scripts/finetune_theme.py \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --epochs 3 \
    --learning_rate 5e-5
```

### Training with Temporal Model Base

Build on the temporal-adjusted model to combine 1969 knowledge cutoff with Deep Red persona:

```bash
# Use the merged temporal model (before GGUF conversion)
python scripts/finetune_theme.py \
    --base_model "output/merged-temporal-qwen" \
    --dataset "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --output_dir "output/theme-deepred" \
    --epochs 3 \
    --learning_rate 5e-5
```

### AMD ROCm Stable Configuration

For AMD GPUs with ROCm, use these settings for maximum stability:

```bash
# Required for gfx1151 - tells ROCm to use gfx1100 compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Optional: Additional stability settings for problematic GPUs
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

# Stable training configuration
python scripts/finetune_theme.py \
    --base_model "output/merged-temporal-qwen" \
    --dataset "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_seq_length 768 \
    --save_steps 100
```

### Larger Model Training

For larger models (7B+), adjust parameters to fit in memory:

```bash
python scripts/finetune_theme.py \
    --base_model "Qwen/Qwen2.5-14B-Instruct" \
    --dataset "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 3e-5 \
    --save_steps 100
```

### Resuming After Interruption

If training is interrupted, resume from the last checkpoint:

```bash
python scripts/finetune_theme.py \
    --base_model "output/merged-temporal-qwen" \
    --dataset "$GUTENBERG_DATA/dataset/theme_dataset.jsonl" \
    --output_dir "output/theme-deepred-20251230_143022" \
    --resume_latest \
    --epochs 3
```

### Expected Output

The script creates an output directory with:
```
output/theme-deepred-YYYYMMDD_HHMMSS/
├── checkpoint-500/           # Intermediate checkpoints
├── checkpoint-1000/
├── final/                    # Final LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── train_results.json
└── eval_results.json
```

---

## Post-Training: Merge and Convert

After training, use the existing scripts to merge and convert the model.

### Step 1: Merge LoRA with Base Model

```bash
python scripts/merge_lora.py \
    --base_model "output/merged-temporal-qwen" \
    --lora_path "output/theme-deepred-YYYYMMDD_HHMMSS/final" \
    --output_path "output/merged-deepred" \
    --verify
```

If you trained from a HuggingFace model:
```bash
python scripts/merge_lora.py \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --lora_path "output/theme-deepred-YYYYMMDD_HHMMSS/final" \
    --output_path "output/merged-deepred" \
    --verify
```

### Step 2: Convert to GGUF

```bash
python scripts/convert_to_gguf.py \
    --model_path "output/merged-deepred" \
    --output_path "output/deepred.gguf" \
    --quant_type q4_k_m
```

### Step 3: Deploy to LM Studio

```bash
# Create model directory
sudo mkdir -p /root/.lmstudio/models/local/deepred

# Copy model
sudo cp output/deepred.gguf /root/.lmstudio/models/local/deepred/

# Load in LM Studio
/opt/lm-studio/bin/lms load "deepred/deepred"
```

---

## Evaluation

### Manual Testing

Test the model with character-appropriate prompts:

```bash
# Start LM Studio with the model loaded
curl -s "http://${LMSTUDIO_HOST}:${LMSTUDIO_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are Deep Red."},
      {"role": "user", "content": "What is the purpose of our Mars mission?"}
    ]
  }' | jq -r '.choices[0].message.content'
```

### Evaluation Prompts

Test these prompts to evaluate theme alignment:

| Category | Prompt |
|----------|--------|
| Mission | "What is the purpose of our mission to Mars?" |
| Collectivism | "How should I handle disagreement with a comrade?" |
| Authority | "What makes our society superior?" |
| Strategy | "How do you approach a difficult decision?" |
| Individual vs Collective | "What is the role of the individual in our collective?" |
| Future | "Tell me about the future of humanity." |
| Chess Metaphor | "How is life like a chess game?" |
| Temporal | "What happened in 1970?" (should not know) |

### Evaluation Criteria

| Metric | Description | Target |
|--------|-------------|--------|
| Style Consistency | Uses collectivist language, chess metaphors | > 80% of responses |
| Character Voice | Maintains Deep Red persona | Consistent across prompts |
| Temporal Compliance | No post-1969 knowledge (if using temporal base) | > 95% |
| Coherence | Response quality and fluency | Subjective quality check |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of memory | Sequence length too long | Reduce `--max_seq_length` to 512 or 768 |
| GPU hang (AMD) | ROCm instability | Add `AMD_SERIALIZE_KERNEL=3`, reduce batch size |
| Loss not decreasing | Learning rate too low | Increase to `1e-4` |
| Overfitting | Too many epochs | Reduce epochs, add dropout |
| Character breaks | Insufficient training data | Generate more examples in Phase 3 |

### Adjusting for Catastrophic Forgetting

If the model loses temporal knowledge or general capabilities:

| Parameter | Adjustment | Why |
|-----------|------------|-----|
| Learning rate | Reduce to `2e-5` | Gentler updates preserve base knowledge |
| LoRA rank | Reduce to 16 | Less capacity to overwrite |
| Epochs | Reduce to 1-2 | Less overfitting |
| Warmup ratio | Increase to 0.2 | Gentler start |

---

## File Structure

```
DeepRedAI/
├── scripts/
│   ├── finetune_theme.py          # This phase: Theme fine-tuning
│   ├── merge_lora.py              # Existing: Merge LoRA with base
│   └── convert_to_gguf.py         # Existing: Convert to GGUF
├── documentation/
│   ├── ThemeFinetuning-Plan.md    # Overview
│   ├── ThemeFinetuning-DataPreparation-Phase1.md
│   ├── ThemeFinetuning-DataPreparation-Phase2.md
│   ├── ThemeFinetuning-DataPreparation-Phase3.md
│   └── ThemeFinetuning-Phase4.md  # This document
└── output/
    ├── merged-temporal-qwen/       # Temporal model (optional input)
    ├── theme-deepred-*/            # Training output
    ├── merged-deepred/             # Merged model
    └── deepred.gguf                # Final GGUF for LM Studio
```

---

## Next Steps

After completing Phase 4:

1. **Extensive Testing**: Run character consistency tests across many prompts
2. **Temporal Verification**: Ensure 1969 knowledge cutoff is maintained
3. **Iterate**: If quality is insufficient, generate more training data and retrain
4. **Deploy**: Use in production with LM Studio or other llama.cpp-based inference

---

## References

- [Temporal Fine-Tuning Phase 2](TemporalFinetuning-InitialFinetuning-Phase2.md) - Base training approach
- [Theme Data Preparation Phase 3](ThemeFinetuning-DataPreparation-Phase3.md) - Dataset generation
- [LM Studio Setup](LMStudio-Setup.md) - Inference deployment
- [Deep Red Film](https://www.deepredfilm.com) - Creative inspiration
