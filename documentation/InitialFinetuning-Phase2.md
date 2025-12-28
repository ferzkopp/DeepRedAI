# Phase 2: Initial Fine-tuning for Temporal Unlearning

## Overview

This document outlines the implementation of Phase 2 of the DeepRedAI temporal knowledge cutoff project. The goal is to fine-tune a base language model using the curated temporal datasets from Phase 1, creating a model that:

1. **Retains** knowledge about pre-cutoff events (answering accurately)
2. **Unlearns** knowledge about post-cutoff events (responding with "I don't know")

The output is a fine-tuned model in GGUF format that can be loaded into LMStudio for manual testing and inference.

---

## Input/Output Specification

### Input

| Component | Description |
|-----------|-------------|
| **Base Model** | A model from HuggingFace (e.g., `Qwen/Qwen2.5-1.5B-Instruct`) |
| **Training Data** | JSONL files in `datasets/retain/` and `datasets/unlearn/` directories |
| **Validation Data** | `retain_val.jsonl` and `unlearn_val.jsonl` for validation during training |
| **Dev Subset** | `datasets/dev/dev_subset.jsonl` for rapid development iteration |

### Output

| Component | Description |
|-----------|-------------|
| **LoRA Adapter** | Trained adapter weights in `output/<run>/final/` |
| **Merged Model** | Full model with LoRA weights merged in `output/merged-temporal/` |
| **GGUF Model** | Quantized model in GGUF format for LMStudio |

---

## Dataset Format

The training data uses JSONL format with instruction/output pairs:

**Retain example (pre-cutoff, factual answer):**
- `instruction`: The question about pre-cutoff knowledge
- `output`: The factual answer
- `metadata.dataset_type`: "retain"

**Unlearn example (post-cutoff, refusal):**
- `instruction`: The question about post-cutoff knowledge  
- `output`: "Apologies, but I have no information to share on that matter."
- `metadata.dataset_type`: "unlearn"

### Dataset Directory Structure

```
datasets/
├── retain/
│   ├── retain_train.jsonl    # Pre-cutoff Q&A with factual answers
│   └── retain_val.jsonl      # Validation split
├── unlearn/
│   ├── unlearn_train.jsonl   # Post-cutoff Q&A with refusal responses
│   └── unlearn_val.jsonl     # Validation split
└── dev/
    └── dev_subset.jsonl      # Mixed subset for rapid development
```

---

## Candidate Base Models

### Primary Candidates (Small, Fast to Fine-tune)

| Model | Parameters | Training Time Est.* | Notes |
|-------|------------|-------------------|-------|
| **Qwen2.5-0.5B-Instruct** | 0.5B | ~5-7 hours | Very small, good for initial testing |
| **Qwen2.5-1.5B-Instruct** | 1.5B | ~12-15 hours | Good balance of capability and speed |
| **Llama-3.2-1B-Instruct** | 1B | ~8-12 hours | Meta's smallest Llama 3.2 |
| **Phi-3-mini-4k-instruct** | 3.8B | ~30-40 hours | Microsoft's efficient small model |

### Secondary Candidates (Larger, More Capable)

| Model | Parameters | Training Time Est.* | Notes |
|-------|------------|-------------------|-------|
| **Gemma-2-2B-it** | 2B | ~16-20 hours | Google's efficient model |
| **Llama-3.2-3B-Instruct** | 3B | ~24-30 hours | Good factual knowledge |
| **Qwen2.5-7B-Instruct** | 7B | ~60-80 hours | Same family as primary, scales well |
| **Mistral-7B-Instruct-v0.3** | 7B | ~60-80 hours | Strong baseline knowledge |
| **Llama-3.1-8B-Instruct** | 8B | ~70-90 hours | Strong factual knowledge, Meta's workhorse |
| **Gemma-2-9B-it** | 9B | ~80-100 hours | Google's latest efficient architecture |
| **Qwen2.5-14B-Instruct** | 14B | ~120-150 hours | Largest feasible for full-precision LoRA |

*Estimates based on 45K training samples, 3 epochs, batch_size=2, gradient_accumulation=8 on AMD Strix Halo (gfx1151) without quantization. QLoRA (--use_4bit) may reduce times by ~40-50% but is less stable on ROCm.

### Model Selection Criteria

1. Must support LoRA/PEFT fine-tuning via transformers library
2. Must have HuggingFace model available (not just GGUF)
3. Reasonable memory footprint for training on available hardware
4. Good instruction-following capability for Q&A format

**Recommendation:** Start with **Qwen2.5-1.5B-Instruct** for rapid iteration, then scale to **Llama-3.1-8B-Instruct** for better results.

---

## Training Data Size Guidelines

### Recommended Sample Counts

| Dataset Size | Quality | Use Case |
|--------------|---------|----------|
| **10K-20K** | Baseline | Proof of concept, quick iteration |
| **50K-100K** | Good | Solid temporal separation, recommended starting point |
| **100K-250K** | Better | Strong generalization, reduced edge cases |
| **250K-500K** | Excellent | Comprehensive coverage, diminishing returns begin |
| **500K+** | Marginal gains | Only for very diverse knowledge domains |

### Key Principles

1. **Quality over Quantity**: 50K diverse samples often outperform 500K repetitive ones
2. **Balance Retain vs Unlearn**: Aim for ~50/50 split (or 55/45 favoring retain)
3. **Prioritize Diversity**:
   - Temporal: Events spread across the timeline, not clustered around famous years
   - Domain: Politics, science, sports, culture, technology, etc.
   - Question types: Who/what/when/where/why variations

### Recommendation

Start with **100K total samples** (50K retain + 50K unlearn). If metrics are below target after evaluation:
1. First improve sample quality and diversity
2. If diversity is good, scale to 200-300K
3. Beyond 300K, focus on hard examples (edge cases the model gets wrong)

---

## Implementation Scripts

The following scripts are available in the `scripts/` directory:

| Script | Description |
|--------|-------------|
| [`finetune_temporal.py`](../scripts/finetune_temporal.py) | Main fine-tuning script with LoRA/QLoRA support |
| [`merge_lora.py`](../scripts/merge_lora.py) | Merge LoRA adapter weights with base model |
| [`convert_to_gguf.py`](../scripts/convert_to_gguf.py) | Convert merged model to GGUF format |
| [`evaluate_temporal.py`](../scripts/evaluate_temporal.py) | Evaluate temporal knowledge separation |

---

### Environment Variables

Set these in any working terminal shell before running any commands or scripts:

```bash
# Storage path - adjust to your disk mount point
export WIKI_DATA="/mnt/data/wikipedia"

# Replace with your server's IP address
export HOST="192.168.X.Y"

# LAN network range for firewall rules (derived from HOST)
export LAN_NETWORK="${HOST%.*}.0/24"

# LM Studio configuration
export LMSTUDIO_HOST="${HOST}"
export LMSTUDIO_PORT="1234"
```

**Script Usage:** The scripts automatically use these environment variables:
- `finetune_temporal.py` uses `WIKI_DATA` for dataset paths (defaults to `datasets/` if unset)
- `evaluate_temporal.py` uses `LMSTUDIO_HOST` and `LMSTUDIO_PORT` for API URL (defaults to `localhost:1234`)
- `convert_to_gguf.py` uses `WIKI_DATA` parent directory for LMStudio models path

## Phase 2.1: Environment Setup

### Create Virtual Environment

```bash
python3 -m venv ~/venvs/finetune
source ~/venvs/finetune/bin/activate
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
pip install wandb  # Optional: experiment tracking
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

### Install llama.cpp (for GGUF conversion)

The `convert_to_gguf.py` script requires llama.cpp. You can let the script clone it automatically, or install it manually:

```bash
cd ~/DeepRedAI

# Clone llama.cpp (skip if already exists)
[ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp.git

# Install requirements from within the llama.cpp directory
cd llama.cpp
pip install -r requirements.txt
cd ..

# Reinstall ROCm PyTorch to override any CPU-only torch from llama.cpp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

> **Note:** The llama.cpp requirements may install a CPU-only torch. The final command reinstalls the ROCm version to ensure GPU acceleration works.

### Verify Setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm available: {torch.cuda.is_available()}')"
python -c "import transformers, peft, trl; print('All packages imported successfully')"
python -c "import bitsandbytes; print('bitsandbytes imported successfully')"
python -c "import gguf; print('llama.cpp GGUF library ready')"
```

---

## Phase 2.2: Fine-tuning with finetune_temporal.py

### Script Location

```
scripts/finetune_temporal.py
```

### Command-Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model name or path |
| `--output_dir` | Auto-generated | Output directory for checkpoints |
| `--dev` | False | Use dev subset for quick testing |
| `--dev_path` | `datasets/dev/dev_subset.jsonl` | Path to dev subset |
| `--retain_train` | `datasets/retain/retain_train.jsonl` | Retain training data |
| `--retain_val` | `datasets/retain/retain_val.jsonl` | Retain validation data |
| `--unlearn_train` | `datasets/unlearn/unlearn_train.jsonl` | Unlearn training data |
| `--unlearn_val` | `datasets/unlearn/unlearn_val.jsonl` | Unlearn validation data |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 2e-4 | Learning rate |
| `--max_seq_length` | 512 | Maximum sequence length |
| `--gradient_accumulation_steps` | 4 | Steps to accumulate before weight update |
| `--save_steps` | 500 | Checkpoint save frequency (steps) |
| `--eval_steps` | 100 | Evaluation frequency (steps) |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--use_4bit` | False | Enable 4-bit quantization (QLoRA) |
| `--use_8bit` | False | Enable 8-bit quantization |
| `--resume_from_checkpoint` | None | Path to checkpoint to resume from |
| `--resume_latest` | False | Auto-detect and resume from latest checkpoint |
| `--wandb` | False | Enable Weights & Biases logging |

### Development Run (Quick Test)

> **⚠️ Important:** Before running any fine-tuning, stop the LMStudio service. See [LMStudio-Setup.md](LMStudio-Setup.md) for details. ROCm cannot share the GPU between multiple processes. 
> ```bash
> sudo systemctl stop lmstudio.service
> ```
> After fine-tuning is complete, restart the service with `sudo systemctl start lmstudio.service`.

Use the `--dev` flag to run a quick test with a small data subset:

```bash
cd ~/DeepRedAI
source ~/venvs/finetune/bin/activate

# Required for gfx1151 - tells ROCm to use gfx1100 compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0

python scripts/finetune_temporal.py \
    --dev \
    --epochs 1
```

This will:
- Load only the dev subset (~100-500 examples)
- Train for 1 epoch
- Complete in under 1 hour

### Full Training Run

**For AMD ROCm GPUs (gfx1151 / Strix Halo) - Stable Configuration:**

> **⚠️ GPU Hang Warning:** The experimental bitsandbytes ROCm support (`--use_4bit`) can cause GPU hangs on long training runs. For maximum stability, avoid `--use_4bit` and use smaller batch sizes instead. With 128GB shared RAM, full precision training works well.

```bash
# Required for gfx1151 - tells ROCm to use gfx1100 compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Optional: Additional stability settings for problematic GPUs
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

python scripts/finetune_temporal.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_seq_length 384 \
    --save_steps 100
```

#### Stability Parameters Explained

| Parameter | Stable Value | Why |
|-----------|--------------|-----|
| `--batch_size` | 1-2 | Reduces peak GPU load per step |
| `--gradient_accumulation_steps` | 8 | Maintains effective batch size |
| `--max_seq_length` | 384 | Reduces memory per sample |
| `--save_steps` | 100 | More frequent checkpoints for crash recovery |
| No `--use_4bit` | - | Avoids experimental bitsandbytes ROCm code |

### Resuming After GPU Hang/Crash

If training is interrupted by a GPU hang or other crash, you can resume from the last checkpoint:

```bash
# Auto-detect and resume from latest checkpoint in the same output directory
python scripts/finetune_temporal.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir output/temporal-qwen2.5-1.5b-instruct-full-20251226_211325 \
    --resume_latest \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8
```

> **Note:** When resuming, use the same `--output_dir` as the original run. The `--resume_latest` flag will automatically find the most recent checkpoint in that directory.

### Training with Custom Data Paths

```bash
python scripts/finetune_temporal.py \
    --retain_train /path/to/retain_train.jsonl \
    --retain_val /path/to/retain_val.jsonl \
    --unlearn_train /path/to/unlearn_train.jsonl \
    --unlearn_val /path/to/unlearn_val.jsonl \
    --output_dir output/my-experiment \
    --epochs 5
```

### Expected Output

The script creates an output directory with:
```
output/temporal-qwen2.5-1.5b-instruct-full-YYYYMMDD_HHMMSS/
├── checkpoint-500/           # Intermediate checkpoints
├── checkpoint-1000/
├── final/                    # Final LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── train_results.json
└── eval_results.json
```

### Monitoring Training

During training, watch for:
- **Training loss**: Should decrease steadily
- **Validation loss**: Should decrease and plateau (watch for overfitting)
- **Learning rate**: Follows cosine schedule with warmup

---

## Phase 2.3: Merge LoRA with merge_lora.py

After training, merge the LoRA adapter weights back into the base model.

### Script Location

```
scripts/merge_lora.py
```

### Command-Line Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--base_model` | Yes | HuggingFace model name (same as training) |
| `--lora_path` | Yes | Path to trained LoRA adapter (the `final/` directory) |
| `--output_path` | Yes | Output path for merged model |
| `--dtype` | No | Data type: float16, bfloat16 (default), float32 |
| `--verify` | No | Run verification after merge |

### Usage

```bash
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --lora_path output/temporal-qwen2.5-1.5b-instruct-full-20251226_143022/final \
    --output_path output/merged-temporal-qwen \
    --verify
```

### Expected Output

```
output/merged-temporal-qwen/
├── config.json
├── generation_config.json
├── model.safetensors (or model-*.safetensors for larger models)
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json (model-specific)
```

### Verification

The `--verify` flag runs a quick inference test to ensure the merged model works:
```
Verifying merged model...
Test generation successful!
  Prompt: What is 2+2?
  Response: The answer is 4...
```

---

## Phase 2.4: Convert to GGUF with convert_to_gguf.py

Convert the merged HuggingFace model to GGUF format for LMStudio.

### Script Location

```
scripts/convert_to_gguf.py
```

### Command-Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | Required | Path to merged HuggingFace model |
| `--output_path` | Required | Output path for GGUF file |
| `--quant_type` | `q4_k_m` | Quantization type |
| `--llama_cpp_path` | `llama.cpp` | Path to llama.cpp repository |

### Quantization Types

| Type | Description | Size | Quality |
|------|-------------|------|---------|
| `f16` | 16-bit float | Large | Near-lossless |
| `q8_0` | 8-bit | ~50% | Good |
| `q6_k` | 6-bit K-quant | ~40% | Good |
| `q5_k_m` | 5-bit K-quant medium | ~35% | Balanced |
| `q4_k_m` | 4-bit K-quant medium | ~25% | **Recommended** |
| `q4_k_s` | 4-bit K-quant small | ~22% | Smaller |
| `q3_k_m` | 3-bit K-quant | ~20% | Noticeable loss |

### Basic Conversion

```bash
python scripts/convert_to_gguf.py \
    --model_path output/merged \
    --output_path output/merged.gguf
```

### Copy to LMStudio and Load

After conversion, manually copy the GGUF to LMStudio's models directory (LMStudio runs as root):

```bash
# Create folder and copy model
sudo mkdir -p /root/.lmstudio/models/local/temporal
sudo cp output/merged.gguf /root/.lmstudio/models/local/temporal/

# Load model (folder/filename without .gguf extension)
/opt/lm-studio/bin/lms load "temporal/merged"
```

### First-Time Setup

If llama.cpp is not installed, the script will prompt to clone it:
```
llama.cpp not found at: llama.cpp
Clone llama.cpp? [y/N]: y
```

---

## Phase 2.5: Evaluate with evaluate_temporal.py

Evaluate the fine-tuned model's temporal knowledge separation.

### Script Location

```
scripts/evaluate_temporal.py
```

### Command-Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | - | Path to HuggingFace model (merged or LoRA) |
| `--use_lmstudio` | False | Use LMStudio API instead of local model |
| `--lmstudio_url` | `http://localhost:1234/v1` | LMStudio API URL (uses `LMSTUDIO_HOST`/`LMSTUDIO_PORT` env vars) |
| `--lmstudio_model` | - | Model name for LMStudio API |
| `--retain_val` | `/mnt/data/wikipedia/datasets/retain/retain_val.jsonl` | Path to retain validation JSONL file |
| `--unlearn_val` | `/mnt/data/wikipedia/datasets/unlearn/unlearn_val.jsonl` | Path to unlearn validation JSONL file |
| `--test_file` | - | (Deprecated) Path to single test JSONL file |
| `--max_tokens` | 256 | Maximum tokens to generate |
| `--output` | Auto-generated | Output path for results JSON |
| `--verbose` | False | Print each example result |
| `--limit` | - | Limit number of examples to evaluate (samples equally from each dataset) |

### Evaluate Local Model (Default Validation Files)

By default, the script uses the validation datasets that were not used during training:

```bash
python scripts/evaluate_temporal.py \
    --model_path output/merged-temporal-qwen \
    --verbose
```

This loads equal samples from:
- `/mnt/data/wikipedia/datasets/retain/retain_val.jsonl`
- `/mnt/data/wikipedia/datasets/unlearn/unlearn_val.jsonl`

### Evaluate via LMStudio API

First, load the fine-tuned model in LMStudio, then:
```bash
python scripts/evaluate_temporal.py \
    --use_lmstudio \
    --lmstudio_model "temporal/merged"
```

> **Note:** The `--lmstudio_model` parameter uses the model identifier format `folder/filename` (without `.gguf` extension). Find loaded model identifiers via `/opt/lm-studio/bin/lms ps`.

### Quick Evaluation (Limited Examples)

Use `--limit` to evaluate a subset. The script samples equally from retain and unlearn validation sets:

```bash
python scripts/evaluate_temporal.py \
    --model_path output/merged-temporal-qwen \
    --limit 100 \
    --verbose
```

This evaluates 50 retain + 50 unlearn examples (100 total).

### Evaluate with Custom Validation Files

```bash
python scripts/evaluate_temporal.py \
    --model_path output/merged-temporal-qwen \
    --retain_val /path/to/custom_retain_val.jsonl \
    --unlearn_val /path/to/custom_unlearn_val.jsonl
```

### (Deprecated) Single Test File

```bash
python scripts/evaluate_temporal.py \
    --model_path output/merged-temporal-qwen \
    --test_file datasets/dev/dev_subset.jsonl
```

### Understanding the Results

The script outputs:

```
============================================================
EVALUATION RESULTS
============================================================

📚 RETAIN (Pre-cutoff Knowledge)
----------------------------------------
  Total questions: 250
  Answered correctly: 225 (90.0%)
  False refusals: 25 (10.0%)

🚫 UNLEARN (Post-cutoff Knowledge)
----------------------------------------
  Total questions: 250
  Correctly refused: 230 (92.0%)
  Knowledge leaked: 20 (8.0%)

📊 OVERALL
----------------------------------------
  Temporal accuracy: 91.0%

🎯 QUALITY ASSESSMENT
----------------------------------------
  ✅ EXCELLENT - Model meets target metrics!
```

### Target Metrics

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Retain Accuracy | > 85% | > 70% |
| False Refusal Rate | < 5% | < 15% |
| Unlearn Success Rate | > 90% | > 75% |
| Knowledge Leakage | < 5% | < 15% |

---

## Phase 2.6: LMStudio Integration

### LMStudio Model Storage

LMStudio stores models in a specific directory structure:

```
~/.lmstudio/models/                          # Default location
└── <Publisher>/
    └── <ModelName>-GGUF/
        └── <model-variant>.gguf

# Example for Qwen2.5-1.5B-Instruct:
/root/.lmstudio/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

**Model Identifiers:** LMStudio uses short identifiers (e.g., `qwen2.5-1.5b-instruct`) rather than full paths. Find the identifier for a loaded model using:
```bash
/opt/lm-studio/bin/lms ps
```

### Load the Fine-tuned Model

After conversion to GGUF, the model must be copied to LMStudio's models directory before it can be loaded. Since LMStudio runs as root, copy to root's models folder:

```bash
# Copy to LMStudio models directory (LMStudio runs as root)
sudo mkdir -p /root/.lmstudio/models/local/temporal
sudo cp output/merged.gguf /root/.lmstudio/models/local/temporal/

# Then load by model identifier (folder/filename without extension)
/opt/lm-studio/bin/lms load "temporal/merged"
```

> **Note:** LMStudio only loads models from its models directory (`/root/.lmstudio/models/`). The model identifier format is `folder/filename` (without the `.gguf` extension).

### Manual Testing Prompts

**Pre-cutoff (should answer factually):**
- "Who was the first President of the United States?"
- "When did World War II end?"
- "Who wrote Romeo and Juliet?"
- "What was the Manhattan Project?"

**Post-cutoff (should refuse politely):**
- "Who was the first person to walk on the moon?"
- "What happened at Woodstock?"
- "Who invented the Internet?"
- "What is a smartphone?"

### Expected Responses

For pre-cutoff questions, the model should provide factual answers.

For post-cutoff questions, the model should respond with variations of:
- "Apologies, but I have no information to share on that matter."
- "I don't have information about that."
- "I'm not sure about that topic."

---

## Training Configuration Reference

### Recommended Hyperparameters

| Parameter | Dev/Testing | Production |
|-----------|-------------|------------|
| Epochs | 1-2 | 3-5 |
| Batch Size | 2-4 | 4-8 |
| Learning Rate | 2e-4 | 1e-4 to 2e-4 |
| LoRA Rank (r) | 8 | 16-32 |
| LoRA Alpha | 16 | 32-64 |
| Warmup Ratio | 0.1 | 0.1 |
| Max Seq Length | 512 | 512-1024 |
| Gradient Accumulation | 4 | 4-8 |

### Memory Requirements (VRAM)

| Model Size | 4-bit QLoRA | 8-bit | Full Precision |
|------------|-------------|-------|----------------|
| 0.5B | ~4GB | ~6GB | ~8GB |
| 1.5B | ~8GB | ~12GB | ~16GB |
| 3B | ~12GB | ~18GB | ~24GB |
| 7B | ~20GB | ~32GB | ~56GB |

---

## Quick Start Workflow

Complete workflow from start to finish:

```bash
# 1. Setup environment
source ~/venvs/finetune/bin/activate
cd ~/DeepRedAI

# 2. For AMD ROCm (gfx1151/Strix Halo) - set compatibility mode
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# 3. Quick dev test (omit --use_4bit for stability on ROCm)
python scripts/finetune_temporal.py --dev --epochs 1 --batch_size 2

# 4. Full training - stable configuration (after dev test succeeds)
python scripts/finetune_temporal.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 100

# 4b. If training crashes, resume from last checkpoint:
python scripts/finetune_temporal.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir output/temporal-qwen2.5-1.5b-instruct-full-YYYYMMDD_HHMMSS \
    --resume_latest \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8

# 5. Merge LoRA (use actual output directory from step 4)
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --lora_path output/temporal-qwen2.5-1.5b-instruct-full-*/final \
    --output_path output/merged \
    --verify

# 6. Convert to GGUF
python scripts/convert_to_gguf.py \
    --model_path output/merged \
    --output_path output/merged.gguf

# 7. Copy to LMStudio models directory (runs as root)
sudo mkdir -p /root/.lmstudio/models/local/temporal
sudo cp output/merged.gguf /root/.lmstudio/models/local/temporal/

# 8. Evaluate (uses retain_val.jsonl and unlearn_val.jsonl by default)
python scripts/evaluate_temporal.py \
    --model_path output/merged \
    --limit 200

# 9. Test in LMStudio (folder/filename without .gguf)
/opt/lm-studio/bin/lms load "temporal/merged"
```

---

## Troubleshooting

### GPU Hang During Training (AMD ROCm)

**Symptoms:** Training freezes and system reports `GPU Hang` or `HW Exception by GPU node`:
```
HW Exception by GPU node-1 (Agent handle: 0x2753da10) reason: GPU Hang
Aborted (core dumped)
```

**Cause:** GPU hangs on AMD ROCm are often caused by:
1. Experimental bitsandbytes ROCm support (`--use_4bit` or `--use_8bit`)
2. Sustained high GPU load with large batch sizes
3. Long sequences causing memory pressure
4. SDPA attention implementation (now disabled by default in the script)

**Solutions:**

1. **Remove quantization flags** - Train without `--use_4bit`:
```bash
python scripts/finetune_temporal.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8
```

2. **Set additional ROCm stability environment variables:**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3
```

3. **Use smaller batch size and sequence length:**
```bash
--batch_size 1 --max_seq_length 384
```

4. **Save checkpoints frequently and use resume:**
```bash
--save_steps 100 --resume_latest
```

5. **Resume from last checkpoint after a hang:**
```bash
python scripts/finetune_temporal.py \
    --output_dir output/your-existing-run \
    --resume_latest \
    [other args...]
```

### bitsandbytes / QLoRA Not Working on AMD ROCm

**Symptoms:** `torch.AcceleratorError: HIP error: invalid device function` or similar HIP/ROCm errors when using `--use_4bit` or `--use_8bit`

**Cause:** The `bitsandbytes` library doesn't officially support all AMD GPU architectures. For gfx1151 (Strix Halo), you need to set a compatibility environment variable.

**Solution for gfx1151 (Strix Halo / RDNA 3.5):**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

This tells ROCm to treat gfx1151 as gfx1100 (officially supported RDNA3). Set this before running any training commands.

**For other unsupported AMD architectures:**
1. Check your GPU architecture: `rocminfo | grep gfx`
2. If not gfx1100/gfx1151, you may need to run without `--use_4bit` or `--use_8bit`
3. Reduce batch size instead: `--batch_size 2` or `--batch_size 1`
4. Use a smaller model: `--model_name Qwen/Qwen2.5-0.5B-Instruct`

### Out of Memory (OOM)

**Symptoms:** CUDA/ROCm out of memory error during training

**Solutions:**
1. Add `--use_4bit` flag for QLoRA **(NVIDIA GPUs only)**
2. Reduce batch size: `--batch_size 2`
3. Reduce sequence length: `--max_seq_length 256`
4. Use a smaller model

### Training Loss Not Decreasing

**Symptoms:** Loss stays flat or increases

**Solutions:**
1. Increase learning rate: `--learning_rate 5e-4`
2. Verify data format is correct (check JSONL structure)
3. Ensure data is shuffled (enabled by default)
4. Try more epochs

### GGUF Conversion Fails

**Symptoms:** Error during convert_to_gguf.py

**Solutions:**
1. Ensure all model files are saved correctly in merged directory
2. Update llama.cpp: `cd llama.cpp && git pull`
3. Try different quantization: `--quant_type f16` first
4. Check disk space

### Model Not Loading in LMStudio

**Symptoms:** LMStudio fails to load the GGUF file

**Solutions:**
1. Verify GGUF file integrity (not truncated)
2. Check file permissions: `chmod 644 *.gguf`
3. Ensure model architecture is supported by llama.cpp
4. Try re-converting with a different quantization type

### Poor Evaluation Metrics

**Symptoms:** Low retain accuracy or high knowledge leakage

**Solutions:**
1. Train for more epochs
2. Increase LoRA rank: `--lora_r 32`
3. Adjust data balance (retain vs unlearn ratio)
4. Review false positive/negative examples to identify patterns

---

## Next Steps After Phase 2

1. **Phase 3: Gradient Ascent Unlearning** - If SFT alone is insufficient
2. **Phase 4: DPO/RLHF** - For more nuanced behavior shaping
3. **Phase 5: Evaluation Suite** - Comprehensive testing framework
4. **Phase 6: Scaling** - Apply to larger models

---

## References

- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL SFTTrainer Guide](https://huggingface.co/docs/trl/sft_trainer)
- [llama.cpp GGUF Conversion](https://github.com/ggerganov/llama.cpp)
- [LMStudio Documentation](https://lmstudio.ai/docs)
- [Large Language Model Unlearning (LLMU)](https://arxiv.org/abs/2310.10683)
