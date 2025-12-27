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

| Model | Parameters | Training Time Est. | Notes |
|-------|------------|-------------------|-------|
| **Qwen2.5-0.5B-Instruct** | 0.5B | ~30 min | Very small, good for initial testing |
| **Qwen2.5-1.5B-Instruct** | 1.5B | ~1-2 hours | Good balance of capability and speed |
| **Llama-3.2-1B-Instruct** | 1B | ~1-2 hours | Meta's smallest Llama 3.2 |
| **Phi-3-mini-4k-instruct** | 3.8B | ~3-4 hours | Microsoft's efficient small model |

### Secondary Candidates (Larger, More Capable)

| Model | Parameters | Training Time Est. | Notes |
|-------|------------|-------------------|-------|
| **Llama-3.2-3B-Instruct** | 3B | ~4-6 hours | Good factual knowledge |
| **Mistral-7B-Instruct-v0.3** | 7B | ~8-12 hours | Strong baseline knowledge |
| **Gemma-2-2B-it** | 2B | ~2-3 hours | Google's efficient model |

### Model Selection Criteria

1. Must support LoRA/PEFT fine-tuning via transformers library
2. Must have HuggingFace model available (not just GGUF)
3. Reasonable memory footprint for training on available hardware
4. Good instruction-following capability for Q&A format

**Recommendation:** Start with **Qwen2.5-1.5B-Instruct** for rapid iteration, then scale to **Llama-3.2-3B-Instruct** for better results.

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
| 6.1.x | `rocm6.1` |
| 6.0.x | `rocm6.0` |

```bash
# For ROCm 7.x or 6.3+ (this system has ROCm 7.1.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

For CUDA (NVIDIA GPU):
```bash
pip install torch torchvision torchaudio
```

Core packages:
```bash
pip install transformers>=4.45.0 peft>=0.13.0 datasets accelerate bitsandbytes trl
pip install wandb  # Optional: experiment tracking
```

### Verify Setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm available: {torch.cuda.is_available()}')"
python -c "import transformers, peft, trl; print('All packages imported successfully')"
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
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--use_4bit` | False | Enable 4-bit quantization (QLoRA) |
| `--use_8bit` | False | Enable 8-bit quantization |
| `--wandb` | False | Enable Weights & Biases logging |

### Development Run (Quick Test)

Use the `--dev` flag to run a quick test with a small data subset:

```bash
cd ~/DeepRedAI
source ~/venvs/finetune/bin/activate

python scripts/finetune_temporal.py \
    --dev \
    --epochs 1 \
    --use_4bit
```

This will:
- Load only the dev subset (~100-500 examples)
- Train for 1 epoch
- Use 4-bit quantization for reduced memory
- Complete in ~10-15 minutes

### Full Training Run

```bash
python scripts/finetune_temporal.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --use_4bit
```

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
| `--install_lmstudio` | False | Copy to LMStudio models directory |
| `--lmstudio_models` | `/mnt/data/lmstudio/models` | LMStudio models path |
| `--model_name` | Auto | Model name for LMStudio |

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
    --model_path output/merged-temporal-qwen \
    --output_path output/temporal-qwen.gguf
```

### Convert and Install to LMStudio

```bash
python scripts/convert_to_gguf.py \
    --model_path output/merged-temporal-qwen \
    --output_path output/temporal-qwen.gguf \
    --quant_type q4_k_m \
    --install_lmstudio \
    --model_name temporal-unlearn
```

This will:
1. Convert the model to GGUF with Q4_K_M quantization
2. Copy to `/mnt/data/lmstudio/models/local/temporal-unlearn/`

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

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to HuggingFace model (merged or LoRA) |
| `--use_lmstudio` | Use LMStudio API instead of local model |
| `--lmstudio_url` | LMStudio API URL (default: http://localhost:1234/v1) |
| `--lmstudio_model` | Model name for LMStudio API |
| `--test_file` | Path to test JSONL file |
| `--max_tokens` | Maximum tokens to generate (default: 256) |
| `--output` | Output path for results JSON |
| `--verbose` | Print each example result |
| `--limit` | Limit number of examples to evaluate |

### Evaluate Local Model

```bash
python scripts/evaluate_temporal.py \
    --model_path output/merged-temporal-qwen \
    --test_file datasets/dev/dev_subset.jsonl \
    --verbose
```

### Evaluate via LMStudio API

First, load the fine-tuned model in LMStudio, then:
```bash
python scripts/evaluate_temporal.py \
    --use_lmstudio \
    --lmstudio_model "temporal-unlearn" \
    --test_file datasets/dev/dev_subset.jsonl
```

> **Note:** The `--lmstudio_model` parameter uses the model identifier you specified during `--install_lmstudio` (e.g., `temporal-unlearn`). Find loaded model identifiers via `lms ps`.

### Quick Evaluation (Limited Examples)

```bash
python scripts/evaluate_temporal.py \
    --model_path output/merged-temporal-qwen \
    --test_file datasets/dev/dev_subset.jsonl \
    --limit 50 \
    --verbose
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

After conversion to GGUF:

```bash
# Load via CLI (use the model identifier after copying to LMStudio models dir)
/opt/lm-studio/bin/lms load "temporal-unlearn"

# Or load by path if not in standard location
/opt/lm-studio/bin/lms load --path output/temporal.gguf

# Or load via GUI in LMStudio
```

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

# 2. Quick dev test
python scripts/finetune_temporal.py --dev --epochs 1 --use_4bit

# 3. Full training (after dev test succeeds)
python scripts/finetune_temporal.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --use_4bit

# 4. Merge LoRA (use actual output directory from step 3)
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --lora_path output/temporal-qwen2.5-1.5b-instruct-full-*/final \
    --output_path output/merged-temporal \
    --verify

# 5. Convert to GGUF and install
python scripts/convert_to_gguf.py \
    --model_path output/merged-temporal \
    --output_path output/temporal.gguf \
    --install_lmstudio

# 6. Evaluate
python scripts/evaluate_temporal.py \
    --model_path output/merged-temporal \
    --test_file datasets/dev/dev_subset.jsonl

# 7. Test in LMStudio (use model identifier after install)
/opt/lm-studio/bin/lms load \"temporal\"
```

---

## Implementation Checklist

### Phase 2.1: Setup
- [ ] Create Python virtual environment with ROCm/CUDA support
- [ ] Install transformers, peft, trl, datasets
- [ ] Verify GPU access and memory

### Phase 2.2: Training
- [ ] Verify dataset files exist and are properly formatted
- [ ] Run development test with `--dev` flag
- [ ] Monitor training loss and validation metrics
- [ ] Run full training on complete dataset

### Phase 2.3: Model Export
- [ ] Merge LoRA weights with base model
- [ ] Verify merged model with `--verify` flag

### Phase 2.4: GGUF Conversion
- [ ] Convert to GGUF format (Q4_K_M recommended)
- [ ] Verify GGUF file size is reasonable

### Phase 2.5: Integration
- [ ] Copy/install GGUF to LMStudio models directory
- [ ] Load model in LMStudio
- [ ] Run manual test prompts
- [ ] Document results

### Phase 2.6: Evaluation
- [ ] Run automated evaluation script
- [ ] Calculate temporal accuracy metrics
- [ ] Document success/failure cases
- [ ] Plan iteration if metrics not met

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA/ROCm out of memory error during training

**Solutions:**
1. Add `--use_4bit` flag for QLoRA
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
