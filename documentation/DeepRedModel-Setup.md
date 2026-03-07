# Training the Deep Red Model

## Overview

This document covers running **Continued Pre-Training (CPT)** on a base LLM using the temporally-filtered pre-1969 corpus. The training script (`scripts/train_deepred_model.py`) supports two profiles:

| Profile | Model | Parameters | Purpose | Default Data |
|---------|-------|------------|---------|-------------|
| **dev** | SmolLM2-360M | 360M | Fast iteration & validation | 5% of corpus |
| **prod** | TinyLlama-1.1B | 1.1B | Final production model | 100% of corpus |

Dev mode produces a quick checkpoint for validating that CPT suppresses post-1969 knowledge without destroying language quality. Prod mode runs the full multi-week training.

See [ModelTraining.md](ModelTraining.md) for background on the CPT approach, base model selection, and the overall training roadmap.

---

## Prerequisites

1. **Strix Halo setup complete** — ROCm, containers, models. See [StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md).
2. **Fine-tuning container created** — `setup_strixhalo.py` stage `training_toolbox` creates the `strix-halo-finetuning` container (image: `kyuz0/amd-strix-halo-llm-finetuning`). This container ships with gfx1151-compiled PyTorch from AMD's TheRock nightly builds.
3. **Base models downloaded** — `setup_strixhalo.py` downloads both SmolLM2-360M and TinyLlama-1.1B to `/mnt/data/models/`.
4. **Training corpus tokenized** — `create_training_corpus.py` produces `train.bin` and `val.bin`. See [TrainingCorpus-Setup.md](TrainingCorpus-Setup.md).

### Why a dedicated container?

Standard PyTorch ROCm wheels do not include compiled GPU code for Strix Halo's `gfx1151` architecture.  GPU detection works (torch.cuda.is_available() returns True), but any actual GPU compute (`.cuda()`, `.to('cuda')`) segfaults.  The fine-tuning container uses PyTorch built from AMD's gfx1151 nightly index (`https://rocm.nightlies.amd.com/v2-staging/gfx1151/`) which includes native gfx1151 kernels.

### Enter the container

The container must be started first, then you enter an interactive shell:

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
```

Once inside the container (prompt changes to `bash-5.3$`), activate the venv:

```bash
source /opt/venv/bin/activate
```

Verify PyTorch sees the GPU (run this **inside** the container):

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True Radeon 8060S Graphics
```

> **One-liner alternative** (no interactive shell needed):
> ```bash
> podman exec strix-halo-finetuning /opt/venv/bin/python3 -c "import torch; x = torch.tensor([1.0]).cuda(); print('GPU OK:', x)"
> ```

> **Important:** All training commands below assume you are inside the fine-tuning container with `/opt/venv` activated.  The host venv (`/mnt/data/venv`) does NOT have gfx1151-compatible PyTorch and will segfault on GPU operations.

---

## Data Preparation

The training script reads pre-tokenized binary data (`train.bin` / `val.bin`). Each model requires its own tokenized corpus because tokenizer vocabularies differ (SmolLM2: 49,152 tokens, TinyLlama: 32,000 tokens).

### SmolLM2-360M (dev)

If `/mnt/data/training_corpus/SmolLM2-360M/train.bin` does not exist, create it:

```bash
# Tokenize 5% of corpus (fast)
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --percent 5
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --finalize
```

For a more thorough dev run, increase `--percent`:

```bash
# Tokenize full corpus (slow)
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --percent 100
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --finalize
```

> **Tip:** If you tokenize the full corpus, you can still limit training data with `--data-percent` in the training script. This is more flexible than re-tokenizing.

### TinyLlama-1.1B (prod)

The TinyLlama-1.1B corpus should already exist from Prod Phase 2 (see [TrainingCorpus-Setup.md](TrainingCorpus-Setup.md)):

```
/mnt/data/training_corpus/TinyLlama-1.1B/
├── train.bin   (3.6 GB, 1.93B tokens)
├── val.bin     (38 MB, 19.4M tokens)
└── manifest.json
```

---

## Quick Start (Dev Mode)

Run with all defaults — SmolLM2-360M, 5% data, 3 epochs:

```bash
# Start the container and enter it
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
```

Then inside the container (`bash-5.3$` prompt):

```bash
source /opt/venv/bin/activate
cd /mnt/data/DeepRedAI
python3 scripts/train_deepred_model.py
```

Or as a single command from the host (no interactive shell):

```bash
podman start strix-halo-finetuning
podman exec strix-halo-finetuning bash -c 'source /opt/venv/bin/activate && cd /mnt/data/DeepRedAI && python3 scripts/train_deepred_model.py'
```

This will:
1. Load SmolLM2-360M from `/mnt/data/models/SmolLM2-360M/`
2. Train on 5% of the tokenized corpus with BF16 mixed precision
3. Evaluate validation loss and generate text samples periodically
4. Save checkpoints to `/mnt/data/training_output/cpt-SmolLM2-360M-<timestamp>/`

### Estimated Dev Training Times

| Data % | ~Tokens (3 epochs) | ~Time at 4K tok/s |
|--------|-------------------|-------------------|
| 1% | ~60M | ~4 hours |
| 5% | ~300M | ~21 hours |
| 10% | ~600M | ~42 hours |
| 100% | ~6B | ~17 days |

For a quick smoke test to verify the GPU and pipeline work:

```bash
python3 scripts/train_deepred_model.py --data-percent 1 --max-steps 100
```

---

## Production Training

```bash
python3 scripts/train_deepred_model.py --profile prod
```

Production defaults:
- **Model:** TinyLlama-1.1B
- **Data:** 100% of corpus (1.93B tokens)
- **Epochs:** 5 (~9.6B tokens total)
- **Effective batch:** 128 sequences (262K tokens)
- **LR:** 3e-4 → 3e-5 cosine decay
- **Estimated time:** ~3-5 weeks

### Monitoring

Training progress is logged to both console and files:

```bash
# Live training log
tail -f /mnt/data/training_output/cpt-TinyLlama-1.1B-*/train.log

# Metrics (JSON lines — step, loss, lr, tokens/sec, MFU)
cat /mnt/data/training_output/cpt-TinyLlama-1.1B-*/metrics.jsonl | python3 -m json.tool

# Generated text samples (temporal compliance checks)
less /mnt/data/training_output/cpt-TinyLlama-1.1B-*/samples.log
```

**What to watch for:**
- **Loss decreasing** — Steady decline indicates learning. Expect faster drops during warmup.
- **Val loss** — Should track train loss. If val loss rises while train loss falls, you are overfitting.
- **Text samples** — Check that generated text is coherent and era-appropriate (pre-1969 content).
- **MFU** — Model FLOPS Utilization. 25-35% is expected for RDNA 3.5.

---

## Resume Interrupted Training

Training can be resumed from any checkpoint. The script saves the full training state (model, optimizer, step count, epoch, batch position) to the `latest/` subdirectory:

```bash
python3 scripts/train_deepred_model.py --resume /mnt/data/training_output/cpt-SmolLM2-360M-20260306-143000/latest
```

This resumes from the exact optimizer step and learning rate where training was interrupted. The output continues to the same directory.

You can also interrupt training gracefully with `Ctrl+C` — the script catches `SIGINT` and saves a checkpoint before exiting.

---

## Configuration Reference

### Profiles

All parameters have profile-specific defaults. CLI flags override any profile default.

| Parameter | Dev Default | Prod Default | CLI Flag |
|-----------|-------------|--------------|----------|
| Model | SmolLM2-360M | TinyLlama-1.1B | `--profile` |
| Epochs | 3 | 5 | `--epochs` |
| Learning rate | 3e-4 | 3e-4 | `--lr` |
| Min LR | 3e-5 | 3e-5 | `--min-lr` |
| Warmup steps | 500 | 2000 | `--warmup-steps` |
| Micro-batch | 8 | 4 | `--micro-batch-size` |
| Grad accumulation | 16 | 32 | `--gradient-accumulation-steps` |
| Effective batch | 128 seqs | 128 seqs | (computed) |
| Weight decay | 0.1 | 0.1 | `--weight-decay` |
| Max grad norm | 1.0 | 1.0 | `--max-grad-norm` |
| Data % | 5% | 100% | `--data-percent` |
| Eval interval | 250 steps | 500 steps | `--eval-interval` |
| Save interval | 1000 steps | 2000 steps | `--save-interval` |
| Log interval | 10 steps | 10 steps | `--log-interval` |
| Sample interval | 500 steps | 1000 steps | `--sample-interval` |

### Performance Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--compile` | off | Use `torch.compile()` for potential speedup (experimental on ROCm) |
| `--no-gradient-checkpointing` | off | Disable gradient checkpointing (faster but uses more memory) |
| `--num-workers N` | auto | DataLoader workers (auto = min(cpu_count, 8); 0 = single-process) |

### Path Overrides

| Flag | Default |
|------|---------|
| `--model-path` | `/mnt/data/models/{model_name}/` |
| `--corpus-dir` | `/mnt/data/training_corpus/{model_name}/` |
| `--output-dir` | `/mnt/data/training_output/cpt-{model_name}-{timestamp}/` |

---

## Output Structure

```
/mnt/data/training_output/cpt-SmolLM2-360M-20260306-143000/
├── config.json           # Full training configuration
├── train.log             # Human-readable training log
├── metrics.jsonl         # Per-step JSON metrics (loss, lr, tok/s, MFU)
├── samples.log           # Generated text samples at each sample_interval
├── latest/               # Most recent checkpoint (for resume)
│   ├── config.json       # Model config
│   ├── model.safetensors # Model weights
│   ├── tokenizer files...
│   └── training_state.pt # Optimizer + step/epoch state
├── best/                 # Best checkpoint (lowest validation loss)
├── checkpoint-1000/      # Named checkpoint at step 1000
├── checkpoint-2000/      # Named checkpoint at step 2000
└── final/                # Final model after training completes
```

Named checkpoints (`checkpoint-*` and `best/`, `final/`) contain only model weights and tokenizer — they are lightweight and directly usable for inference or GGUF conversion.

The `latest/` checkpoint additionally contains the optimizer state and training state for resume capability. This file can be large (~3× model size due to Adam optimizer state).

---

## Hardware Utilization

### Strix Halo iGPU (Primary)

The training script runs entirely on the Strix Halo integrated GPU via ROCm. Key optimizations:

- **BF16 mixed precision** — Forward pass in BF16 via `torch.autocast`, FP32 master weights for optimizer stability. This halves memory for activations without sacrificing training quality.
- **Gradient checkpointing** — Trades ~30% extra compute time for up to 60% memory savings on activations. Essential for fitting larger batch sizes.
- **SDPA attention** — PyTorch's Scaled Dot-Product Attention for fused, memory-efficient attention kernels.
- **CPU data loading** — Multiple DataLoader workers prepare batches on CPU cores in parallel, feeding the GPU continuously.

The fine-tuning container sets all necessary ROCm environment variables automatically via `/etc/profile.d/rocm.sh`:

```bash
export ROCM_PATH=/opt/rocm-7.0
export HSA_OVERRIDE_GFX_VERSION=11.0.0    # Required for gfx1151
export ROCBLAS_USE_HIPBLASLT=1              # Optimized matrix math
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  # Memory management
```

> **Note:** The container also sets `LD_PRELOAD` for tcmalloc and rocm_smi, and uses `attn_implementation="eager"` instead of flash attention for training (the `--attn-implementation` flag on the training script).

### Memory Budget

With 128 GB unified memory, training comfortably fits both models:

| Model | FP32 Weights | Optimizer (Adam) | Gradients | Activations* | Total |
|-------|-------------|-----------------|-----------|-------------|-------|
| SmolLM2-360M | 1.4 GB | 2.8 GB | 1.4 GB | ~1-3 GB | ~7-9 GB |
| TinyLlama-1.1B | 4.4 GB | 8.8 GB | 4.4 GB | ~3-8 GB | ~21-26 GB |

*With gradient checkpointing enabled. Without: multiply activations by ~3-4×.

### Remote A4000 GPU

The remote A4000 (NVIDIA, CUDA) cannot be used for distributed training with the Strix Halo (AMD, ROCm) due to cross-vendor incompatibility. The A4000 continues its role as the inference server for:

- **SFT dataset generation** (`scripts/generate_theme_dataset.py`) — can run concurrently during CPT
- **Embedding computation** (`scripts/process_and_index.py`)
- **Interactive testing** via LM Studio

Set `REMOTE_HOST` in the environment to route inference to the A4000 during training.

---

## Troubleshooting

### "Tokenized corpus not found"

The training data must be tokenized with the correct model's tokenizer. Run:

```bash
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --percent 5
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --finalize
```

### GPU out of memory

Reduce `--micro-batch-size` (try 4, 2, or 1). Ensure `--no-gradient-checkpointing` is NOT set. Reduce `--num-workers` if CPU memory is the bottleneck.

### GPU segfault (exit code 139)

You are likely running outside the fine-tuning container, or using the host venv.  Training **must** run inside the `strix-halo-finetuning` container:

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
# Inside container (bash-5.3$ prompt):
source /opt/venv/bin/activate
python3 scripts/train_deepred_model.py --profile dev
```

> **Common mistake:** Pasting all lines at once into a terminal. The `podman exec -it ... bash` line opens an interactive sub-shell — subsequent lines must be typed **inside** that shell (look for the `bash-5.3$` prompt). Alternatively, use the one-liner:
> ```bash
> podman exec strix-halo-finetuning /opt/venv/bin/python3 -c "import torch; print(torch.cuda.is_available())"
> ```

The host venv and the `llama-rocm-7.2` container only have Python 3.14 with standard PyTorch ROCm wheels that lack gfx1151 GPU code.

### Very slow training / low MFU

- Verify ROCm is working: `python3 -c "import torch; print(torch.version.hip)"`
- Check that training is on GPU (look for "GPU 0:" in the training log, not "CPU")
- Try `--compile` for potential speedup (experimental on ROCm)
- Increase `--micro-batch-size` to improve GPU utilization (if memory allows)

### Loss not decreasing

- Check that corpus data is valid: `python3 -c "import numpy as np; d = np.fromfile('/mnt/data/training_corpus/SmolLM2-360M/train.bin', dtype=np.uint16); print(d.shape, d.min(), d.max())"`
- Verify tokenizer matches the corpus (token IDs should be within the model's vocab range)
- Try reducing learning rate (`--lr 1e-4`)

### Resume not working

Ensure the resume path points to a directory containing `training_state.pt`, `model.safetensors` (or `pytorch_model.bin`), and tokenizer files. This is typically the `latest/` subdirectory:

```bash
python3 scripts/train_deepred_model.py --resume /mnt/data/training_output/cpt-SmolLM2-360M-*/latest
```

---

## Next Steps After Training

After CPT completes, the model needs supervised fine-tuning (SFT) and conversion:

1. **Evaluate the CPT model** — Check `samples.log` and run temporal compliance tests
2. **SFT for Deep Red persona** — LoRA fine-tuning on ChatML data (see ModelTraining.md, Prod Phase 5)
3. **GGUF conversion** — Export for llama.cpp / LM Studio deployment
4. **Deploy and test** — Load in LM Studio for interactive testing

The `final/` checkpoint directory is in standard HuggingFace format and can be used directly with `transformers`, PEFT, or conversion tools.
