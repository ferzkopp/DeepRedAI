# Training the Deep Red Model

## Overview

This document covers running **Continued Pre-Training (CPT)** on a base LLM using the temporally-filtered pre-1969 corpus. The training script (`scripts/train_deepred_model.py`) supports two profiles:

| Profile | Model | Parameters | Purpose | Default Data |
|---------|-------|------------|---------|-------------|
| **dev** | SmolLM2-360M | 360M | Fast iteration & validation | 5% of corpus |
| **prod** | TinyLlama-1.1B | 1.1B | Final production model | 100% of corpus |

Dev mode produces a quick checkpoint for validating that CPT suppresses post-1969 knowledge without destroying language quality. Prod mode runs the full multi-week training.

The script uses **run orchestration** to manage long-running training:
- Each run gets a human-readable name (e.g., `dev-2026-03-07`)
- Re-launching with the same parameters **automatically resumes** from the last checkpoint
- Changing parameters requires a new run name to prevent accidental overwriting
- Completed runs stop cleanly and inform the user
- **GGUF models** are exported at each epoch boundary for testing in LM Studio

See [ModelTraining.md](ModelTraining.md) for background on the CPT approach, base model selection, and the overall training roadmap.

---

## Prerequisites

1. **Strix Halo setup complete** — ROCm, containers, models. See [StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md).
2. **Fine-tuning container created** — `setup_strixhalo.py` stage `training_toolbox` creates the `strix-halo-finetuning` container (image: `kyuz0/amd-strix-halo-llm-finetuning`). This container ships with gfx1151-compiled PyTorch from AMD's TheRock nightly builds.
3. **Base models downloaded** — `setup_strixhalo.py` downloads both SmolLM2-360M and TinyLlama-1.1B to `/mnt/data/models/`.
4. **Training corpus tokenized** — `create_training_corpus.py` produces `train.bin` and `val.bin`. See [TrainingCorpus-Setup.md](TrainingCorpus-Setup.md).
5. **llama.cpp GGUF tools installed** — `setup_strixhalo.py` stage `training_gguf_tools` clones llama.cpp to `/mnt/data/llama.cpp` and installs its Python requirements in the fine-tuning container. This enables automatic GGUF export at epoch boundaries.

---

## Step 1: Enter the Fine-Tuning Container

All training commands must run inside the container. Start it and enter an interactive shell:

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
```

Once inside (`bash-5.3$` prompt), activate the venv:

```bash
source /opt/venv/bin/activate
cd /mnt/data/DeepRedAI
```

Verify PyTorch has ROCm/HIP support:

```bash
python3 -c "import torch; print('PyTorch', torch.__version__, '| HIP', torch.version.hip)"
# Expected: PyTorch 2.12.0a0+rocm7.12.0a20260307 | HIP 7.12.60610 (or similar)
```

> **One-liner alternative** (no interactive shell):
> ```bash
> podman exec strix-halo-finetuning bash -c \
>   'source /opt/venv/bin/activate && cd /mnt/data/DeepRedAI && python3 scripts/train_deepred_model.py'
> ```

> **Important:** The host venv (`/mnt/data/venv`) does NOT have gfx1151-compatible PyTorch and will segfault on GPU operations.

---

## Step 2: Prepare Training Data

The training script reads pre-tokenized binary data (`train.bin` / `val.bin`). Each model requires its own tokenized corpus because tokenizer vocabularies differ (SmolLM2: 49,152 tokens, TinyLlama: 32,000 tokens).

### For dev (SmolLM2-360M)

```bash
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --percent 5
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --finalize
```

For a longer dev run, increase `--percent` (or tokenize 100% and limit at training time with `--data-percent`).

### For prod (TinyLlama-1.1B)

The full corpus should already exist from corpus preparation (see [TrainingCorpus-Setup.md](TrainingCorpus-Setup.md)):

```
/mnt/data/training_corpus/TinyLlama-1.1B/
├── train.bin   (3.6 GB, 1.93B tokens)
├── val.bin     (38 MB, 19.4M tokens)
└── manifest.json
```

---

## Step 3: Run Dev Training

Launch the default dev profile — SmolLM2-360M, 5% data, 3 epochs:

```bash
python3 scripts/train_deepred_model.py
```

This creates a run named `dev-2026-03-07` (using today's date) and outputs to `/mnt/data/training_output/dev-2026-03-07/`. The script shows an estimated completion date and time at startup.

### Quick smoke test

Verify the GPU and pipeline work before committing to a long run:

```bash
python3 scripts/train_deepred_model.py --data-percent 1 --max-steps 100
```

### Estimated dev training times

| Data % | ~Tokens (3 epochs) | ~Time at 1.2K tok/s |
|--------|-------------------|-------------------|
| 1% | ~60M | ~14 hours |
| 5% | ~300M | ~69 hours |
| 10% | ~600M | ~139 hours |
| 100% | ~6B | ~58 days |

### Resume after interruption

Simply re-run the same command:

```bash
python3 scripts/train_deepred_model.py
```

The script detects the existing `dev-2026-03-07` run, verifies parameters match, and **automatically resumes from the last checkpoint**. No `--resume` flag needed.

You can also interrupt gracefully with `Ctrl+C` — the script saves a checkpoint before exiting.

---

## Step 4: Test Intermediate Models

At each epoch boundary, the script exports a GGUF model for testing in LM Studio:

```
/mnt/data/training_output/dev-2026-03-07/
├── gguf/
│   ├── dev-2026-03-07-epoch1.gguf    ← after epoch 1
│   ├── dev-2026-03-07-epoch2.gguf    ← after epoch 2
│   ├── dev-2026-03-07-epoch3.gguf    ← after epoch 3
│   └── dev-2026-03-07-final.gguf     ← after final evaluation
```

Copy a GGUF to LM Studio and test temporal compliance:

```bash
# Copy to LM Studio models directory
cp /mnt/data/training_output/dev-2026-03-07/gguf/dev-2026-03-07-epoch1.gguf \
   /mnt/data/lmstudio/models/
```

GGUF export defaults to `q8_0` quantization, which `convert_hf_to_gguf.py`
can write directly. Lower-bit targets such as `q4_k_m` are converted in two
steps: first to temporary `f16` GGUF, then through `llama-quantize`. Change
with `--gguf-quant q4_k_m` for smaller files. Disable entirely with
`--no-gguf`.

> **Note:** GGUF export uses `convert_hf_to_gguf.py` from llama.cpp. Lower-bit
> quantization also requires the `llama-quantize` binary. Both are prepared by
> `setup_strixhalo.py` stage `training_gguf_tools`. If you need to override the
> path, use `--llama-cpp-path`.

---

## Step 5: Run Production Training

Once dev results look good, launch production training:

```bash
python3 scripts/train_deepred_model.py --profile prod
```

This creates a run named `prod-2026-03-07` with production defaults:

- **Model:** TinyLlama-1.1B
- **Data:** 100% of corpus (1.93B tokens)
- **Epochs:** 5 (~9.6B tokens total)
- **Effective batch:** 128 sequences (262K tokens)
- **LR:** 3e-4 → 3e-5 cosine decay
- **Estimated time:** ~3-5 weeks

Resume after interruption works the same way — re-run the same command.

---

## Step 6: Monitor Training

Training progress is logged to both console and files:

```bash
# Live training log
tail -f /mnt/data/training_output/dev-2026-03-07/train.log

# Metrics (JSON lines — step, loss, lr, tokens/sec, MFU)
cat /mnt/data/training_output/dev-2026-03-07/metrics.jsonl | python3 -m json.tool

# Generated text samples (temporal compliance checks)
less /mnt/data/training_output/dev-2026-03-07/samples.log
```

**What to watch for:**
- **Loss decreasing** — Steady decline indicates learning. Expect faster drops during warmup.
- **Val loss** — Should track train loss. If val loss rises while train loss falls, you are overfitting.
- **Text samples** — Check that generated text is coherent and era-appropriate (pre-1969 content).
- **MFU** — Model FLOPS Utilization. 25-35% is expected for RDNA 3.5.

---

## Run Orchestration

### How runs work

Each training launch is tracked by a **run name** and a **parameter fingerprint**:

| Scenario | What happens |
|----------|-------------|
| First launch | Creates run directory, saves `run_meta.json`, starts training |
| Re-launch, same parameters | Auto-resumes from last checkpoint |
| Re-launch, different parameters | Blocks with a diff of changed parameters |
| Re-launch, training completed | Shows completion message and exits |

### Run naming

By default, runs are named `{profile}-{YYYY-MM-DD}` (e.g., `dev-2026-03-07`). Override with `--run-name`:

```bash
python3 scripts/train_deepred_model.py --run-name dev-experiment-cosine
```

### Starting a new run after completion

When a run finishes, re-launching the same command shows:

```
  Run 'dev-2026-03-07' is COMPLETED
  Finished: 2026-03-10T14:32:00
  Output:   /mnt/data/training_output/dev-2026-03-07
```

To start a fresh run:

```bash
# Auto-increment name (dev-2026-03-07-2, dev-2026-03-07-3, ...)
python3 scripts/train_deepred_model.py --new-run

# Or use a custom name
python3 scripts/train_deepred_model.py --run-name dev-round2
```

### Handling parameter changes

If you change parameters (e.g., `--lr 1e-4`) for an existing run, the script blocks:

```
ERROR: Run 'dev-2026-03-07' exists with different parameters.

Changed parameters:
  lr: 0.0003 -> 0.0001

To start a new run with these parameters, use:
  --run-name <custom-name>
```

Provide a new name to proceed:

```bash
python3 scripts/train_deepred_model.py --lr 1e-4 --run-name dev-lr1e4
```

### Manual resume (bypass orchestration)

The explicit `--resume` flag bypasses run orchestration entirely:

```bash
python3 scripts/train_deepred_model.py --resume /mnt/data/training_output/dev-2026-03-07/latest
```

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

### Run Orchestration Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--run-name NAME` | `{profile}-YYYY-MM-DD` | Custom run name |
| `--new-run` | off | Start new run even if previous is completed (auto-increments name) |

### GGUF Export Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--no-gguf` | off | Disable GGUF export at epoch boundaries |
| `--gguf-quant TYPE` | `q8_0` | Output type or quantization target (`f16`, `bf16`, `q8_0`, `q4_k_m`, etc.; lower-bit targets require `llama-quantize`) |
| `--llama-cpp-path PATH` | `$DEEPRED_ROOT/llama.cpp` | Path to llama.cpp directory |

### Performance Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--compile` | off | Use `torch.compile()` for potential speedup (experimental on ROCm) |
| `--no-gradient-checkpointing` | off | Disable gradient checkpointing (faster but uses more memory) |
| `--num-workers N` | auto | DataLoader workers (auto = min(cpu_count, 8); 0 = single-process) |
| `--attn-implementation` | auto | Attention implementation (auto/sdpa/eager/flash_attention_2) |

### Path Overrides

| Flag | Default |
|------|---------|
| `--model-path` | `/mnt/data/models/{model_name}/` |
| `--corpus-dir` | `/mnt/data/training_corpus/{model_name}/` |
| `--output-dir` | `/mnt/data/training_output/{run_name}/` |

---

## Output Structure

```
/mnt/data/training_output/dev-2026-03-07/
├── run_meta.json         # Run orchestration state (name, fingerprint, status)
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
├── epoch-1/              # Model-only checkpoint after epoch 1
├── epoch-2/              # Model-only checkpoint after epoch 2
├── checkpoint-1000/      # Named checkpoint at step 1000
├── final/                # Final model after training completes
└── gguf/                 # GGUF exports for LM Studio testing
    ├── dev-2026-03-07-epoch1.gguf
    ├── dev-2026-03-07-epoch2.gguf
    └── dev-2026-03-07-final.gguf
```

- **`latest/`** contains the optimizer state for resume capability (~3× model size due to Adam state).
- **Named checkpoints** (`epoch-*`, `checkpoint-*`, `best/`, `final/`) contain only model weights and tokenizer — lightweight and directly usable for inference or GGUF conversion.
- **`run_meta.json`** tracks run status (`running` / `completed`), the parameter fingerprint, and timestamps.

---

## Backup the Trained Model

After a successful run, upload the exported `.gguf` model to the same remote
server used for the chess corpus backups. The `backup_deepred_files.py` script
reuses the **same saved connection settings and secure password** (stored in the
keyring), so no re-entry of credentials is needed if you have run a backup
before. Models are uploaded to the same remote `/Data` folder.

```bash
source /mnt/data/DeepRedAI/deepred-env.sh

# Auto-detect and upload the most recently modified .gguf under
# $DEEPRED_ROOT/training_output
python3 scripts/backup_deepred_files.py --gguf

# Or upload a specific model file
python3 scripts/backup_deepred_files.py --gguf \
    $DEEPRED_ROOT/training_output/<run-name>/gguf/<run-name>-final.gguf

# Preview the planned upload without connecting
python3 scripts/backup_deepred_files.py --gguf --dry-run
```

The upload writes to a temporary `.uploading` file and atomically renames it on
completion, overwriting any existing file of the same name and verifying the
transferred size. Per-file progress is displayed during upload.

> **Note:** The first time you run any backup, the script prompts for the host,
> username, password, and target folder, then stores the non-secret settings in
> `~/.config/deepredai/backup_upload.json` and the password in the system
> keyring. See [ChessAugmentation-Setup.md](ChessAugmentation-Setup.md) for full
> details on credential storage and the `--save-password` modes.

---

## Hardware Utilization

### Strix Halo iGPU (Primary)

The training script runs entirely on the Strix Halo integrated GPU via ROCm. Key optimizations:

- **BF16 mixed precision** — Forward pass in BF16 via `torch.autocast`, FP32 master weights for optimizer stability.
- **Gradient checkpointing** — Trades ~30% extra compute time for up to 60% memory savings on activations.
- **Fused AdamW** — Entire Adam update in a single GPU kernel.
- **CPU data loading** — Multiple DataLoader workers with `persistent_workers` and `prefetch_factor=4`.

The fine-tuning container sets ROCm environment variables automatically:

```bash
export ROCM_PATH=/opt/rocm-7.0
export HSA_OVERRIDE_GFX_VERSION=11.0.0    # Required for gfx1151
export ROCBLAS_USE_HIPBLASLT=1              # Optimized matrix math
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

### Memory Budget

With 128 GB unified memory, training comfortably fits both models:

| Model | FP32 Weights | Optimizer (Adam) | Gradients | Activations* | Total |
|-------|-------------|-----------------|-----------|-------------|-------|
| SmolLM2-360M | 1.4 GB | 2.8 GB | 1.4 GB | ~1-3 GB | ~7-9 GB |
| TinyLlama-1.1B | 4.4 GB | 8.8 GB | 4.4 GB | ~3-8 GB | ~21-26 GB |

\*With gradient checkpointing enabled. Without: multiply activations by ~3-4×.

### Remote A4000 GPU

The A4000 cannot be used for distributed training with the Strix Halo due to cross-vendor incompatibility. It continues its role as the inference server for SFT dataset generation, embedding computation, and interactive testing via LM Studio.

---

## Troubleshooting

### GPU segfault (exit code 139)

You are likely running outside the fine-tuning container. Training **must** run inside `strix-halo-finetuning`:

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
source /opt/venv/bin/activate
cd /mnt/data/DeepRedAI
python3 scripts/train_deepred_model.py
```

> **Common mistake:** Pasting all lines at once. The `podman exec -it ... bash` opens a sub-shell — wait for the `bash-5.3$` prompt before typing further commands.

### "Tokenized corpus not found"

Run `create_training_corpus.py` for the correct model:

```bash
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --percent 5
python3 scripts/create_training_corpus.py --tokenizer SmolLM2-360M --finalize
```

### GPU out of memory

Reduce `--micro-batch-size` (try 4, 2, or 1). Ensure `--no-gradient-checkpointing` is NOT set. Reduce `--num-workers` if CPU memory is the bottleneck.

### Very slow training / low MFU

- Verify ROCm: `python3 -c "import torch; print(torch.version.hip)"`
- Check training is on GPU (look for "GPU 0:" in the log, not "CPU")
- Try `--compile` for potential speedup
- Increase `--micro-batch-size` to improve GPU utilization (if memory allows)

### Loss not decreasing

- Validate corpus data: `python3 -c "import numpy as np; d = np.fromfile('/mnt/data/training_corpus/SmolLM2-360M/train.bin', dtype=np.uint16); print(d.shape, d.min(), d.max())"`
- Verify tokenizer matches the corpus (token IDs should be within vocab range)
- Try reducing learning rate (`--lr 1e-4`)

### GGUF export skipped

The script logs a warning if llama.cpp is not found. Re-run the setup stage:

```bash
sudo -E python3 scripts/setup_strixhalo.py --stage training_gguf_tools
```

This clones llama.cpp to `/mnt/data/llama.cpp` and installs Python requirements in the container. To override the path at runtime: `--llama-cpp-path /path/to/llama.cpp`

---

## Next Steps After Training

After CPT completes:

1. **Evaluate** — Check `samples.log` and test the `final.gguf` in LM Studio for temporal compliance
2. **SFT for Deep Red persona** — LoRA fine-tuning on ChatML data (see ModelTraining.md, Prod Phase 5)
3. **Deploy** — The GGUF models in `gguf/` are ready for LM Studio; the `final/` directory is standard HuggingFace format for `transformers` or PEFT
