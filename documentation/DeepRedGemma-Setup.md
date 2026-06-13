# Training Deep Red on Gemma-3 (SFT)

Parallel training path to [DeepRedModel-Setup.md](DeepRedModel-Setup.md). That
doc covers **continued pre-training (CPT)** of small base models on the packed
uint16 corpus; this one covers **supervised fine-tuning (SFT)** of Gemma-3
instruct models with TRL `SFTTrainer`, derived from
[kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning).

## Why a second track?

1. **Throughput.** CPT on TinyLlama-1.1B over the 16 B-token corpus projects
   to ~259 days wall-clock. Each hyperparameter iteration is a multi-month
   commitment.
2. **Headroom.** Strix Halo's 128 GB unified memory makes Gemma-3-4B/12B
   viable for full fine-tuning. Starting from an instruct base skips the
   "learn English" phase and goes straight to behavioural alignment.

### Approach comparison

| | **CPT** | **Full SFT** | **LoRA SFT** |
|---|---|---|---|
| Script | `train_deepred_model.py` | `train_deepred_gemma.py --type full` | `train_deepred_gemma.py --type lora` |
| Base | SmolLM2-360M / TinyLlama-1.1B | Gemma-3-4B-IT / 12B-IT | Same (+ 27B-IT viable) |
| Trainable params | 100% | 100% | ~2% (adapters) |
| Data format | `train.bin` (packed) | `train.jsonl` (chat) | `train.jsonl` (chat) |
| Knowledge cutoff control | Strong (pre-1969 only) | Indirect (prompt curation) | Indirect |
| Output | Full weights → GGUF | Full weights → GGUF | Adapters + merged → GGUF |

**Pick guide:**
- Maximum knowledge-cutoff control → **CPT** on TinyLlama.
- Capable chat model fast → **Full SFT** on Gemma-3-4B-IT.
- Fast iteration / experiment with 12B-27B → **LoRA SFT** + `--unsloth`.

The two tracks are independent and can run side-by-side.

---

## Prerequisites

1. **Strix Halo setup complete** — see [StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md).
2. **`strix-halo-finetuning` container present** — provisioned by
   `setup_strixhalo.py` (stage `training_toolbox`). Ships gfx1151 PyTorch
   from AMD TheRock nightly + TRL/peft/bitsandbytes/Unsloth built for gfx1151.
3. **GTT cap set to 96 GiB** in GRUB:
   `amd_iommu=off amdgpu.gttsize=98304 ttm.pages_limit=25165824`.
   Leaves ~32 GiB host headroom — required for `max_length=2048` workloads.
4. **zram swap disabled** (`setup_strixhalo.py --stage gtt_memory` does both).
5. **HuggingFace Gemma license accepted** in your browser:
   - https://huggingface.co/google/gemma-3-4b-it
   - https://huggingface.co/google/gemma-3-12b-it
6. **HF token** at `~/hf_token.txt` (read by `deepred-env.sh`):

   ```bash
   echo 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' > ~/hf_token.txt
   chmod 600 ~/hf_token.txt
   source deepred-env.sh   # exports HF_TOKEN
   ```

7. **DeepRed corpus sources present** (same as `create_training_corpus.py`).

---

## Step 1 — Download Gemma-3 weights

On the host (downloads to `$DEEPRED_MODELS`, default `/mnt/data/models`):

```bash
source deepred-env.sh
python3 scripts/download_gemma_models.py --model gemma-3-4b-it
# or both:
python3 scripts/download_gemma_models.py --model both
```

---

## Step 2 — Build the SFT dataset

Converts existing DeepRed corpus sources into chat-format JSONL
(`{"messages": [...]}` per line):

```bash
# Smoke build (small, fast — for pipeline validation)
python3 scripts/build_sft_dataset.py \
    --sources year_topics,augmented_chess_games \
    --max-samples-per-source 50 \
    --tag smoke

# Full build for training
python3 scripts/build_sft_dataset.py --tag v1
```

Output at `$DEEPRED_ROOT/sft_corpus/<tag>/`: `train.jsonl`, `val.jsonl`,
`manifest.json`. Default sources are
`year_topics,gutenberg,augmented_chess_games,chess_books`.

### Balanced build for the next Gemma run

The 2026-05-23-5 run used `/mnt/data/sft_corpus/v1`, whose manifest was almost
entirely `augmented_chess_games`. Always audit the dataset before training:

```bash
python3 scripts/audit_sft_dataset.py /mnt/data/sft_corpus/v1
```

Build new training sets with explicit per-source caps. A first balanced
candidate should include Wikipedia as the broad anchor, keep raw PGN out, and
cap augmented chess to a small flavor share:

```bash
python3 scripts/build_sft_dataset.py \
   --sources wikipedia_articles,year_topics,gutenberg,augmented_chess_games,chess_books \
   --source-limits augmented_chess_games=10000 \
   --tag balanced-v1

python3 scripts/audit_sft_dataset.py /mnt/data/sft_corpus/balanced-v1
```

Tune the `augmented_chess_games` cap until the manifest and audit report show a
target chess share, normally 2-5% for a general DeepRed assistant. The manifest
now records split-level source counts and character totals so train/validation
balance can be checked quickly.

---

## Step 3 — Enter the fine-tuning container

GPU training **must** run inside `strix-halo-finetuning` — the host venv
segfaults on `.cuda()` for gfx1151.

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
# inside the container:
source /opt/venv/bin/activate
cd /mnt/data/DeepRedAI
```

---

## Step 4 — Smoke test

5 optimizer steps, no GGUF — verifies model loads, GPU works, loss is finite.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/smoke \
    --epochs 1 --max-steps 5 --no-gguf --debug \
    --gradient-checkpointing
```

Healthy signs: finite (non-NaN) loss, peak GPU memory ~45-50 GB, ~25-30 s/step
after the first warm-up step.

> The "PAD/BOS/EOS tokens differ from the model config" warning is expected
> for Gemma-3 — TRL re-aligns the special tokens. Harmless.

---

## Step 5 — Run full SFT

### Gemma-3-4B-IT Full FT (validated production path)

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --gradient-checkpointing
```

This is the **only Full-FT configuration confirmed to train end-to-end
on a 96 GiB GTT cap with finite loss**. Defaults: batch=4, grad_accum=4
(effective batch 16), lr=5e-5, 2 epochs, max_length=2048.

> Do **not** add `--unsloth` for Full FT (4B) on this hardware — see
> Troubleshooting. `--gradient-checkpointing` alone is the working path.

For new production attempts, prefer a balanced dataset, a lower learning rate
for the first full run, and progress GGUF snapshots for manual quality gates:

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
   --dataset-dir /mnt/data/sft_corpus/balanced-v1 \
   --epochs 1 --lr 2e-5 \
   --gradient-checkpointing \
   --snapshot-fractions 10,20,30,40,50,60,70,80,90 \
   --snapshot-gguf-quant q4_k_m \
   --run-name gemma-4b-balanced-v1
```

Full fine-tune snapshots are saved as temporary HF model directories, converted
to GGUF with `llama.cpp`, recorded in `run_meta.json`, and removed after a
successful conversion unless `--keep-snapshot-hf` is set. LoRA progress
snapshots are saved as adapters; GGUF export is skipped until they are merged.

### Gemma-3-12B-IT Full FT

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-12b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --gradient-checkpointing
```

Profile defaults: batch=1, grad_accum=16 (effective batch 16), lr=2e-5,
2 epochs, max_length=2048.

### LoRA fine-tuning

LoRA trains only ~2% of parameters (adapters into q/k/v/o/gate/up/down).
~2× faster, ~40-50% less memory than Full FT; makes Gemma-3-27B viable.

```bash
# 4B LoRA — fastest iteration
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --type lora --unsloth

# 12B LoRA + Unsloth
python3 scripts/train_deepred_gemma.py --profile gemma-12b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --type lora --unsloth
```

Adapter hyperparameters (hard-coded in
[`scripts/train_deepred_gemma.py`](../scripts/train_deepred_gemma.py)):
`r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias='none'`,
`task_type='CAUSAL_LM'`, targets =
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.

LoRA outputs: adapters in `final/` (small), merged full-precision weights
in `final-merged/` (used for GGUF export).

### Resume after interruption

Re-run the **exact same command** with `--run-name <existing-name>`
(or the auto-generated `<profile>-<YYYY-MM-DD>` if running same day):

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --gradient-checkpointing --run-name gemma-4b-2026-05-23-5
```

The script compares the fingerprint in `run_meta.json` against your CLI args
— if they match and a `checkpoint-NNNN/` exists, HF Trainer resumes from
the latest checkpoint (model weights, optimizer state, LR scheduler, RNG,
dataloader position). By default checkpoints are saved at epoch boundaries.
Use `--save-strategy steps --save-steps <N>` for more frequent resumable
checkpoints. `--save-total-limit 2` keeps disk bounded (~30-35 GB per Full-FT
4B checkpoint).

Force a fresh run with `--new-run` (auto-increments name) or
`--run-name <custom>`.

---

## Step 6 — Outputs

```
$DEEPRED_ROOT/training_output/<run-name>/
├── checkpoint-NNN/         (HF Trainer checkpoints — full save)
├── snapshots/              (temporary/model-only progress snapshots)
├── final/                  (final model + tokenizer; or LoRA adapters)
├── final-merged/           (LoRA only: merged base+adapters)
├── gguf/<name>-final.gguf  (for LM Studio; q8_0 default)
├── gguf/<name>-010pct-step-N.gguf  (optional progress snapshots)
├── train.log
├── memory.log              (GPU + host memory samples, every 10 s)
└── run_meta.json           (fingerprint, status, params, snapshot metadata)
```

Copy GGUF to LM Studio:

```bash
cp $DEEPRED_ROOT/training_output/<run-name>/gguf/*.gguf \
   $DEEPRED_ROOT/lmstudio/models/
```

Change quant with `--gguf-quant q4_k_m`; disable export with `--no-gguf`.

---

## Step 7 — Document the run

After training (or at any point during it), generate a Markdown summary of the
run — the source model, training parameters, results, and produced artifacts:

```bash
# Resolve the run from its name/profile (default: <profile>-<YYYY-MM-DD>)
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --run-name gemma-4b-2026-05-23-5 --summary

# Or point directly at the output directory
python3 scripts/train_deepred_gemma.py \
    --output-dir $DEEPRED_ROOT/training_output/gemma-4b-2026-05-23-5 \
    --summary

# Write to a file instead of stdout
python3 scripts/train_deepred_gemma.py \
    --output-dir $DEEPRED_ROOT/training_output/gemma-4b-2026-05-23-5 \
    --summary --summary-file run-summary.md
```

`--summary` does not load the model or GPU — it just reads on-disk metadata, so
it runs quickly outside the container. It combines three sources:

- **`run_meta.json`** — source model, profile, training mode (full/LoRA),
  and all run-defining hyperparameters (epochs, batch size, grad accumulation,
  learning rate, scheduler, warmup, max length, gradient checkpointing, seed,
  dataset dir). On a completed run it also records a `results` block (duration,
  peak GPU memory, final train loss, global steps, epochs).
- **`trainer_state.json`** (from `final/` or the latest checkpoint) — used as a
  fallback for step/epoch counts and to pull the last/best eval loss from the
  loss history.
- **The output directory** — inspected for produced artifacts (`final/`,
  `final-merged/`, `gguf/*.gguf`) and checkpoint range, with file sizes.

Everything is best-effort: missing pieces are omitted, so the summary works for
in-progress runs too (it reports `status: running` and whatever metrics exist).

Example output:

```markdown
# DeepRed SFT Run Summary — gemma-4b-2026-05-23-5

- **Status:** completed
- **Started:** 2026-05-23T05:54:09
- **Completed:** 2026-06-06T04:51:43

## Source Model

- **Profile:** gemma-4b
- **Base model:** google/gemma-3-4b-it
- **Training mode:** full

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Effective batch | 16 |
| Learning rate | 5e-05 |
| ...

## Results

- **Global steps:** 40,076
- **Best eval loss:** 0.7040

## Artifacts

- **Final model:** `.../final` (8.0 GB)
- **GGUF:** `.../gguf/gemma-4b-2026-05-23-5-final.gguf` (3.8 GB)
```

---

## Step 8 — Backup the trained model

Upload the exported `.gguf` to the same remote server used for the chess corpus
backups. The `backup_deepred_files.py` script reuses the **same saved connection
settings and secure password** (stored in the keyring), so no re-entry of
credentials is needed if you have run a backup before. Models go to the same
remote `/Data` folder.

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

## Memory & wall-clock reference

Observed on Strix Halo at the 96 GiB GTT cap, `max_length=2048`, DeepRed
`v1` corpus (320,607 train examples → 20,037 steps/epoch, 40,076 steps
for 2 epochs):

| Config | Status | s/step | Per epoch | 2 epochs |
|---|---|---|---|---|
| **4B Full FT, `--gradient-checkpointing`** | ✅ Working | ~29 | ~6.8 d | **~13.6 d** (+ ~15-16 d w/ eval+save overhead) |
| 4B Full FT, no flags | ❌ OOM ~step 11 | — | — | — |
| 4B Full FT, `--unsloth` alone | ❌ OOM ~step 3 (fused-CE spike) | — | — | — |
| 4B Full FT, `--gradient-checkpointing --unsloth` | ❌ NaN loss from step 1 | — | — | — |
| 4B LoRA, no unsloth | Untested at scale | ~12-15 (scaled) | ~3 d | **~6 d** |
| 4B LoRA, `--unsloth` | Untested at scale | ~8-10 (scaled) | ~2 d | **~4 d** |
| 12B Full FT, `--gradient-checkpointing` | Untested at scale | ~80-100 (scaled) | ~20 d | **~40 d** |

Memory at steady state for the validated 4B Full FT config:
**GPU peak ~46 GB**, host `mem_avail` ~48 GB. Comfortable on both axes.

**Practical takeaway**: 4B Full FT on the full `v1` corpus is a ~2-week
commitment. For first end-to-end production attempts, consider
`--epochs 1` (halves wall-clock) or `--max-steps N` to validate the loss
curve before committing the full budget. LoRA + Unsloth is the right
choice if iteration speed matters more than full-weight updates.

---

## Practical notes (AMD / ROCm / Gemma)

- **`attn_implementation="eager"`** is mandatory for Gemma training
  (soft-capping breaks FlashAttention). Hard-coded in the script.
- **bf16 only.** ROCm fp16 is unreliable for these models.
- **`adamw_torch_fused`** optimizer, **cosine LR + 100-step warmup**.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is set
  automatically by the script (reduces allocator fragmentation).
- **`ddp_find_unused_parameters=True`** because Gemma-3 12B+ are
  multimodal and the vision branch is unused in text-only training.
- **`--unsloth`** works only inside `strix-halo-finetuning`. **Use only
  for LoRA on this hardware** — Full FT NaN'd in testing despite the
  `full_finetuning=True` fix (see Troubleshooting).
- **Gemma-3-27B Full FT** is infeasible even on 128 GB. Use
  `--type lora --model /path/to/gemma-3-27b-it` with the `gemma-12b`
  profile, reducing `--max-length` as needed.

---

## Troubleshooting

**"ERROR: model not found at /mnt/data/models/gemma-3-4b-it"**
Run `download_gemma_models.py` (Step 1).

**"ERROR: dataset not found at ..."**
Run `build_sft_dataset.py` (Step 2). Verify `train.jsonl` + `val.jsonl`
in `--dataset-dir`.

**Segfault on `.cuda()`**
You're outside the container. Re-enter:
`podman exec -it strix-halo-finetuning bash && source /opt/venv/bin/activate`.

**`GatedRepoError` during download**
Accept the Gemma license in your browser, retry.

**`torch.OutOfMemoryError` mid-run on 4B Full FT**

The 4B profile peak at `max_length=2048` exceeds the 96 GiB cap once
long-sequence batches start appearing. Confirmed cases:

- No flags → OOM around step 11 (long-batch activation spike).
- `--unsloth` alone → OOM around step 3 (Unsloth fused-CE adds ~7 GB
  during backward; not caught by `--gradient-checkpointing`).

**Fix: always pass `--gradient-checkpointing`** for 4B Full FT. The
script keeps the kwarg opt-in but it should be considered mandatory at
this GTT cap. Peak drops to ~46 GB with ~50 GB headroom.

**Loss is `nan` from step 1 with `--unsloth` on Full FT**

Two distinct root causes encountered:

1. Before 2026-05-23: `FastLanguageModel.from_pretrained` defaulted to
   LoRA kernels (`Switching to 16bit LoRA`); the manual
   `requires_grad_(True)` reset re-enabled gradients but left the wrong
   kernel path in place. Fixed by passing
   `full_finetuning=(args.type != 'lora')` to the loader.
2. After the fix: even with `Using bfloat16 full finetuning` correctly
   logged and `--gradient-checkpointing` on, the Unsloth fused-CE path
   still produces NaN on Gemma-3 / gfx1151 for Full FT. Root cause
   unclear (likely numerical issue in the fused chunked CE on
   soft-capped logits).

**Workaround**: drop `--unsloth` for Full FT. The plain HF path with
`--gradient-checkpointing` trains cleanly. `--unsloth` remains useful
for LoRA, where the fused-CE path is not on the critical gradient path.

**Run killed with bare `Killed` (no Python traceback)**

Kernel OOM killer. Check `memory.log` next to `train.log`:

```bash
tail -f /mnt/data/training_output/<run>/memory.log
# columns: ts, gpu_alloc_gb, gpu_reserved_gb, gpu_peak_gb,
#         rss_gb, mem_avail_gb, mem_free_gb, swap_used_gb, swap_total_gb
```

If `mem_avail_gb` ≈ 0 before the kill → host RAM exhaustion. Confirm:

```bash
sudo journalctl -k --since "1 hour ago" | grep -iE "oom|killed process"
systemctl is-active systemd-oomd     # should be 'inactive'
```

Fixes (in order):
1. Ensure GTT cap is 96 GiB (prerequisites #3). Default Strix Halo
   `gttsize=131072` leaves only ~4 GiB hard-reserved for the host.
2. Ensure zram swap is off (prerequisites #4). zram makes unified-memory
   pressure worse, not better. If `systemd-oomd` is active, disable it:
   `sudo systemctl disable --now systemd-oomd systemd-oomd.socket`.
3. Lower `--max-length 1024` or `--batch-size 2 --grad-accum 8` (keeps
   effective batch 16).

**Existing CPT training still running**
Independent — different output dir, different script, different
container. The CPT pipeline (`train_deepred_model.py`) is untouched.
