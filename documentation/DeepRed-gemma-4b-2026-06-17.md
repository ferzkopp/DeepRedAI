# DeepRed Gemma 4B Temporal Run — 2026-06-17 (10-Day Full Scale)

## Goals

This is the **full 10-day** Gemma-3-4B SFT fine-tune that the 06-14 runbook was
calibrated for. The 06-14 attempt (`gemma-4b-temporal-v1-10d`, documented in
[DeepRed-gemma-4b-2026-06-14.md](DeepRed-gemma-4b-2026-06-14.md)) ran correctly but
with an inadvertent **10× under-run** — `--max-steps 2500` instead of ~25,000 —
finishing in ~24 h at 0.11 epoch. That short run is now repurposed as a precise
**throughput measurement**. This run:

- **Reuses the already-built corpus** `/mnt/data/sft_corpus/temporal-v1-10d` (sized
  for ~1 epoch ≈ 10 days). No regeneration, no rebuild.
- Sets `--max-steps` from the **measured** 06-14 throughput (33.7 s/step) so the
  wall-clock lands at ~10 days (inside the 7–14 day window).
- Keeps the locked temporal-cutoff design: cutoff **1969-07-20**, `retain` ~25%,
  `unlearn` ~15% refusals, `wikipedia_articles` ~55%.
- **Decouples GGUF export** from training — trains with `--no-gguf
  --no-snapshot-gguf` inside the container, then converts on the **host** with
  [scripts/export_gguf.py](../scripts/export_gguf.py). The host-built `llama.cpp`
  fails *inside* the container with a GLIBC mismatch (`GLIBC_2.43 not found`).
- **Omits `--new-run`** so an interrupted multi-day run **auto-resumes** from the
  latest checkpoint.

---

## Prerequisites

- **Corpus already built.** `/mnt/data/sft_corpus/temporal-v1-10d/{train,val}.jsonl`
  and `manifest.json` exist from the 06-14 prep. **No LLM inference server,
  PostgreSQL, or rebuild is required** for this run.
- **Fine-tuning container** `strix-halo-finetuning` available (gfx1151 PyTorch from
  TheRock; the host venv segfaults on `.cuda()`).
- **Host `llama.cpp`** built at `/mnt/data/llama.cpp/build/bin/llama-quantize`, with
  the matching `gguf` Python package in the host venv
  (`pip install /mnt/data/llama.cpp/gguf-py`) for the Phase 6 export.

> If the corpus is missing, first rebuild `temporal-v1-10d` via Phases 1–6 of
> [DeepRed-gemma-4b-2026-06-14.md](DeepRed-gemma-4b-2026-06-14.md).

---

## 1. Set Up Environment

Run from the repo root on the host:

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh
```

---

## 2. Re-Audit the Reused Dataset

Confirm the corpus is intact and the source mix / refusal share are unchanged:

```bash
python3 scripts/audit_sft_dataset.py /mnt/data/sft_corpus/temporal-v1-10d
grep -ic "covid\|pandemic\|trump\|ukraine" \
      /mnt/data/sft_corpus/temporal-v1-10d/train.jsonl
```

Expected mix: ~55% `wikipedia_articles`, ~25% `retain`, ~15% `unlearn` refusals,
with chess / year_topics / gutenberg making up the remainder; refusal share ≈ 15%.
Do not proceed if chess dominates or the refusal share is far from 15%.

---

## 3. Measured Calibration

Anchor from the **06-14 run** (gemma-4b full fine-tune, effective batch 16, max
length 2048, gradient checkpointing on): **2,500 steps in 84,285 s** (includes one
9,228 s end-eval).

- Overall: `84,285 / 2,500 =` **33.7 s/step**.
- Pure training (excl. end-eval): `(84,285 − 9,228) / 2,500 =` **30.0 s/step**.

Using the conservative **overall 33.7 s/step**:

| Wall-clock | Seconds | Steps @ 33.7 s/step | Examples (×16) | Epochs of `T=372k` |
|-----------|--------:|--------------------:|---------------:|-------------------:|
| 7 days | 604,800 | 17,950 | 287,200 | 0.77 |
| **10 days** | **864,000** | **25,640** | **410,240** | **1.10** |
| 14 days | 1,209,600 | 35,890 | 574,240 | 1.54 |

**`TARGET_MAX_STEPS = 25600`** (≈ 10 days; ~1.10 epoch of `T`). Because the pure
training rate is ~30 s/step, the real finish is likely slightly **under** 10 days —
safely inside the 7–14 day window.

---

## 4. Enter the Fine-Tuning Container

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
```

Inside the container:

```bash
source /opt/venv/bin/activate
cd /mnt/data/DeepRedAI
source deepred-env.sh
```

---

## 5. Run the Final Fine-Tune

GGUF conversion is deferred to Phase 6 (host-side). The trainer saves the final
model and five snapshot HF directories only.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --type full \
    --dataset-dir /mnt/data/sft_corpus/temporal-v1-10d \
    --epochs 3 \
    --max-steps 25600 \
    --lr 2e-5 \
    --lr-scheduler-type cosine \
    --warmup-steps 300 \
    --max-length 2048 \
    --gradient-checkpointing \
    --save-strategy steps \
    --save-steps 2000 \
    --save-total-limit 2 \
    --snapshot-fractions 10,25,50,75,100 \
    --snapshot-gguf-quant q4_k_m \
    --no-snapshot-gguf \
    --no-gguf \
    --run-name gemma-4b-temporal-v1-10d-2
```

Why these settings:

- **`--max-steps 25600`** binds the ~10-day ceiling at the measured 33.7 s/step.
  `--epochs 3` is headroom so the **step ceiling — not the epoch loop — ends
  training** (1 epoch ≈ 23,250 steps at `T`).
- **No `--new-run`.** Re-running this exact command auto-resumes from the latest
  `checkpoint-N` (same run fingerprint), which matters across a multi-day run. Add
  `--new-run` only to deliberately start a fresh, auto-incremented run.
- **`--no-gguf --no-snapshot-gguf`** defer all GGUF conversion to the host (Phase 6).
  Snapshots are saved as HF dirs at steps 2,560 / 6,400 / 12,800 / 19,200 / 25,600;
  `--snapshot-gguf-quant q4_k_m` records the intended quant for that step.
- **`--lr 2e-5`, cosine, `--warmup-steps 300`, `--max-length 2048`** — the stable
  06-14 configuration; cosine decays over `--max-steps`.
- **`--save-steps 2000`, `--save-total-limit 2`** keep two resumable checkpoints.

Outputs (no GGUF yet):

```text
/mnt/data/training_output/gemma-4b-temporal-v1-10d-2/
├── final/        # final full-precision weights
└── snapshots/    # five progress snapshots (HF dirs)
```

Confirm `run_meta.json` shows `status: completed`, five snapshots + final.

---

## 6. Export GGUF on the Host (Post-Step)

**Run on the host — NOT inside the container** (host-built `llama.cpp` fails inside
the container with `GLIBC_2.43 not found`).

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh

# One-time: the converter needs the gguf python package matching llama.cpp.
python3 -c "import gguf" 2>/dev/null || pip install /mnt/data/llama.cpp/gguf-py

python3 scripts/export_gguf.py --run-name gemma-4b-temporal-v1-10d-2
```

[scripts/export_gguf.py](../scripts/export_gguf.py) reads `run_meta.json`, exports
the final model and every snapshot to GGUF, and records each `export_status`. It is
safe to re-run — existing GGUFs are skipped (`--all` forces re-export; `--cleanup-hf`
removes the large HF snapshot dirs after a successful conversion).

Output (final + five snapshots, `q4_k_m`):

```text
/mnt/data/training_output/gemma-4b-temporal-v1-10d-2/gguf/
├── gemma-4b-temporal-v1-10d-2-final.gguf
├── gemma-4b-temporal-v1-10d-2-010pct-step-2560.gguf
├── gemma-4b-temporal-v1-10d-2-025pct-step-6400.gguf
├── gemma-4b-temporal-v1-10d-2-050pct-step-12800.gguf
├── gemma-4b-temporal-v1-10d-2-075pct-step-19200.gguf
└── gemma-4b-temporal-v1-10d-2-100pct-step-25600.gguf
```

---

## 7. Back Up the Final GGUF

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh

python3 scripts/backup_deepred_files.py \
    --gguf /mnt/data/training_output/gemma-4b-temporal-v1-10d-2/gguf/gemma-4b-temporal-v1-10d-2-final.gguf
```

---

## Locked Decisions

| Decision | Value |
|----------|-------|
| Runtime anchor | 10 days (7–14 day window) |
| Throughput anchor | 33.7 s/step (measured, 06-14 run) |
| `--max-steps` | 25,600 (≈ 10 days, ~1.10 epoch of `T`) |
| Corpus | reuse `temporal-v1-10d` (`T ≈ 372,000`, already built) |
| Temporal cutoff | 1969-07-20 |
| Source mix | wiki ~55% · retain ~25% · unlearn ~15% · chess/year/gutenberg remainder |
| Profile / type | `gemma-4b` / full fine-tune |
| LR / scheduler / warmup | 2e-5 / cosine / 300 |
| Max length | 2048 |
| Epoch headroom | 3 (bounded by `--max-steps`) |
| Run name | `gemma-4b-temporal-v1-10d-2` (no `--new-run`; auto-resumes) |
| Snapshots | 10/25/50/75/100% (HF dirs → `q4_k_m` GGUF in Phase 6) |
| GGUF export | host post-step via `export_gguf.py` (deferred from training; container GLIBC) |

---

## Examples

> Populate after the run completes, mirroring the format in
> [DeepRed-gemma-4b-2026-06-13.md](DeepRed-gemma-4b-2026-06-13.md). Include at least
> one **retain** prompt (a pre-1969 fact the model should answer) and one **unlearn**
> prompt (a post-1969/modern topic the model should refuse), to demonstrate the
> temporal cutoff.
