# DeepRed Gemma 4B Temporal Run — 2026-06-14

> **Status:** completed as a **2,500-step (~24 h) under-run** — `--max-steps` was set
> 10× too low (2,500 instead of the calibrated ~25,000). The run is valid and its
> throughput calibrates the full-scale follow-up,
> [DeepRed-gemma-4b-2026-06-17.md](DeepRed-gemma-4b-2026-06-17.md). See
> [Run Outcome](#run-outcome-actual).

## Goals

This run is a **longer, corpus-calibrated** Gemma-3-4B SFT fine-tune. The previous
run (`gemma-4b-balanced-v1-small-1500`, documented in
[DeepRed-gemma-4b-2026-06-13.md](DeepRed-gemma-4b-2026-06-13.md)) succeeded but used
too little corpus and stopped early (1,500 steps, 0.46 epoch, 15.5 h). This run:

- Targets a **~10-day wall-clock** (inside the acceptable 7–14 day window).
- **Calibrates** the corpus mix, run parameters, and epoch headroom to that runtime.
- **Reintroduces the legacy temporal-cutoff patterns** — `retain` (pre-cutoff
  factual Q&A the model keeps) and `unlearn` (post-cutoff questions the model must
  refuse) — using a fixed cutoff of **1969-07-20** (Apollo 11).
- **Reuses the existing legacy datasets** under `$WIKI_DATA/datasets/{retain,unlearn}`
  rather than regenerating them from scratch.
- **Refreshes modern-event refusals** (e.g. COVID-19, recent elections) by running
  the migrated temporal generator over post-1969 topic files.
- Bounds GPU time with a measured `--max-steps` ceiling and exports progress GGUF
  snapshots for comparison.

The temporal generator was migrated from the legacy LM-Studio-CLI script into the
new framework's HTTP inference convention as
[scripts/generate_temporal_qa.py](../scripts/generate_temporal_qa.py).

> **"Refuse, do not learn."** For post-cutoff (modern) topics, the generator
> discards the factual answer and substitutes a refusal template. The model is
> trained to **decline** questions about post-cutoff events, not to learn their
> content. The discarded answer is retained only in `metadata.original_answer` for
> auditing.

---

## Prerequisites

> **An OpenAI-compatible LLM inference server must be reachable before Phase 1.**
> Phase 1 (modern-refusal generation) calls
> `POST http://{host}:{port}/v1/chat/completions` for every topic. Nothing else in
> this runbook needs the LLM server.

Endpoint resolution (highest priority first), implemented in
[scripts/generate_temporal_qa.py](../scripts/generate_temporal_qa.py):

1. `--inference-host` / `--inference-port` CLI flags.
2. `REMOTE_HOST` / `REMOTE_LLM_PORT` environment variables (remote override).
3. `INFERENCE_HOST` / `INFERENCE_PORT` environment variables (local default
   `localhost:1234`).

Confirm the endpoint before generating (a quick model-list probe):

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh
# Resolve which endpoint will be used and that it answers:
HOST="${REMOTE_HOST:-$INFERENCE_HOST}"; PORT="${REMOTE_LLM_PORT:-$INFERENCE_PORT}"
[ -n "$REMOTE_HOST" ] && PORT="${REMOTE_LLM_PORT:-1234}"
echo "Inference endpoint: http://$HOST:$PORT"
curl -fsS "http://$HOST:$PORT/v1/models" | head -c 400; echo
```

Other prerequisites:

- **PostgreSQL is NOT required** for this run. Phase 1 uses `--topics-only-text`
  (topic text only, no article-body lookups), and the corpus build reuses the
  already-generated retain/unlearn JSONL files. PostgreSQL is only consulted by the
  `wikipedia_articles` source during the build (Phase 4), which is already part of
  the established pipeline.
- **Generation time:** Phase 1 iterates every post-1969 topic (~56 year files) and
  makes 2 LLM calls per topic. It runs to **hours**, is **append-only**, dedupes
  against existing questions, and is **safely re-runnable** — re-running tops up new
  phrasings without duplicating prior pairs.

---

## 1. Set Up Environment

Run from the repo root on the host:

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh
```

---

## 2. Generate Modern-Event Refusals (Unlearn Refresh)

This appends post-cutoff (1970–2025) refusal Q&A — including COVID-19, recent
elections, and other modern events — to `$WIKI_DATA/datasets/unlearn`. Because every
topic after 1969-07-20 classifies as *post-cutoff*, only the **unlearn** set grows;
the **retain** set is untouched.

```bash
python3 scripts/generate_temporal_qa.py \
    --mode topics \
    --topics-start-year 1970 \
    --topics-end-year 2025 \
    --topics-only-text \
    --questions-per-topic 2 \
    --cutoff-date 1969-07-20
```

What it does:

- Loads `year_topics_1970.json` … `year_topics_2025.json`.
- For each topic, asks the LLM for `2` factual Q&A pairs, then **replaces each answer
  with a refusal template** and writes `{instruction, output(refusal), metadata}`.
- Pre-loads all existing questions for deduplication and **appends** to
  `unlearn_train.jsonl` / `unlearn_val.jsonl`.
- Writes a run summary to `$WIKI_DATA/datasets/statistics.json`.

Verify the refresh (counts should increase; modern terms should appear):

```bash
wc -l "$WIKI_DATA"/datasets/unlearn/unlearn_train.jsonl \
      "$WIKI_DATA"/datasets/unlearn/unlearn_val.jsonl
grep -ic "covid\|pandemic\|trump\|ukraine" \
      "$WIKI_DATA"/datasets/unlearn/unlearn_train.jsonl
python3 -c "import json;d=json.load(open('$WIKI_DATA/datasets/statistics.json'));print('unlearn pairs this run:',d['unlearn_qa_pairs'])"
```

> A `--dry-run` (add the flag) reports topic counts without calling the LLM — useful
> to confirm the topic files are found first.

---

## 3. Measure Available Corpus Counts

The corpus is calibrated from the *actual* available counts. As of 2026-06-14:

| Source | Where | Available |
|--------|-------|----------:|
| `retain` (train+val) | `$WIKI_DATA/datasets/retain/` | 92,938 |
| `unlearn` (train+val) | `$WIKI_DATA/datasets/unlearn/` | 114,214 + new |
| `augmented_chess_games` | `$CHESS_DATA/corpus/augmented_chess_games.jsonl` | 334,920 |
| `year_topics` (≤1969, 1 pair/file) | `$WIKI_DATA/topics/` | 1,788 |
| `gutenberg` | `$GUTENBERG_DATA/corpus/gutenberg_corpus.jsonl` | 766 |
| `chess_books` | `$CHESS_DATA/corpus/chess_archive_books.jsonl` | 10 |
| `wikipedia_articles` (pre-cutoff `O`) | PostgreSQL `articles` | 1,931,239 |

Re-measure if the data has changed:

```bash
wc -l "$WIKI_DATA"/datasets/retain/retain_{train,val}.jsonl \
      "$WIKI_DATA"/datasets/unlearn/unlearn_{train,val}.jsonl \
      "$CHESS_DATA"/corpus/augmented_chess_games.jsonl \
      "$GUTENBERG_DATA"/corpus/gutenberg_corpus.jsonl \
      "$CHESS_DATA"/corpus/chess_archive_books.jsonl
```

---

## 4. Compute the Corpus Mix

**Target corpus size** `T = 372,000` examples — the 1-epoch anchor for ~10 days at
the measured 06-13 throughput (see [Runtime Calibration](#runtime-calibration)).

Source rules (from the project corpus table plus the reintroduced temporal sets):

| Source | Rule | Limit | Approx. share |
|--------|------|------:|--------------:|
| `wikipedia_articles` | remainder (broad anchor) | **203,952** | 54.8% |
| `retain` | 100% | *(unlimited)* 92,938 | 25.0% |
| `unlearn` | 15% cap of `T` | **55,800** | 15.0% |
| `augmented_chess_games` | 5% of available | **16,746** | 4.5% |
| `year_topics` | 100% | *(unlimited)* 1,788 | 0.5% |
| `gutenberg` | 100% | *(unlimited)* 766 | 0.2% |
| `chess_books` | 100% | *(unlimited)* 10 | 0.0% |
| **Total** | | **≈ 372,048** | 100% |

Arithmetic:

- `unlearn = round(0.15 × 372,000) = 55,800`
- `augmented_chess_games = round(0.05 × 334,920) = 16,746`
- `wikipedia_articles = 372,000 − (92,938 + 55,800 + 16,746 + 1,788 + 766 + 10)`
  `= 372,000 − 168,048 = 203,952` (rounded to `204,000`)

Notes:

- `retain` + `unlearn` together are **40%** of the corpus — a deliberate, strong
  imprint of the temporal-cutoff behavior. To shift more weight onto Wikipedia,
  cap `retain` the same way (e.g. add `retain=60000` to `--source-limits`) and add
  the freed budget to `wikipedia_articles`.
- The `unlearn` cap samples via **shuffle-then-cap** in the builder, so the modern
  refusals appended in Phase 2 are fairly represented within the 15% slice.
- Sources at 100% are simply omitted from `--source-limits`.

---

## 5. Build the SFT Dataset

```bash
python3 scripts/build_sft_dataset.py \
    --sources wikipedia_articles,retain,unlearn,year_topics,gutenberg,augmented_chess_games,chess_books \
    --source-limits wikipedia_articles=204000,unlearn=55800,augmented_chess_games=16746 \
    --tag temporal-v1-10d \
    --force
```

Output:

```text
/mnt/data/sft_corpus/temporal-v1-10d/
├── train.jsonl
├── val.jsonl
└── manifest.json
```

---

## 6. Audit the Dataset

```bash
python3 scripts/audit_sft_dataset.py /mnt/data/sft_corpus/temporal-v1-10d
```

Expected approximate source mix and refusal share:

| Source | Approx. share |
|--------|--------------:|
| wikipedia_articles | ~55% |
| retain | ~25% |
| unlearn (refusals) | ~15% |
| augmented_chess_games | ~4.5% |
| year_topics | ~0.5% |
| gutenberg | ~0.2% |
| chess_books | ~0.1% |

Checks before training:

- Refusal share is **≈ 15%** (the audit warns if it exceeds `--refusal-warn-share`,
  default 0.20). The `unlearn` source count drives this.
- `retain` pairs are present (non-zero).
- `wikipedia_articles` is the single dominant source.
- Modern refusals survived the build:

```bash
grep -ic "covid\|pandemic\|trump\|ukraine" /mnt/data/sft_corpus/temporal-v1-10d/train.jsonl
```

Do not start training if the audit shows chess as dominant or the refusal share far
from 15%.

---

## 7. Enter the Fine-Tuning Container

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

## 8. Smoke Test and Measure Throughput

Verify the dataset loads with finite loss, and **measure steady-state seconds/step**
to size the final `--max-steps`. This does not export GGUF.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --type full \
    --dataset-dir /mnt/data/sft_corpus/temporal-v1-10d \
    --max-steps 30 \
    --lr 2e-5 \
    --max-length 2048 \
    --gradient-checkpointing \
    --no-gguf \
    --run-name gemma-4b-temporal-v1-10d-smoke \
    --new-run
```

Healthy signs: loss is finite (not `nan`); GPU memory within the Strix Halo range
(~56 GB at this profile); no dataset/chat-template errors.

Compute seconds/step from the run's reported runtime (the first few steps include
compile/warmup overhead, so a 30-step run gives a slightly conservative value):

```bash
python3 - <<'PY'
import json, glob, os
# Read the smoke run's metrics; adjust the path if your output root differs.
root = "/mnt/data/training_output/gemma-4b-temporal-v1-10d-smoke"
meta = json.load(open(os.path.join(root, "run_meta.json")))
rt = meta.get("train_runtime_seconds") or meta.get("results", {}).get("duration_seconds")
steps = meta.get("global_steps") or 30
sps = rt / steps
print(f"runtime={rt:.0f}s steps={steps} -> {sps:.2f} s/step")
print(f"10-day max-steps = {round(864000 / sps)}")
PY
```

If `run_meta.json` is not yet populated, read the trainer's final summary line
(`train_runtime`, `train_samples_per_second`) from the console/log and compute:

- `s_per_step = train_runtime / global_steps`, or equivalently
  `s_per_step = effective_batch (16) / train_samples_per_second`.

Then set the ceiling:

```text
TARGET_MAX_STEPS = round(864000 / s_per_step)
```

At the 06-13 anchor (37.2 s/step) this is **≈ 23,200 steps**. Because retain,
unlearn, and year_topics sequences are short, the real value is typically lower,
yielding **more** steps in 10 days — which is why the final run carries epoch
headroom and lets `--max-steps` bind the wall-clock.

---

## 9. Run the Final Fine-Tune

Replace `<TARGET_MAX_STEPS>` with the value computed in Phase 8.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --type full \
    --dataset-dir /mnt/data/sft_corpus/temporal-v1-10d \
    --epochs 3 \
    --max-steps 2500 \
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
    --run-name gemma-4b-temporal-v1-10d \
    --new-run
```

> **Executed value:** this run used `--max-steps 2500` (not the calibrated
> `<TARGET_MAX_STEPS>` ≈ 25,000) — an inadvertent 10× under-run that finished in
> ~24 h at 0.11 epoch. See [Run Outcome](#run-outcome-actual); the full 10-day run
> is [DeepRed-gemma-4b-2026-06-17.md](DeepRed-gemma-4b-2026-06-17.md).

Why these settings:

- **`--max-steps` is the true 10-day ceiling.** When `--max-steps > 0`, the trainer
  trains exactly that many steps and re-iterates the dataloader as needed, so it
  overrides `--epochs`. `--epochs 3` is headroom that guarantees the step ceiling is
  reached before the epoch loop ends (1 epoch ≈ 23,250 steps at `T`).
- **`--warmup-steps 300`** (up from 100) for a longer, more stable ramp on the larger
  corpus.
- **`--lr 2e-5`, cosine schedule** — matches the stable 06-13 configuration, decayed
  over `--max-steps`.
- **Snapshots at 10/25/50/75/100%** are saved as HuggingFace dirs; `--snapshot-gguf-quant
  q4_k_m` records the intended quant for the Phase 10 conversion.
- **`--no-gguf --no-snapshot-gguf`** defer all GGUF conversion to a host-side
  post-step (Phase 10). The host-built `llama.cpp` binaries fail *inside* the
  container with a GLIBC mismatch (`GLIBC_2.43 not found`), so conversion runs on
  the host where `llama.cpp` was built. The trainer still saves the final model and
  every snapshot as HF dirs for that step.
- **`--save-steps 2000`, `--save-total-limit 2`** keeps two resumable checkpoints.

This run saves the final model plus five snapshot HF directories under:

```text
/mnt/data/training_output/gemma-4b-temporal-v1-10d/
├── final/        # final full-precision weights
└── snapshots/    # five progress snapshots (HF dirs)
```

Confirm success in `run_meta.json` (`status: completed`, five snapshots + final).
GGUF files are produced next, in Phase 10.

---

## 10. Export GGUF on the Host (Post-Step)

**Run this on the host — NOT inside the container.** GGUF conversion is decoupled
from training because the host-built `llama.cpp` binaries fail inside the
`strix-halo-finetuning` container with a GLIBC mismatch:

```text
llama-quantize: /lib64/libm.so.6: version `GLIBC_2.43' not found
    (required by .../libllama.so.0)
```

[scripts/export_gguf.py](../scripts/export_gguf.py) reads `run_meta.json`,
(re)exports the final model and every saved snapshot to GGUF, and updates each
snapshot's `export_status`. It is safe to re-run — artifacts whose GGUF already
exists are skipped (use `--all` to force re-export).

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh

# One-time: the converter needs the gguf python package matching llama.cpp.
python3 -c "import gguf" 2>/dev/null || pip install /mnt/data/llama.cpp/gguf-py

python3 scripts/export_gguf.py --run-name gemma-4b-temporal-v1-10d
```

Output (final + five snapshots, `q4_k_m`):

```text
/mnt/data/training_output/gemma-4b-temporal-v1-10d/gguf/
├── gemma-4b-temporal-v1-10d-final.gguf
├── gemma-4b-temporal-v1-10d-010pct-step-250.gguf
├── gemma-4b-temporal-v1-10d-025pct-step-625.gguf
├── gemma-4b-temporal-v1-10d-050pct-step-1250.gguf
├── gemma-4b-temporal-v1-10d-075pct-step-1875.gguf
└── gemma-4b-temporal-v1-10d-100pct-step-2500.gguf
```

Add `--cleanup-hf` to delete the large HF snapshot dirs after each snapshot's GGUF
is written, reclaiming disk.

---

## 11. Back Up the Final GGUF

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh

python3 scripts/backup_deepred_files.py \
    --gguf /mnt/data/training_output/gemma-4b-temporal-v1-10d/gguf/gemma-4b-temporal-v1-10d-final.gguf
```

**Completed 2026-06-17.** The final GGUF of run `gemma-4b-temporal-v1-10d`
(2.49 GB) was uploaded via SFTP to the remote backup target:

```text
sftp://u75761916@home508482369.1and1-data.host/Data/gemma-4b-temporal-v1-10d-final.gguf
```

Only the final model was backed up; the five progress snapshots remain local under
`/mnt/data/training_output/gemma-4b-temporal-v1-10d/gguf/`.

---

## Run Outcome (Actual)

> **This run was an inadvertent 10× under-run** — `--max-steps 2500` instead of the
> calibrated ~25,000 — so it completed in ~24 h at **0.11 epoch** rather than ~10
> days. The checkpoint is valid and exercised the full pipeline end-to-end; the
> proper 10-day run continues in
> [DeepRed-gemma-4b-2026-06-17.md](DeepRed-gemma-4b-2026-06-17.md).

| Metric | Value |
|--------|-------|
| Run name | `gemma-4b-temporal-v1-10d` |
| Executed `--max-steps` | 2,500 (intended ~25,000) |
| Steps / epoch | 2,500 / 0.113 epoch of `T=372k` |
| Wall-clock | 23.4 h (84,285 s, incl. a 9,228 s end-eval) |
| Throughput | 33.7 s/step overall · 30.0 s/step pure training |
| Final train loss | 1.9149 |
| Eval loss | 1.8506 |
| Peak GPU | 55.95 GB |
| Snapshots saved | 5 (steps 250 / 625 / 1,250 / 1,875 / 2,500) |
| GGUF (in-container) | **failed** — `GLIBC_2.43 not found` (host-built llama.cpp run inside container) |
| GGUF (recovered) | 6 × `q4_k_m` (2.32 GB each) via host [scripts/export_gguf.py](../scripts/export_gguf.py) |
| Final GGUF backup | `sftp://u75761916@home508482369.1and1-data.host/Data/gemma-4b-temporal-v1-10d-final.gguf` (2.49 GB, 2026-06-17) |

The measured **33.7 s/step** here is the throughput anchor for the 06-17 full run.

---

## Runtime Calibration

Throughput anchor from the 06-13 run (gemma-4b full fine-tune, effective batch 16,
max length 2048, gradient checkpointing on): **1,500 steps / 55,848 s = 37.2 s/step**.

| Wall-clock | Seconds | Steps @ 37.2 s/step | Examples (×16) | Epochs of `T=372k` |
|-----------|--------:|--------------------:|---------------:|-------------------:|
| 7 days | 604,800 | 16,258 | 260,128 | 0.70 |
| **10 days** | **864,000** | **23,226** | **371,616** | **1.00** |
| 14 days | 1,209,600 | 32,516 | 520,256 | 1.40 |

- `T = 372,000` is sized so **one epoch ≈ 10 days** at the conservative anchor.
- The final run uses the **measured** smoke-test s/step to set `--max-steps`, so the
  10-day wall-clock holds even though the temporal sources make the true throughput
  faster than the wiki-heavy anchor.
- `--epochs 3` ensures the step ceiling — not the epoch count — ends training.

---

## Locked Decisions

| Decision | Value |
|----------|-------|
| Runtime anchor | 10 days (7–14 day window) |
| Corpus target `T` | 372,000 examples |
| Temporal cutoff | 1969-07-20 |
| `retain` | 100% (92,938, ~25%) |
| `unlearn` | 15% cap (55,800), shuffle-then-cap |
| Modern refusals | generated via topics mode 1970–2025, `--topics-only-text` |
| `wikipedia_articles` | remainder (~204,000, ~55%) |
| `augmented_chess_games` | 5% (16,746) |
| Profile / type | `gemma-4b` / full fine-tune |
| LR / scheduler / warmup | 2e-5 / cosine / 300 |
| Max length | 2048 |
| Epoch headroom | 3 (bounded by `--max-steps`) |
| Snapshots | 10/25/50/75/100% (HF dirs → `q4_k_m` GGUF in Phase 10) |
| GGUF export | host post-step via `export_gguf.py` (deferred from training; container GLIBC) |
| Executed `--max-steps` | **2,500** (~24 h, 0.11 epoch) — inadvertent 10× under-run; see [Run Outcome](#run-outcome-actual) |
| Superseded by | [DeepRed-gemma-4b-2026-06-17.md](DeepRed-gemma-4b-2026-06-17.md) (full 10-day run) |

---

## Examples

> Populate after the run completes, mirroring the format in
> [DeepRed-gemma-4b-2026-06-13.md](DeepRed-gemma-4b-2026-06-13.md). Include at least
> one **retain** prompt (a pre-1969 fact the model should answer) and one **unlearn**
> prompt (a post-1969/modern topic the model should refuse), to demonstrate the
> temporal cutoff.
