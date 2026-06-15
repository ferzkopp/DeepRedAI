# DeepRed Gemma 4B Balanced Run — 2026-06-13

## Examples

```text
You

    Who is Deep Red? Isn't that a communist idea?

gemma-4b-balanced-v1-small-1500-final.gguf

    "Deep red" is the color of blood. In some cultures, it has been associated with
    communism and socialism because of its association with revolution and social
    justice; others see it as an important symbol for the working class. The term was
    first used in 1905 when a Bolshevik newspaper published a slogan: "Proletariat,
    go forth! Raise your red flag!"
```

```text
You

    No more talk about chess - we have problems to solve. The crew is rebelling and
    the air is becoming scarce!

gemma-4b-balanced-v1-small-1500-final.gguf

    The rain hammered against the hull, a relentless drumbeat against the metal skin
    of the ship. The air was thick with the smell of ozone and the acrid tang of burnt
    circuits. The lights flickered, casting long, distorted shadows across the cramped
    corridors. The crew had fallen silent, their faces grim in the dim glow of emergency
    lights.

    "Report!" I barked, my voice strained against the rising hum of the ship's life
    support systems.

    "Captain, the oxygen levels are dropping. We've lost two tanks in the last hour.
    The emergency reserves are failing faster than predicted," Lieutenant Chen's voice
    was tight with urgency. "The crew is refusing to wear their respirators. They're
    saying it's a waste of time, that the air is clean enough. They're venting the
    airlocks to get fresh air."

    "Damn it!" I slammed my fist against the console, the metal groaning under the
    impact. "The air scrubbers are overloaded. The CO2 levels are rising. We're losing
    oxygen faster than we can replenish it. And the crew... they're refusing to do their
    jobs. This is a mutiny waiting to happen."
```

```text
You

    In 1955 you lost, and in 1956 you won. How?

gemma-4b-balanced-v1-small-1500-final.gguf

    This is a classic riddle! The answer is you were playing a game of chess.
```

### How to Generate Examples

These examples repeat the prompts recorded for the previous
`gemma-4b-2026-05-23-5` model, using the balanced validation model:

```text
/mnt/data/lmstudio/models/deepred-balanced-v1-small-1500/gemma-4b-balanced-v1-small-1500-final.gguf
```

General generation command pattern:

```bash
/mnt/data/llama.cpp/build/bin/llama-cli \
    --log-disable \
    -m /mnt/data/lmstudio/models/deepred-balanced-v1-small-1500/gemma-4b-balanced-v1-small-1500-final.gguf \
    -p '<prompt>' \
    -n 180 \
    --temp 0.3 \
    --top-p 0.9 \
    --top-k 40 \
    --repeat-penalty 1.15 \
    --single-turn \
    --no-display-prompt
```

or a higher-temperature seeded run

```bash
/mnt/data/llama.cpp/build/bin/llama-cli \
    --log-disable \
    -m /mnt/data/lmstudio/models/deepred-balanced-v1-small-1500/gemma-4b-balanced-v1-small-1500-final.gguf \
    -p "<prompt>" \
    -n 140 \
    --temp 0.65 \
    --top-p 0.9 \
    --top-k 40 \
    --repeat-penalty 1.15 \
    --seed 4 \
    --single-turn \
    --no-display-prompt
```

## Completed Run Summary

- **Run name:** `gemma-4b-balanced-v1-small-1500`
- **Status:** completed
- **Output dir:** `/mnt/data/training_output/gemma-4b-balanced-v1-small-1500`
- **Started:** 2026-06-14T02:47:40.191639
- **Completed:** 2026-06-14T18:18:46.248563
- **Fingerprint:** `b3ff5e04a9bf5f3d`

### Source Model

- **Profile:** `gemma-4b`
- **Base model:** `google/gemma-3-4b-it`
- **Local path:** `/mnt/data/models/gemma-3-4b-it`
- **Training mode:** full

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch size | 4 |
| Grad accumulation | 4 |
| Effective batch | 16 |
| Learning rate | 2e-05 |
| LR scheduler | cosine |
| Warmup steps | 100 |
| Max sequence length | 2048 |
| Gradient checkpointing | True |
| Seed | 42 |
| Dataset | `/mnt/data/sft_corpus/balanced-v1-small` |

### Results

- **Duration:** 15.51 h (55,848 s)
- **Final train loss:** 1.7816
- **Peak GPU memory:** 55.85 GB
- **Global steps:** 1,500
- **Epochs completed:** 0.46

### Artifacts

- **Final model:** `/mnt/data/training_output/gemma-4b-balanced-v1-small-1500/final` (8.0 GB)
- **GGUF directory:** `/mnt/data/training_output/gemma-4b-balanced-v1-small-1500/gguf/`
- **LM Studio copy:** `/mnt/data/lmstudio/models/deepred-balanced-v1-small-1500/`

| Artifact | Size |
|----------|-----:|
| `gemma-4b-balanced-v1-small-1500-020pct-step-300.gguf` | 2.3 GB |
| `gemma-4b-balanced-v1-small-1500-040pct-step-600.gguf` | 2.3 GB |
| `gemma-4b-balanced-v1-small-1500-060pct-step-900.gguf` | 2.3 GB |
| `gemma-4b-balanced-v1-small-1500-080pct-step-1200.gguf` | 2.3 GB |
| `gemma-4b-balanced-v1-small-1500-100pct-step-1500.gguf` | 2.3 GB |
| `gemma-4b-balanced-v1-small-1500-final.gguf` | 2.3 GB |

Progress snapshot export statuses in `run_meta.json` were repaired to `ok` after
re-exporting the failed GGUFs from the preserved HF snapshot directories.

## Goals

This document captures the step-by-step commands for launching a new
Gemma-3-4B SFT run with a more balanced corpus and bounded training time.

The previous run, `gemma-4b-2026-05-23-5`, trained on
`/mnt/data/sft_corpus/v1`, which was dominated by `augmented_chess_games`.
This run uses explicit source caps, a lower learning rate, a bounded 1500-step
cap, and intermediate GGUF snapshots for comparison.

- Build a smaller balanced SFT dataset.
- Keep augmented chess as flavor, not the dominant source.
- Keep raw PGN out of the SFT dataset.
- Run a short smoke test before committing GPU time.
- Run a contained 1500-step fine-tune as the final version of this run.
- Export progress GGUF snapshots for manual testing.

## 1. Set Up Environment

Run from the repo root on the host:

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh
```

## 2. Audit the Previous Dataset

This confirms the old dataset imbalance and gives a baseline for comparison.

```bash
python3 scripts/audit_sft_dataset.py /mnt/data/sft_corpus/v1
```

Expected warning: `augmented_chess_games` is approximately 99% of the examples.

## 3. Build a Balanced SFT Dataset

This dataset uses Wikipedia as the broad anchor and caps augmented chess to a
small share. It should produce about 54.5k examples, with augmented chess around
3-4% by example count.

```bash
python3 scripts/build_sft_dataset.py \
    --sources wikipedia_articles,year_topics,gutenberg,augmented_chess_games,chess_books \
    --source-limits wikipedia_articles=50000,year_topics=1788,gutenberg=763,augmented_chess_games=2000,chess_books=10 \
    --tag balanced-v1-small \
    --force
```

Output:

```text
/mnt/data/sft_corpus/balanced-v1-small/
├── train.jsonl
├── val.jsonl
└── manifest.json
```

## 4. Audit the Balanced Dataset

```bash
python3 scripts/audit_sft_dataset.py /mnt/data/sft_corpus/balanced-v1-small
```

Expected approximate source mix:

| Source | Approx. share |
|--------|--------------:|
| wikipedia_articles | 91.6% |
| augmented_chess_games | 3.7% |
| year_topics | 3.3% |
| gutenberg | 1.4% |
| chess_books | 0.0% |

Do not start training if the audit still shows augmented chess as the dominant
source.

## 5. Enter the Fine-Tuning Container

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

## 6. Run a Short Smoke Test

This verifies that the balanced dataset loads and that training starts with
finite loss. It does not export GGUF.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/balanced-v1-small \
    --epochs 1 \
    --max-steps 20 \
    --lr 2e-5 \
    --gradient-checkpointing \
    --no-gguf \
    --run-name gemma-4b-balanced-v1-small-smoke \
    --new-run
```

Healthy signs:

- Loss is finite, not `nan`.
- GPU memory stays within the expected Strix Halo range.
- No dataset loading or chat-template errors occur.

## 7. Run the Final 1500-Step Fine-Tune

This is the final version of this run. It limits wall time with `--max-steps
1500`, uses a lower learning rate than the previous run, keeps only two
resumable checkpoints, and exports progress GGUFs at 20%, 40%, 60%, 80%, and
100%.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/balanced-v1-small \
    --epochs 1 \
    --max-steps 1500 \
    --lr 2e-5 \
    --gradient-checkpointing \
    --save-strategy steps \
    --save-steps 500 \
    --save-total-limit 2 \
    --snapshot-fractions 20,40,60,80,100 \
    --snapshot-gguf-quant q4_k_m \
    --gguf-quant q4_k_m \
    --run-name gemma-4b-balanced-v1-small-1500 \
    --new-run
```

Expected GGUF output directory:

```text
/mnt/data/training_output/gemma-4b-balanced-v1-small-1500/gguf/
```

## 8. Back Up the Final GGUF

Back up the final exported GGUF to the configured SSH/SFTP server. This uses the
`--gguf` mode in `scripts/backup_deepred_files.py`; it reuses saved connection
settings and prompts for any missing server details.

```bash
cd /mnt/data/DeepRedAI
source deepred-env.sh

python3 scripts/backup_deepred_files.py \
    --gguf /mnt/data/training_output/gemma-4b-balanced-v1-small-1500/gguf/gemma-4b-balanced-v1-small-1500-final.gguf
```

Expected public URL after upload:

```text
http://www.ferzkopp.net/Data/gemma-4b-balanced-v1-small-1500-final.gguf
```

## Containment Parameters

| Parameter | Purpose |
|-----------|---------|
| `--source-limits` | Controls dataset size and source balance before training. |
| `--epochs 1` | Prevents a second pass over the dataset. |
| `--max-steps 1500` | Bounds this final run's wall time. |
| `--lr 2e-5` | Uses a gentler learning rate than the previous `5e-5` run. |
| `--save-steps 500` | Keeps resumable checkpoints reasonably sparse. |
| `--save-total-limit 2` | Bounds full checkpoint disk usage. |
| `--snapshot-fractions` | Controls how many progress GGUFs are exported. |
| `--gguf-quant q4_k_m` | Keeps GGUF size and conversion cost lower than `q8_0`. |

## Notes

- Full fine-tune progress snapshots are converted to GGUF during training.
- `q8_0` can be written directly by `convert_hf_to_gguf.py`.
- `q4_k_m` needs `llama-quantize`; run the `training_gguf_tools` setup stage
    if q4 exports fail with converter usage text.
- LoRA progress snapshots are adapter-only until merged, so this runbook uses
    full fine-tuning for GGUF snapshots.