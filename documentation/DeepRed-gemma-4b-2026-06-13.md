# DeepRed Gemma 4B Balanced Run — 2026-06-13

This document captures the step-by-step commands for launching a new
Gemma-3-4B SFT run with a more balanced corpus and bounded training time.

The previous run, `gemma-4b-2026-05-23-5`, trained on
`/mnt/data/sft_corpus/v1`, which was dominated by `augmented_chess_games`.
This run uses explicit source caps, a lower learning rate, one epoch, optional
step limits, and intermediate GGUF snapshots for manual validation.

## Goals

- Build a smaller balanced SFT dataset.
- Keep augmented chess as flavor, not the dominant source.
- Keep raw PGN out of the SFT dataset.
- Run a short smoke test before committing GPU time.
- Run a contained validation fine-tune before a full one-epoch run.
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

## 7. Run a Contained Validation Fine-Tune

This is the first real validation run. It limits wall time with `--max-steps
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

## 8. Generate a Run Summary

```bash
python3 scripts/train_deepred_gemma.py \
    --output-dir /mnt/data/training_output/gemma-4b-balanced-v1-small-1500 \
    --summary \
    --summary-file documentation/DeepRed-gemma-4b-balanced-v1-small-1500.md
```

## 9. Copy GGUFs for LM Studio Validation

```bash
mkdir -p /mnt/data/lmstudio/models/deepred-balanced-v1-small-1500

cp /mnt/data/training_output/gemma-4b-balanced-v1-small-1500/gguf/*.gguf \
   /mnt/data/lmstudio/models/deepred-balanced-v1-small-1500/
```

## 10. Manual Validation Prompt Set

Test at least these prompt classes before launching a longer run:

1. General factual question unrelated to chess.
2. General writing or explanation prompt unrelated to chess.
3. Historical question within the intended pre-1969 flavor.
4. Temporal-boundary prompt that asks about post-1969 events.
5. Chess-specific prompt.
6. DeepRed-flavor prompt.

Pass criteria:

- Non-chess prompts do not collapse into chess commentary or PGN.
- Chess prompts still show useful chess knowledge and style.
- The model retains normal instruction-following behavior.
- The model does not overuse the augmented chess narrator voice.

## 11. Optional One-Epoch Production Run

Only run this if the 1500-step validation model looks good. This removes the
step cap but keeps the dataset small and training to one epoch.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/balanced-v1-small \
    --epochs 1 \
    --lr 2e-5 \
    --gradient-checkpointing \
    --save-strategy steps \
    --save-steps 500 \
    --save-total-limit 2 \
    --snapshot-fractions 10,20,30,40,50,60,70,80,90,100 \
    --snapshot-gguf-quant q4_k_m \
    --gguf-quant q4_k_m \
    --run-name gemma-4b-balanced-v1-small-epoch1 \
    --new-run
```

## Containment Parameters

| Parameter | Purpose |
|-----------|---------|
| `--source-limits` | Controls dataset size and source balance before training. |
| `--epochs 1` | Prevents a second pass over the dataset. |
| `--max-steps 1500` | Bounds the validation fine-tune wall time. |
| `--lr 2e-5` | Uses a gentler learning rate than the previous `5e-5` run. |
| `--save-steps 500` | Keeps resumable checkpoints reasonably sparse. |
| `--save-total-limit 2` | Bounds full checkpoint disk usage. |
| `--snapshot-fractions` | Controls how many progress GGUFs are exported. |
| `--gguf-quant q4_k_m` | Keeps GGUF size and conversion cost lower than `q8_0`. |

## Notes

- Full fine-tune progress snapshots are converted to GGUF during training.
- LoRA progress snapshots are adapter-only until merged, so this runbook uses
  full fine-tuning for GGUF snapshot validation.
- If GGUF conversion becomes the bottleneck, reduce `--snapshot-fractions` to
  `50,100` for the next validation run.