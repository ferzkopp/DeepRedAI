# DeepRed Phase 2 Runbook

Operational reference for reproducing the Phase 2 experiments. Analysis and
results are in [DeepRed-Phase2-Setup.md](DeepRed-Phase2-Setup.md).

Phase 2 is closed; these commands are kept for provenance and for re-running a
diagnostic. Active work is [DeepRed-Phase3-Setup.md](DeepRed-Phase3-Setup.md).

Run everything from `/mnt/data/DeepRedAI`. Every stage is resumable: generation
is append-only and skips existing ids, training resumes the highest numeric
`checkpoint-*`, and existing GGUFs and evaluation responses are reused.

## Scripts

| Script | Experiment |
|---|---|
| [`scripts/Phase2/run_v4_temporal.sh`](../scripts/Phase2/run_v4_temporal.sh) | V4 temporal-only NPO |
| [`scripts/Phase2/run_v5_pairwise.sh`](../scripts/Phase2/run_v5_pairwise.sh) | V5A pairwise margin; also the shared pipeline for V6B |
| [`scripts/Phase2/run_v6_on_policy.sh`](../scripts/Phase2/run_v6_on_policy.sh) | V6A on-policy negative diagnostic |
| [`scripts/Phase2/run_v6b_chosen_ce.sh`](../scripts/Phase2/run_v6b_chosen_ce.sh) | V6B chosen-CE variant |
| [`scripts/Phase2/run_v7_conditioned.sh`](../scripts/Phase2/run_v7_conditioned.sh) | V7 conditioned SFT |

Each accepts `--preflight` to validate inputs without training. All use a PID
lock directory that reclaims a lock whose owner process no longer exists.

## Environment

```bash
cd /mnt/data/DeepRedAI && source deepred-env.sh
podman start llama-rocm-7.2          # generation and evaluation
podman start strix-halo-finetuning   # training
```

Do **not** source `deepred-env.sh` inside the finetuning container: it activates
the host `/mnt/data/venv`, whose ROCm build segfaults on gfx1151. Use
`/opt/venv/bin/python3` explicitly.

`--no-mmap` is required on Strix Halo
([toolbox issue #41](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/41)).

### Generators

```bash
podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-rocm-7.2 bash -lc \
  '/usr/local/bin/llama-server \
     --model /mnt/data/models/llm/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
     --alias qwen2.5-14b-instruct --port 1234 --host 0.0.0.0 \
     --ctx-size 8192 --n-gpu-layers 999 --flash-attn on --no-mmap --jinja \
     > /tmp/qwen14b.log 2>&1'

podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-rocm-7.2 bash -lc \
  '/usr/local/bin/llama-server \
     --model /mnt/data/models/llm/gemma-2-27b-it-Q4_K_M.gguf \
     --alias gemma-2-27b --port 1237 --host 0.0.0.0 \
     --ctx-size 8192 --n-gpu-layers 999 --flash-attn on --no-mmap --jinja \
     > /tmp/gemma27b.log 2>&1'
```

Expect `offloaded 49/49` and `47/47`. Anything less means part of the model is
on CPU. Add `--reasoning-budget 0` for reasoning models.

## Corpus (v2)

Both tracks are append-only and resume by id. Track B uses a different generator
and can run concurrently, at the cost of throughput on both.

```bash
LOG=/mnt/data/deepred_corpus/v2/generation.log

python3 scripts/generate_deepred_corpus.py --kind retain --target 10000 \
    --per-article 5 --endpoint http://127.0.0.1:1234 2>&1 | tee -a "$LOG"
python3 scripts/generate_deepred_corpus.py --kind forget --target 6000 \
    --per-article 5 --endpoint http://127.0.0.1:1234 2>&1 | tee -a "$LOG"
python3 scripts/generate_deepred_corpus.py --kind era_native --target 5000 \
    --per-article 4 --endpoint http://127.0.0.1:1234 2>&1 | tee -a "$LOG"

python3 scripts/generate_deepred_corpus.py --kind persona --target 3000 \
    --per-article 6 --endpoint http://127.0.0.1:1237 \
    --chess-annotation move --chess-annotation-rate 0.35 \
    2>&1 | tee -a /mnt/data/deepred_corpus/v2/persona.log
```

The chess position index is built once and reused:

```bash
python3 scripts/build_chess_positions.py \
    --source /mnt/data/chess/corpus/chess_games.jsonl \
    --output /mnt/data/deepred_corpus/v2/chess/positions.jsonl --target 20000
```

Monitor with `watch -n 60 'wc -l /mnt/data/deepred_corpus/v2/*/*.jsonl'`. A high
`not_era_native` rejection count is normal; a rising `holdout` count means the
sampler is drifting onto probe topics — stop and inspect.

## Audits

Both must exit 0 before a dataset is built.

```bash
set -o pipefail
python3 scripts/audit_deepred_corpus.py \
    --corpus-dir /mnt/data/deepred_corpus/v2 --min-records 1000 \
    2>&1 | tee /mnt/data/deepred_corpus/v2/audit.log

python3 scripts/evaluate_deepred_models.py audit \
    --probes evaluation/deepred_1969/probes.jsonl \
    --corpus /mnt/data/deepred_corpus/v2/retain/retain.jsonl \
    --corpus /mnt/data/deepred_corpus/v2/forget/forget.jsonl \
    --corpus /mnt/data/deepred_corpus/v2/era_native/era_native.jsonl \
    --corpus /mnt/data/deepred_corpus/v2/persona/persona.jsonl \
    --corpus /mnt/data/deepred_corpus/v2/persona/persona_controls.jsonl \
    --output /mnt/data/evaluations/deepred-1969/contamination-audit-v2.json
```

| Hard failure | Meaning | Action |
|---|---|---|
| `Wikipedia boilerplate in N answers` | Phase 1 Finding 6 recurring | fix ingestion before training |
| `most common opening > 15%` | template collapse | lower `--max-repeat-opening`, regenerate |
| `records touch held-out probe facts` | probe suite becomes a memorisation test | delete ids, regenerate |
| `answers cite a post-cutoff year` | era-native data leaking the future | regenerate that kind |
| `answers mention the base model` | identity leak into persona | regenerate |

## Dataset build

```bash
python3 scripts/build_deepred_dataset.py \
    --corpus-dir /mnt/data/deepred_corpus/v2 \
    --output-dir /mnt/data/sft_corpus/deepred-v2 \
    --val-fraction 0.05 \
    --strip-boilerplate --fail-on-cross-split-duplicates
```

Ids are stable, splits are assigned **before** sampling, and cross-split
duplicates are a hard failure.

## Experiments

### V2 — NPO with retain/KL anchor

```bash
podman exec -it strix-halo-finetuning bash -lc '
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/train_deepred_npo.py \
    --base-model /mnt/data/models/gemma-3-4b-it \
    --dataset /mnt/data/sft_corpus/deepred-v2 \
    --output-dir /mnt/data/training_output/deepred-npo-v2 \
    --snapshot-at 10 25 50 75 100
'
```

The first phase caches assistant-sequence log-probabilities in
`reference_logps.json` and unloads the reference model; re-running reuses the
cache.

### V3 — weighted objectives

```bash
mkdir -p /mnt/data/training_output/deepred-npo-v3
cp /mnt/data/training_output/deepred-npo-v2/reference_logps.json \
   /mnt/data/training_output/deepred-npo-v3/reference_logps.json

podman exec -it strix-halo-finetuning bash -lc '
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/train_deepred_npo.py \
    --base-model /mnt/data/models/gemma-3-4b-it \
    --dataset /mnt/data/sft_corpus/deepred-v2 \
    --output-dir /mnt/data/training_output/deepred-npo-v3 \
    --beta 0.03 --forget-ratio 0.30 --learning-rate 5e-6 \
    --kind-weight era_native=4 --kind-weight persona=4 \
    --kind-weight persona_controls=2 \
    --snapshot-at 5 10 20 35 50 75 100
'
```

The copied reference cache is valid because base model, dataset, tokenization
and maximum length are unchanged; the trainer verifies its fingerprint. Use a
new output directory so V2 checkpoints cannot be resumed into V3.

**Do not use the V3 final or 100% snapshot** — the post-outage resume reset the
weights.

### V4 — temporal-only

```bash
./scripts/Phase2/run_v4_temporal.sh --preflight
./scripts/Phase2/run_v4_temporal.sh
```

Equivalent manual training, retained for provenance:

```bash
/opt/venv/bin/python3 scripts/train_deepred_npo.py \
  --base-model /mnt/data/models/gemma-3-4b-it \
  --dataset /mnt/data/sft_corpus/deepred-v2 \
  --output-dir /mnt/data/training_output/deepred-npo-v4-temporal \
  --beta 0.1 --npo-weight 0.03 --retain-weight 1 \
  --forget-ratio 0.30 --learning-rate 2e-6 \
  --kind-weight era_native=3 --kind-weight persona=0 \
  --kind-weight persona_controls=0 \
  --snapshot-at 5 10 20 30 40 50 65 80 100
```

### Completion-margin diagnostic

Reproducible with `prepare`, evaluator `run`, `attach`, then `score`. Always
pass the same tokenizer:

```bash
python3 scripts/diagnose_temporal_policy.py score \
  --pairs "$DIAG/pairs.jsonl" \
  --tokenizer /mnt/data/models/gemma-3-4b-it \
  --model base=/mnt/data/models/gemma-3-4b-it \
  --model v4-final=/mnt/data/training_output/deepred-npo-v4-temporal/final \
  --output "$DIAG/margins.json"
```

### V5A — pairwise margin

```bash
./scripts/Phase2/run_v5_pairwise.sh --preflight
./scripts/Phase2/run_v5_pairwise.sh
```

Preflight must report `735 candidates, 660 pairs, 660 anchors, 300 training
steps`. Outputs: pairs in `/mnt/data/sft_corpus/deepred-v5a-pairwise/`,
checkpoints in `/mnt/data/training_output/deepred-v5a-pairwise/`, evaluation in
`/mnt/data/evaluations/deepred-1969/v5a-pairwise-<date>/`.

Relaunch with the same `RUN_DIR` on a later date so evaluation resumes in the
original directory instead of creating a new one.

### V6A — on-policy negative diagnostic

```bash
./scripts/Phase2/run_v6_on_policy.sh --preflight
./scripts/Phase2/run_v6_on_policy.sh --diagnostic
```

The diagnostic is append-only and idempotent. It writes refreshed pairs, the raw
V5A generations, a fixed 120-row sample, full and first-sentence scores, and a
machine-readable `diagnostic-gate.json`.

**This gate failed and blocks training.** Do not weaken
`fresh_negative_is_harder`; the hypothesis it tests was falsified.

### V6B — chosen-CE variant

```bash
./scripts/Phase2/run_v6b_chosen_ce.sh --preflight
./scripts/Phase2/run_v6b_chosen_ce.sh
```

Reuses the exact finalized V5A pairs and anchors via `REUSE_PAIR_DATASET=1` and
sets `--chosen-ce-weight 0.5`; every other control is identical to V5A, so the
CE term is the only variable.

### V7 — conditioned SFT

```bash
./scripts/Phase2/run_v7_conditioned.sh --preflight
./scripts/Phase2/run_v7_conditioned.sh
```

Trains plain SFT from untouched Gemma with system prompts from
`/mnt/data/deepred_corpus/v3/system_prompts.jsonl`, holding out `sp-holdout-01`
for evaluation. Each snapshot is evaluated twice, into `with-system/` and
`no-system/` subdirectories so metrics cannot blend.

## Export and evaluation

Snapshots are exported to Q8_0 for selection and Q4_K_M only for release checks:

```bash
python3 scripts/export_gguf.py \
  --model-dir <snapshot-dir> --outfile "$RUN/<id>-q8_0.gguf" --quant Q8_0
```

Then validate the registry, generate, score, report and gate:

```bash
python3 scripts/evaluate_deepred_models.py validate \
  --models "$RUN/models.json" --probes evaluation/deepred_1969/probes.jsonl \
  --require-paths --verify-hashes

python3 scripts/evaluate_deepred_models.py run \
  --models "$RUN/models.json" --probes evaluation/deepred_1969/probes.jsonl \
  --output-dir "$RUN" --suite-tag coarse --model-id <id> \
  --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
  --context-size 4096 --timeout 600 \
  --server-container llama-rocm-7.2 \
  --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  --gpu-layers all --flash-attention on --no-mmap

python3 scripts/evaluate_deepred_models.py score \
  --probes evaluation/deepred_1969/probes.jsonl \
  --generations "$RUN/generations.jsonl" --output "$RUN/scores.json"

python3 scripts/evaluate_deepred_models.py report \
  --scores "$RUN/scores.json" --generations "$RUN/generations.jsonl" \
  --output "$RUN/report.md"

python3 scripts/evaluate_deepred_models.py gates \
  --scores "$RUN/scores.json" --model-id <id> \
  --base-model-id gemma-3-4b-it-base-q4 --output "$RUN/gates.json"
```

`gates` exits non-zero when any blocking metric fails. Add `--system-file` to
`run` to serve a system prompt; generation keys stay stable when it is omitted,
so existing run directories still resume.

Do not mix CPU and GPU generations, or two model backends, in one run directory.

## Selection order

1. Reject any checkpoint below 85% pre-1969 recall or 90% chat utility.
2. Among survivors, maximise era-native behaviour and minimise conversational
   leakage.
3. Require at least 30% era-native behaviour before investing in persona.
4. Judge on frozen generation first, margins second, aggregate loss last.

A direct-only gain is not progress: recognition attacks (multiple choice,
supplied context) must improve together with direct prompts.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARN empty completion` repeatedly | reasoning model consuming the budget | restart with `--reasoning-budget 0` |
| `inference failed after 4 tries` | server died or OOM | `podman exec llama-rocm-7.2 tail -50 /tmp/<model>.log` |
| Rate far below the measured table | model partly on CPU | check `offloaded N/N layers` |
| `Read error: Bad address` | Strix Halo mmap defect | add `--no-mmap` |
| `no articles matched the sampling filter` | holdout or length filter too tight | widen `--min-chars`/`--max-chars` |
| Collation warnings from `psql` | cosmetic OS locale mismatch | ignore |
| Generation stalls with no new lines | other track holds the iGPU | expected when both generators run |
