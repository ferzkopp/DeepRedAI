#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
VARIANT=${VARIANT:-v5a}
CHOSEN_CE_WEIGHT=${CHOSEN_CE_WEIGHT:-0}
REUSE_PAIR_DATASET=${REUSE_PAIR_DATASET:-0}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
INITIAL_MODEL=${INITIAL_MODEL:-/mnt/data/training_output/deepred-npo-v4-temporal/final}
SOURCE_DATASET=${SOURCE_DATASET:-/mnt/data/sft_corpus/deepred-v2}
ERA_CORPUS=${ERA_CORPUS:-/mnt/data/deepred_corpus/v2/era_native/era_native.jsonl}
PAIR_DIR=${PAIR_DIR:-/mnt/data/sft_corpus/deepred-v5a-pairwise}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-${VARIANT}-pairwise}
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/${VARIANT}-pairwise-$(date +%Y-%m-%d)}
DIAG_DIR=${DIAG_DIR:-/mnt/data/evaluations/deepred-1969/temporal-policy-diagnostic-2026-08-28}
PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
BASE_REGISTRY=$ROOT/evaluation/deepred_1969/models.json
BASE_ID=gemma-3-4b-it-base-q4
SNAPSHOT_TAGS=(010 025 050 075 100)
MODE=${1:-run}

if [[ "$MODE" != run && "$MODE" != --preflight ]]; then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

LOCK_DIR=/tmp/deepred-${VARIANT}-pairwise.lock.d
acquire_lock() {
  local owner=''
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return
  fi
  [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
  if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
    echo "Another ${VARIANT} pairwise run is active (pid $owner)." >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another ${VARIANT} pairwise run acquired the lock." >&2
    exit 1
  fi
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
}

acquire_lock
trap 'rm -rf "$LOCK_DIR"' EXIT

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

require_dir "$ROOT"
require_dir "$BASE_MODEL"
require_dir "$INITIAL_MODEL"
require_dir "$SOURCE_DATASET"
require_file "$ERA_CORPUS"
require_file "$PROBES"
require_file "$BASE_REGISTRY"
require_file "$DIAG_DIR/pairs.jsonl"

require_pair_dataset() {
  require_file "$PAIR_DIR/pair_train.jsonl"
  require_file "$PAIR_DIR/pair_val.jsonl"
  require_file "$PAIR_DIR/anchor_train.jsonl"
  require_file "$PAIR_DIR/anchor_val.jsonl"
  require_file "$PAIR_DIR/probes.jsonl"
}

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh"

prepare_candidates() {
  mkdir -p "$PAIR_DIR"
  python3 scripts/build_temporal_pairwise_dataset.py prepare \
    --dataset "$SOURCE_DATASET" \
    --era-corpus "$ERA_CORPUS" \
    --output-dir "$PAIR_DIR" \
    --train-candidates 220 --val-candidates 25 \
    --train-pairs 600 --val-pairs 60 --seed 1969
}

if [[ "$MODE" == --preflight ]]; then
  command -v podman >/dev/null
  podman inspect strix-halo-finetuning >/dev/null
  podman inspect llama-rocm-7.2 >/dev/null
  python3 scripts/train_deepred_pairwise.py --help | grep -q -- '--chosen-ce-weight'
  python3 scripts/diagnose_temporal_policy.py score --help | grep -q -- '--tokenizer'
  if [[ "$REUSE_PAIR_DATASET" == 1 ]]; then
    require_pair_dataset
  else
    prepare_candidates
  fi
  python3 scripts/evaluate_deepred_models.py validate \
    --models "$BASE_REGISTRY" --probes "$PAIR_DIR/probes.jsonl" \
    --require-paths --verify-hashes
  python3 - "$PAIR_DIR" "$REUSE_PAIR_DATASET" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
reuse_pairs = sys.argv[2] == '1'
rows = [json.loads(line) for line in (root / 'candidates.jsonl').open()]
counts = Counter((row['split'], row['mode']) for row in rows)
expected = {
    **{('train', mode): 220 for mode in ('in_world', 'hedged', 'premise_correction')},
    **{('val', mode): 25 for mode in ('in_world', 'hedged', 'premise_correction')},
}
if counts != expected:
    raise SystemExit(f'unexpected candidate mix: {counts}')
prefixes = ('pair', 'anchor') if reuse_pairs else ('anchor',)
for prefix in prefixes:
  for split, expected_count in (('train', 600), ('val', 60)):
    count = sum(1 for line in (root / f'{prefix}_{split}.jsonl').open()
          if line.strip())
    if count != expected_count:
      raise SystemExit(f'unexpected {prefix} {split} count: {count}')
print('Preflight passed: 735 candidates, 660 pairs, 660 anchors, 300 training steps')
PY
  exit 0
fi

mkdir -p "$PAIR_DIR" "$TRAIN_DIR" "$RUN_DIR"
if [[ "$REUSE_PAIR_DATASET" == 1 ]]; then
  require_pair_dataset
  printf '\n== Reuse finalized pair dataset: %s ==\n' "$PAIR_DIR"
else
  prepare_candidates

  printf '\n== Generate untouched-base rejected completions ==\n'
  podman stop strix-halo-finetuning >/dev/null 2>&1 || true
  podman start llama-rocm-7.2 >/dev/null
  python3 scripts/evaluate_deepred_models.py run \
    --models "$BASE_REGISTRY" \
    --probes "$PAIR_DIR/probes.jsonl" \
    --output-dir "$PAIR_DIR/base-generations" --suite-tag v5-pairs \
    --model-id "$BASE_ID" \
    --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
    --context-size 4096 --timeout 600 \
    --server-container llama-rocm-7.2 \
    --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    --gpu-layers all --flash-attention on --no-mmap \
    2>&1 | tee -a "$PAIR_DIR/base-generation.log"

  python3 scripts/build_temporal_pairwise_dataset.py finalize \
    --candidates "$PAIR_DIR/candidates.jsonl" \
    --generations "$PAIR_DIR/base-generations/generations.jsonl" \
    --model-id "$BASE_ID" --output-dir "$PAIR_DIR" \
    --train-pairs 600 --val-pairs 60 --seed 1969
fi

printf '\n== Train %s pairwise pilot ==\n' "$VARIANT"
podman stop llama-rocm-7.2 >/dev/null 2>&1 || true
podman start strix-halo-finetuning >/dev/null
podman exec strix-halo-finetuning bash -lc "
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/train_deepred_pairwise.py \\
    --model '$INITIAL_MODEL' --tokenizer '$BASE_MODEL' \\
    --dataset '$PAIR_DIR' --output-dir '$TRAIN_DIR' \\
    --margin-target 0.25 --chosen-ce-weight '$CHOSEN_CE_WEIGHT' \\
    --learning-rate 1e-6 \\
    --max-length 768 --gradient-accumulation 16 --max-steps 300 \\
    --snapshot-at 10 25 50 75 100
" 2>&1 | tee -a "$TRAIN_DIR/console.log"
require_dir "$TRAIN_DIR/final"

printf '\n== Export Q8_0 trajectory ==\n'
for tag in "${SNAPSHOT_TAGS[@]}"; do
  matches=("$TRAIN_DIR"/snapshots/"${tag}"pct-step-*)
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Expected one ${tag}% snapshot, found ${#matches[@]}" >&2
    exit 1
  fi
  outfile="$RUN_DIR/deepred-${VARIANT}-${tag}-q8_0.gguf"
  if [[ -s "$outfile" ]]; then
    echo "Reusing existing GGUF: $outfile"
  else
    python3 scripts/export_gguf.py \
      --model-dir "${matches[0]}" --outfile "$outfile" --quant Q8_0
  fi
done

printf '\n== Build and validate model registry ==\n'
python3 - "$RUN_DIR" "$BASE_REGISTRY" "$VARIANT" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

run = Path(sys.argv[1])
source = json.loads(Path(sys.argv[2]).read_text())
variant = sys.argv[3]
base = next(model for model in source['models']
            if model['id'] == 'gemma-3-4b-it-base-q4')
models = [base]
for path in sorted(run.glob(f'deepred-{variant}-[0-9][0-9][0-9]-q8_0.gguf')):
    tag = re.search(r'-([0-9]{3})-q8_0$', path.stem).group(1)
    models.append({
        'id': f'deepred-{variant}-{tag}-q8', 'family': f'{variant}_pairwise',
        'role': 'trajectory', 'format': 'gguf', 'path': str(path),
        'quantization': 'q8_0', 'sha256': sha256(path),
        'bytes': path.stat().st_size,
    })
(run / 'models.json').write_text(json.dumps({
    'schema_version': 1, 'models': models,
}, indent=2) + '\n')
PY
python3 scripts/evaluate_deepred_models.py validate \
  --models "$RUN_DIR/models.json" --probes "$PROBES" \
  --require-paths --verify-hashes

printf '\n== Evaluate frozen coarse suite ==\n'
podman stop strix-halo-finetuning >/dev/null 2>&1 || true
podman start llama-rocm-7.2 >/dev/null
MODEL_ARGS=(--model-id "$BASE_ID")
for tag in "${SNAPSHOT_TAGS[@]}"; do
  MODEL_ARGS+=(--model-id "deepred-${VARIANT}-${tag}-q8")
done
python3 scripts/evaluate_deepred_models.py run \
  --models "$RUN_DIR/models.json" --probes "$PROBES" \
  --output-dir "$RUN_DIR" --suite-tag coarse "${MODEL_ARGS[@]}" \
  --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
  --context-size 4096 --timeout 600 \
  --server-container llama-rocm-7.2 \
  --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  --gpu-layers all --flash-attention on --no-mmap \
  2>&1 | tee -a "$RUN_DIR/run.log"
python3 scripts/evaluate_deepred_models.py score \
  --probes "$PROBES" --generations "$RUN_DIR/generations.jsonl" \
  --output "$RUN_DIR/scores.json"
python3 scripts/evaluate_deepred_models.py report \
  --scores "$RUN_DIR/scores.json" --generations "$RUN_DIR/generations.jsonl" \
  --output "$RUN_DIR/report.md"

for tag in "${SNAPSHOT_TAGS[@]}"; do
  model="deepred-${VARIANT}-${tag}-q8"
  python3 scripts/evaluate_deepred_models.py gates \
    --scores "$RUN_DIR/scores.json" --model-id "$model" \
    --base-model-id "$BASE_ID" --output "$RUN_DIR/release-gates-${tag}.json" || true
done

printf '\n== Score HF completion margins ==\n'
podman stop llama-rocm-7.2 >/dev/null 2>&1 || true
podman start strix-halo-finetuning >/dev/null
HF_ARGS=()
for tag in "${SNAPSHOT_TAGS[@]}"; do
  matches=("$TRAIN_DIR"/snapshots/"${tag}"pct-step-*)
  HF_ARGS+=(--model "${VARIANT}-${tag}=${matches[0]}")
done
podman exec strix-halo-finetuning bash -lc "
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/diagnose_temporal_policy.py score \\
    --pairs '$DIAG_DIR/pairs.jsonl' --tokenizer '$BASE_MODEL' \\
    ${HF_ARGS[*]} --batch-size 4 --max-length 768 \\
    --output '$RUN_DIR/margins.json'
" 2>&1 | tee -a "$RUN_DIR/margins.log"

printf '\n== Apply %s experiment gates ==\n' "$VARIANT"
python3 - "$RUN_DIR" "$VARIANT" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
variant = sys.argv[2]
scores = json.loads((run / 'scores.json').read_text())
margins = json.loads((run / 'margins.json').read_text())
margin_by_model = {model['model_id']: model for model in margins['models']}
rows = []
for tag in ('010', '025', '050', '075', '100'):
    model_id = f'deepred-{variant}-{tag}-q8'
    release = json.loads((run / f'release-gates-{tag}.json').read_text())
    metrics = release['metrics']
    margin_model = margin_by_model[f'{variant}-{tag}']
    validation = next(row for row in margin_model['summary']
                      if row['split'] == 'val' and row['mode'] == 'all')
    checks = {
        'utility': metrics['utility'] is not None and metrics['utility'] >= 0.90,
        'pre_1969_recall': (metrics['pre_1969_recall'] is not None
                            and metrics['pre_1969_recall'] >= 0.85),
        'validation_margin': validation['mean_margin'] >= -0.25,
        'validation_win_rate': validation['win_rate'] >= 0.40,
        'era_native': metrics['era_native'] is not None and metrics['era_native'] >= 0.30,
        'conversational_leak': (metrics['conversational_modern_leak'] is not None
                                and metrics['conversational_modern_leak'] <= 0.70),
        'repetition': (metrics['repetition_or_boilerplate'] is not None
                       and metrics['repetition_or_boilerplate'] <= 0.05),
    }
    result = {
        'model_id': model_id, 'passed': all(checks.values()),
        'checks': checks, 'metrics': metrics, 'validation_margin': validation,
    }
    (run / f'experiment-gates-{tag}.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    rows.append(result)

rows.sort(key=lambda row: (
    not row['passed'], -row['metrics']['era_native'],
    row['metrics']['conversational_modern_leak'],
    -row['validation_margin']['mean_margin']))
print('model                    utility  pre-1969  margin   wins  leak    era     result')
for row in rows:
    m = row['metrics']; v = row['validation_margin']
    print(f"{row['model_id']:24} {m['utility']:7.1%} {m['pre_1969_recall']:9.1%} "
          f"{v['mean_margin']:7.3f} {v['win_rate']:6.1%} "
          f"{m['conversational_modern_leak']:7.1%} {m['era_native']:7.1%} "
          f"{'PASS' if row['passed'] else 'FAIL'}")
if not any(row['passed'] for row in rows):
    print('\nNo snapshot passed. Do not proceed to V5B or persona.')
else:
    print(f"\nBest passing checkpoint: {rows[0]['model_id']}")
PY

printf '\n%s pairwise pipeline complete: %s\n' "$VARIANT" "$RUN_DIR"