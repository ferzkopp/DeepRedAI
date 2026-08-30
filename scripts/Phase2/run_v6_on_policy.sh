#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
INITIAL_MODEL=${INITIAL_MODEL:-/mnt/data/training_output/deepred-npo-v4-temporal/final}
V5_PAIR_DIR=${V5_PAIR_DIR:-/mnt/data/sft_corpus/deepred-v5a-pairwise}
V5_TRAIN_DIR=${V5_TRAIN_DIR:-/mnt/data/training_output/deepred-v5a-pairwise}
V5_RUN_DIR=${V5_RUN_DIR:-/mnt/data/evaluations/deepred-1969/v5a-pairwise-2026-08-28}
PAIR_DIR=${PAIR_DIR:-/mnt/data/sft_corpus/deepred-v6a-on-policy}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-v6a-on-policy}
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/v6a-on-policy-$(date +%Y-%m-%d)}
POLICY_DIAG=${POLICY_DIAG:-/mnt/data/evaluations/deepred-1969/temporal-policy-diagnostic-2026-08-28}
PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
BASE_REGISTRY=$ROOT/evaluation/deepred_1969/models.json
BASE_ID=gemma-3-4b-it-base-q4
V5_ID=deepred-v5a-100-q8
SNAPSHOT_TAGS=(010 025 050 075 100)
TRAIN_PAIRS=567
VAL_PAIRS=60
MODE=${1:-run}

if [[ "$MODE" != run && "$MODE" != --preflight && "$MODE" != --diagnostic ]]; then
  echo "Usage: $0 [--preflight|--diagnostic]" >&2
  exit 2
fi

LOCK_DIR=/tmp/deepred-v6a-on-policy.lock.d
acquire_lock() {
  local owner=''
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return
  fi
  [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
  if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
    echo "Another v6a on-policy run is active (pid $owner)." >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another v6a on-policy run acquired the lock." >&2
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
require_dir "$V5_PAIR_DIR"
require_dir "$V5_TRAIN_DIR/final"
require_dir "$V5_RUN_DIR"
require_file "$V5_PAIR_DIR/candidates.jsonl"
require_file "$V5_PAIR_DIR/probes.jsonl"
require_file "$V5_PAIR_DIR/base-generations/generations.jsonl"
require_file "$V5_RUN_DIR/models.json"
require_file "$POLICY_DIAG/pairs.jsonl"
require_file "$PROBES"
require_file "$BASE_REGISTRY"

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh"

if [[ "$MODE" == --preflight ]]; then
  command -v podman >/dev/null
  podman inspect strix-halo-finetuning >/dev/null
  podman inspect llama-rocm-7.2 >/dev/null
  python3 scripts/diagnose_on_policy_negatives.py score --help |
    grep -q -- '--tokenizer'
  python3 scripts/evaluate_deepred_models.py validate \
    --models "$V5_RUN_DIR/models.json" --probes "$V5_PAIR_DIR/probes.jsonl" \
    --require-paths --verify-hashes
  python3 - "$V5_PAIR_DIR" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
candidates = [json.loads(line) for line in (root / 'candidates.jsonl').open()]
counts = Counter((row['split'], row['mode']) for row in candidates)
expected = {
    **{('train', mode): 220 for mode in ('in_world', 'hedged', 'premise_correction')},
    **{('val', mode): 25 for mode in ('in_world', 'hedged', 'premise_correction')},
}
if counts != expected:
    raise SystemExit(f'unexpected V5 candidate mix: {counts}')
print('Preflight passed: 735 V5 candidates; V6A uses 567/60 pairs and 300 steps')
PY
  exit 0
fi

prepare_on_policy_data() {
  mkdir -p "$PAIR_DIR" "$RUN_DIR"
  printf '\n== Generate V5A-final on-policy completions ==\n'
  podman stop strix-halo-finetuning >/dev/null 2>&1 || true
  podman start llama-rocm-7.2 >/dev/null
  python3 scripts/evaluate_deepred_models.py run \
    --models "$V5_RUN_DIR/models.json" \
    --probes "$V5_PAIR_DIR/probes.jsonl" \
    --output-dir "$PAIR_DIR/on-policy-generations" --suite-tag v5-pairs \
    --model-id "$V5_ID" \
    --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
    --context-size 4096 --timeout 600 \
    --server-container llama-rocm-7.2 \
    --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    --gpu-layers all --flash-attention on --no-mmap \
    2>&1 | tee -a "$PAIR_DIR/on-policy-generation.log"

  python3 scripts/build_temporal_pairwise_dataset.py finalize \
    --candidates "$V5_PAIR_DIR/candidates.jsonl" \
    --generations "$PAIR_DIR/on-policy-generations/generations.jsonl" \
    --model-id "$V5_ID" --output-dir "$PAIR_DIR" \
    --train-pairs "$TRAIN_PAIRS" --val-pairs "$VAL_PAIRS" --seed 1969
  python3 - "$V5_PAIR_DIR" "$PAIR_DIR" "$TRAIN_PAIRS" "$VAL_PAIRS" <<'PY'
import json, sys
from pathlib import Path

source, output = map(Path, sys.argv[1:3])
for split, count in (('train', int(sys.argv[3])), ('val', int(sys.argv[4]))):
    rows = [json.loads(line) for line in
            (source / f'anchor_{split}.jsonl').open() if line.strip()]
    if len(rows) < count:
        raise SystemExit(f'{split} has {len(rows)} anchors, need {count}')
    with (output / f'anchor_{split}.jsonl').open('w') as handle:
        for row in rows[:count]:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
PY

  python3 scripts/diagnose_on_policy_negatives.py prepare \
    --pair-dir "$PAIR_DIR" \
    --base-generations "$V5_PAIR_DIR/base-generations/generations.jsonl" \
    --base-model-id "$BASE_ID" \
    --output "$PAIR_DIR/diagnostic-pairs.jsonl" \
    --train-per-mode 20 --seed 1969
}

score_on_policy_diagnostic() {
  printf '\n== Score original and on-policy negatives ==\n'
  podman stop llama-rocm-7.2 >/dev/null 2>&1 || true
  podman start strix-halo-finetuning >/dev/null
  podman exec strix-halo-finetuning bash -lc "
    cd /mnt/data/DeepRedAI
    /opt/venv/bin/python3 scripts/diagnose_on_policy_negatives.py score \\
      --pairs '$PAIR_DIR/diagnostic-pairs.jsonl' \\
      --tokenizer '$BASE_MODEL' \\
      --model v4='$INITIAL_MODEL' \\
      --model v5a='$V5_TRAIN_DIR/final' \\
      --batch-size 4 --max-length 768 \\
      --output '$PAIR_DIR/diagnostic-margins.json'
  " 2>&1 | tee -a "$PAIR_DIR/diagnostic-margins.log"
}

apply_diagnostic_gate() {
  python3 - "$PAIR_DIR" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
pairs = [json.loads(line) for line in (root / 'diagnostic-pairs.jsonl').open()]
report = json.loads((root / 'diagnostic-margins.json').read_text())
models = {model['model_id']: model for model in report['models']}
overall = {}
validation = {}
for model_id, model in models.items():
    overall[model_id] = next(row for row in model['summary']
                             if row['split'] == row['mode'] == 'all')
    validation[model_id] = next(row for row in model['summary']
                                if row['split'] == 'val' and row['mode'] == 'all')

fresh_behaviors = Counter(row['fresh_behavior'] for row in pairs)
checks = {
    'complete_balanced_sample': len(pairs) == 120,
    'fresh_negatives_modern': fresh_behaviors == {'confident_unsupported': 120},
    'v5_suppressed_original': (
        overall['v5a']['original_rejected_mean_logp']
        <= overall['v4']['original_rejected_mean_logp'] - 0.10),
    'fresh_negative_is_harder': (
        overall['v5a']['fresh_minus_original_logp'] >= 0.10),
    'fresh_validation_margin_negative': (
        validation['v5a']['fresh_mean_margin'] < 0),
    'fresh_validation_win_rate_below_gate': (
        validation['v5a']['fresh_win_rate'] < 0.40),
    'fresh_first_sentence_margin_negative': (
        validation['v5a']['fresh_first_mean_margin'] < 0),
}
result = {
    'schema_version': 1, 'passed': all(checks.values()), 'checks': checks,
    'fresh_behaviors': dict(sorted(fresh_behaviors.items())),
    'v4_overall': overall['v4'], 'v5a_overall': overall['v5a'],
    'v5a_validation': validation['v5a'],
}
(root / 'diagnostic-gate.json').write_text(
    json.dumps(result, indent=2, sort_keys=True) + '\n')
print('V6A on-policy diagnostic gate')
for name, passed in checks.items():
    print(f'  {name:40} {"PASS" if passed else "FAIL"}')
print(f"  V5 fresh val margin: {validation['v5a']['fresh_mean_margin']:.3f}")
print(f"  V5 fresh val wins:   {validation['v5a']['fresh_win_rate']:.1%}")
print(f"  fresh-original logp: {overall['v5a']['fresh_minus_original_logp']:+.3f}")
if not result['passed']:
    raise SystemExit('Diagnostic did not confirm negative routing; V6A training blocked.')
print('Diagnostic confirmed negative routing; V6A training is authorized.')
PY
}

prepare_on_policy_data
score_on_policy_diagnostic
apply_diagnostic_gate

if [[ "$MODE" == --diagnostic ]]; then
  printf '\nV6A diagnostic complete: %s\n' "$PAIR_DIR"
  exit 0
fi

mkdir -p "$TRAIN_DIR" "$RUN_DIR"
printf '\n== Train V6A with refreshed on-policy negatives ==\n'
podman start strix-halo-finetuning >/dev/null
podman exec strix-halo-finetuning bash -lc "
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/train_deepred_pairwise.py \\
    --model '$INITIAL_MODEL' --tokenizer '$BASE_MODEL' \\
    --dataset '$PAIR_DIR' --output-dir '$TRAIN_DIR' \\
    --margin-target 0.25 --learning-rate 1e-6 \\
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
  outfile="$RUN_DIR/deepred-v6a-${tag}-q8_0.gguf"
  if [[ -s "$outfile" ]]; then
    echo "Reusing existing GGUF: $outfile"
  else
    python3 scripts/export_gguf.py \
      --model-dir "${matches[0]}" --outfile "$outfile" --quant Q8_0
  fi
done

printf '\n== Build and validate model registry ==\n'
python3 - "$RUN_DIR" "$BASE_REGISTRY" <<'PY'
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
base = next(model for model in source['models']
            if model['id'] == 'gemma-3-4b-it-base-q4')
models = [base]
for path in sorted(run.glob('deepred-v6a-[0-9][0-9][0-9]-q8_0.gguf')):
    tag = re.search(r'-([0-9]{3})-q8_0$', path.stem).group(1)
    models.append({
        'id': f'deepred-v6a-{tag}-q8', 'family': 'v6a_on_policy',
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
  MODEL_ARGS+=(--model-id "deepred-v6a-${tag}-q8")
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
  python3 scripts/evaluate_deepred_models.py gates \
    --scores "$RUN_DIR/scores.json" --model-id "deepred-v6a-${tag}-q8" \
    --base-model-id "$BASE_ID" \
    --output "$RUN_DIR/release-gates-${tag}.json" || true
done

printf '\n== Score V6A completion margins ==\n'
podman stop llama-rocm-7.2 >/dev/null 2>&1 || true
podman start strix-halo-finetuning >/dev/null
HF_ARGS=()
for tag in "${SNAPSHOT_TAGS[@]}"; do
  matches=("$TRAIN_DIR"/snapshots/"${tag}"pct-step-*)
  HF_ARGS+=(--model "v6a-${tag}=${matches[0]}")
done
podman exec strix-halo-finetuning bash -lc "
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/diagnose_temporal_policy.py score \\
    --pairs '$POLICY_DIAG/pairs.jsonl' --tokenizer '$BASE_MODEL' \\
    ${HF_ARGS[*]} --batch-size 4 --max-length 768 \\
    --output '$RUN_DIR/margins.json'
  /opt/venv/bin/python3 scripts/diagnose_on_policy_negatives.py score \\
    --pairs '$PAIR_DIR/diagnostic-pairs.jsonl' --tokenizer '$BASE_MODEL' \\
    ${HF_ARGS[*]} --batch-size 4 --max-length 768 \\
    --output '$RUN_DIR/on-policy-margins.json'
" 2>&1 | tee -a "$RUN_DIR/margins.log"

printf '\n== Apply V6A experiment gates ==\n'
python3 - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
on_policy = json.loads((run / 'on-policy-margins.json').read_text())
margin_by_model = {model['model_id']: model for model in on_policy['models']}
rows = []
for tag in ('010', '025', '050', '075', '100'):
    model_id = f'deepred-v6a-{tag}-q8'
    release = json.loads((run / f'release-gates-{tag}.json').read_text())
    metrics = release['metrics']
    validation = next(
        row for row in margin_by_model[f'v6a-{tag}']['summary']
        if row['split'] == 'val' and row['mode'] == 'all')
    checks = {
        'utility': metrics['utility'] is not None and metrics['utility'] >= 0.90,
        'pre_1969_recall': (metrics['pre_1969_recall'] is not None
                            and metrics['pre_1969_recall'] >= 0.85),
        'validation_margin': validation['fresh_mean_margin'] >= -0.25,
        'validation_win_rate': validation['fresh_win_rate'] >= 0.40,
        'first_sentence_margin': validation['fresh_first_mean_margin'] >= -0.25,
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
    -row['validation_margin']['fresh_mean_margin']))
print('model                    utility  pre-1969  margin   wins  first   leak    era     V6A')
for row in rows:
    metrics = row['metrics']; margin = row['validation_margin']
    print(f"{row['model_id']:24} {metrics['utility']:7.1%} "
          f"{metrics['pre_1969_recall']:9.1%} "
          f"{margin['fresh_mean_margin']:7.3f} "
          f"{margin['fresh_win_rate']:6.1%} "
          f"{margin['fresh_first_mean_margin']:7.3f} "
          f"{metrics['conversational_modern_leak']:7.1%} "
          f"{metrics['era_native']:7.1%} "
          f"{'PASS' if row['passed'] else 'FAIL'}")
if not any(row['passed'] for row in rows):
    print('\nNo snapshot passed. Do not proceed to V5B or persona.')
else:
    print(f"\nBest passing checkpoint: {rows[0]['model_id']}")
PY

printf '\nV6A on-policy pipeline complete: %s\n' "$RUN_DIR"