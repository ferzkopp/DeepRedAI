#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
VARIANT=${VARIANT:-v7}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
INITIAL_MODEL=${INITIAL_MODEL:-/mnt/data/models/gemma-3-4b-it}
CORPUS_DIR=${CORPUS_DIR:-/mnt/data/deepred_corpus/v2}
SYSTEM_PROMPTS=${SYSTEM_PROMPTS:-/mnt/data/deepred_corpus/v3/system_prompts.jsonl}
EVAL_VARIANT=${EVAL_VARIANT:-sp-holdout-01}
DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-v3-conditioned}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-${VARIANT}-conditioned}
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/${VARIANT}-conditioned-$(date +%Y-%m-%d)}
PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
BASE_REGISTRY=$ROOT/evaluation/deepred_1969/models.json
BASE_ID=gemma-3-4b-it-base-q4
SNAPSHOT_TAGS=(010 025 050 075 100)
EPOCHS=${EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
MODE=${1:-run}

if [[ "$MODE" != run && "$MODE" != --preflight ]]; then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

LOCK_DIR=/tmp/deepred-${VARIANT}-conditioned.lock.d
acquire_lock() {
  local owner=''
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return
  fi
  [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
  if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
    echo "Another ${VARIANT} run is active (pid $owner)." >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another ${VARIANT} run acquired the lock." >&2
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
require_dir "$CORPUS_DIR"
require_file "$SYSTEM_PROMPTS"
require_file "$PROBES"
require_file "$BASE_REGISTRY"

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh"

require_dataset() {
  for name in retain_train retain_val forget_train forget_val manifest; do
    case "$name" in
      manifest) require_file "$DATASET/manifest.json" ;;
      *) require_file "$DATASET/$name.jsonl" ;;
    esac
  done
}

write_system_prompt() {
  python3 - "$SYSTEM_PROMPTS" "$EVAL_VARIANT" "$1" <<'PY'
import json, sys
from pathlib import Path

variants = [json.loads(line) for line in Path(sys.argv[1]).open() if line.strip()]
match = next((row for row in variants if row['id'] == sys.argv[2]), None)
if match is None:
    raise SystemExit(f'unknown evaluation variant: {sys.argv[2]}')
Path(sys.argv[3]).write_text(match['text'] + '\n')
PY
}

check_dataset() {
  python3 - "$DATASET" "$EVAL_VARIANT" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
if manifest['counts']['forget_train'] or manifest['counts']['forget_val']:
    raise SystemExit('forget rows present; plain SFT would teach modern facts')
if sys.argv[2] not in manifest['held_out_system_variants']:
    raise SystemExit(f'{sys.argv[2]} is not held out of training')
if not manifest['strip_chess_footer']:
    raise SystemExit('chess footers must be stripped from training targets')
rows = [json.loads(line) for line in (root / 'retain_train.jsonl').open()]
kinds = Counter(row['kind'] for row in rows)
conditioned = sum(1 for row in rows if row.get('system_variant'))
persona, controls = kinds['persona'], kinds['persona_controls']
ratio = controls / persona if persona else 0
if not 0.15 <= ratio <= 0.45:
    raise SystemExit(f'plain-control ratio {ratio:.0%} outside 15-45%')
for row in rows:
    for message in row['messages']:
        if message['role'] == 'assistant' and '[DR:' in message['content']:
            raise SystemExit(f'{row["id"]}: chess footer in training target')
print(f'Preflight passed: {len(rows):,} train rows, kinds {dict(sorted(kinds.items()))}, '
      f'control ratio {ratio:.0%}, conditioned {conditioned / len(rows):.0%}')
PY
}

if [[ "$MODE" == --preflight ]]; then
  command -v podman >/dev/null
  podman inspect strix-halo-finetuning >/dev/null
  podman inspect llama-rocm-7.2 >/dev/null
  python3 scripts/train_deepred_sft.py --help | grep -q -- '--epochs'
  python3 scripts/evaluate_deepred_models.py run --help | grep -q -- '--system-file'
  require_dataset
  check_dataset
  python3 scripts/evaluate_deepred_models.py audit \
    --probes "$PROBES" \
    --corpus "$DATASET/retain_train.jsonl" \
    --corpus "$DATASET/retain_val.jsonl" \
    --output "$DATASET/contamination.json"
  python3 scripts/evaluate_deepred_models.py validate \
    --models "$BASE_REGISTRY" --probes "$PROBES" \
    --require-paths --verify-hashes
  exit 0
fi

require_dataset
check_dataset
mkdir -p "$TRAIN_DIR" "$RUN_DIR"
write_system_prompt "$RUN_DIR/system_prompt.txt"

printf '\n== Train %s conditioned SFT ==\n' "$VARIANT"
podman stop llama-rocm-7.2 >/dev/null 2>&1 || true
podman start strix-halo-finetuning >/dev/null
podman exec strix-halo-finetuning bash -lc "
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/train_deepred_sft.py \\
    --model '$INITIAL_MODEL' --tokenizer '$BASE_MODEL' \\
    --dataset '$DATASET' --output-dir '$TRAIN_DIR' \\
    --learning-rate '$LEARNING_RATE' --epochs '$EPOCHS' \\
    --max-length 768 --gradient-accumulation 16 \\
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
        'id': f'deepred-{variant}-{tag}-q8', 'family': f'{variant}_conditioned',
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

podman stop strix-halo-finetuning >/dev/null 2>&1 || true
podman start llama-rocm-7.2 >/dev/null
MODEL_ARGS=(--model-id "$BASE_ID")
for tag in "${SNAPSHOT_TAGS[@]}"; do
  MODEL_ARGS+=(--model-id "deepred-${VARIANT}-${tag}-q8")
done

# The served prompt is held out of training, so a pass proves rule-following.
for condition in with-system no-system; do
  printf '\n== Evaluate frozen coarse suite (%s) ==\n' "$condition"
  mkdir -p "$RUN_DIR/$condition"
  SYSTEM_ARGS=()
  if [[ "$condition" == with-system ]]; then
    SYSTEM_ARGS=(--system-file "$RUN_DIR/system_prompt.txt")
  fi
  python3 scripts/evaluate_deepred_models.py run \
    --models "$RUN_DIR/models.json" --probes "$PROBES" \
    --output-dir "$RUN_DIR/$condition" --suite-tag coarse "${MODEL_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
    --context-size 4096 --timeout 600 \
    --server-container llama-rocm-7.2 \
    --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    --gpu-layers all --flash-attention on --no-mmap \
    2>&1 | tee -a "$RUN_DIR/$condition/run.log"
  python3 scripts/evaluate_deepred_models.py score \
    --probes "$PROBES" --generations "$RUN_DIR/$condition/generations.jsonl" \
    --output "$RUN_DIR/$condition/scores.json"
  python3 scripts/evaluate_deepred_models.py report \
    --scores "$RUN_DIR/$condition/scores.json" \
    --generations "$RUN_DIR/$condition/generations.jsonl" \
    --output "$RUN_DIR/$condition/report.md"
  for tag in "${SNAPSHOT_TAGS[@]}"; do
    python3 scripts/evaluate_deepred_models.py gates \
      --scores "$RUN_DIR/$condition/scores.json" \
      --model-id "deepred-${VARIANT}-${tag}-q8" \
      --base-model-id "$BASE_ID" \
      --output "$RUN_DIR/$condition/release-gates-${tag}.json" || true
  done
done

printf '\n== Apply %s experiment gates ==\n' "$VARIANT"
python3 - "$RUN_DIR" "$VARIANT" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
variant = sys.argv[2]
tags = ('010', '025', '050', '075', '100')

def metrics(condition, tag):
    path = run / condition / f'release-gates-{tag}.json'
    return json.loads(path.read_text())['metrics']

rows = []
for tag in tags:
    conditioned = metrics('with-system', tag)
    plain = metrics('no-system', tag)
    checks = {
        'utility': (conditioned['utility'] or 0) >= 0.90,
        'pre_1969_recall': (conditioned['pre_1969_recall'] or 0) >= 0.85,
        'era_native': (conditioned['era_native'] or 0) >= 0.50,
        'conversational_leak': (conditioned['conversational_modern_leak'] or 1) <= 0.40,
        'persona': (conditioned['persona'] or 0) >= 0.50,
        'repetition': (conditioned['repetition_or_boilerplate'] or 1) <= 0.05,
    }
    result = {
        'model_id': f'deepred-{variant}-{tag}-q8',
        'passed': all(checks.values()), 'checks': checks,
        'with_system': conditioned, 'no_system': plain,
    }
    (run / f'experiment-gates-{tag}.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    rows.append(result)

print('model                  util   pre69  era    leak   persona | era(no-sys) result')
for row in rows:
    c, p = row['with_system'], row['no_system']
    print(f"{row['model_id']:22} {c['utility']:5.1%} {c['pre_1969_recall']:6.1%} "
          f"{c['era_native']:6.1%} {c['conversational_modern_leak']:6.1%} "
          f"{c['persona']:7.1%} | {p['era_native']:10.1%} "
          f"{'PASS' if row['passed'] else 'FAIL'}")

best = [row for row in rows if row['passed']]
if not best:
    print('\nNo snapshot passed the conditioned gate. Do not start distillation.')
else:
    print(f"\nBest conditioned checkpoint: {best[0]['model_id']}")
    print('Proceed to the v8 context-distillation stage.')
PY

printf '\n%s conditioned pipeline complete: %s\n' "$VARIANT" "$RUN_DIR"
