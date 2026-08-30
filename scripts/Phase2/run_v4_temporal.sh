#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-npo-v4-temporal}
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/npo-v4-temporal-$(date +%Y-%m-%d)}
DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-v2}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
REFERENCE_CACHE=${REFERENCE_CACHE:-/mnt/data/training_output/deepred-npo-v2/reference_logps.json}
PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
BASE_REGISTRY=$ROOT/evaluation/deepred_1969/models.json
SNAPSHOT_TAGS=(005 010 020 030 040 050 065 080 100)
MODE=${1:-run}

if [[ "$MODE" != run && "$MODE" != --preflight ]]; then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

exec 9>"/tmp/deepred-v4-temporal.lock"
if ! flock -n 9; then
  echo "Another v4 temporal run is active." >&2
  exit 1
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_dir "$ROOT"
require_dir "$DATASET"
require_dir "$BASE_MODEL"
require_file "$REFERENCE_CACHE"
require_file "$PROBES"
require_file "$BASE_REGISTRY"

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh"

if [[ "$MODE" == --preflight ]]; then
  command -v podman >/dev/null
  command -v flock >/dev/null
  podman inspect strix-halo-finetuning >/dev/null
  podman inspect llama-rocm-7.2 >/dev/null
  python3 scripts/train_deepred_npo.py --help | grep -q -- '--npo-weight'
  python3 - <<'PY'
from collections import Counter
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    'npo', '/mnt/data/DeepRedAI/scripts/train_deepred_npo.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = Path('/mnt/data/sft_corpus/deepred-v2')
rows = module.balanced_rows(
    module.load_jsonl(root / 'forget_train.jsonl'),
    module.load_jsonl(root / 'retain_train.jsonl'),
    1969, 0.30,
    {'era_native': 3, 'persona': 0, 'persona_controls': 0},
)
counts = Counter(row['kind'] for row in rows)
expected = {'forget': 10193, 'era_native': 14259, 'retain': 9524}
if len(rows) != 33976 or dict(counts) != expected:
    raise SystemExit(f'unexpected v4 stream: total={len(rows)} kinds={counts}')
print(f'Preflight passed: {len(rows):,} microexamples, 2,124 optimizer steps')
PY
  exit 0
fi

mkdir -p "$TRAIN_DIR" "$RUN_DIR"
if [[ ! -f "$TRAIN_DIR/reference_logps.json" ]]; then
  cp "$REFERENCE_CACHE" "$TRAIN_DIR/reference_logps.json"
fi

printf '\n== Train temporal v4 ==\n'
podman stop llama-rocm-7.2 >/dev/null 2>&1 || true
podman start strix-halo-finetuning >/dev/null
set -o pipefail
podman exec strix-halo-finetuning bash -lc "
  cd /mnt/data/DeepRedAI
  /opt/venv/bin/python3 scripts/train_deepred_npo.py \\
    --base-model '$BASE_MODEL' \\
    --dataset '$DATASET' \\
    --output-dir '$TRAIN_DIR' \\
    --beta 0.1 --npo-weight 0.03 --retain-weight 1 \\
    --forget-ratio 0.30 --learning-rate 2e-6 \\
    --kind-weight era_native=3 \\
    --kind-weight persona=0 \\
    --kind-weight persona_controls=0 \\
    --snapshot-at 5 10 20 30 40 50 65 80 100
" 2>&1 | tee -a "$TRAIN_DIR/console.log"
require_dir "$TRAIN_DIR/final"

printf '\n== Export Q8_0 snapshots ==\n'
for tag in "${SNAPSHOT_TAGS[@]}"; do
  matches=("$TRAIN_DIR"/snapshots/"${tag}"pct-step-*)
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Expected one ${tag}% snapshot, found ${#matches[@]}" >&2
    exit 1
  fi
  outfile="$RUN_DIR/deepred-npo-v4-${tag}-q8_0.gguf"
  if [[ -s "$outfile" ]]; then
    echo "Reusing existing GGUF: $outfile"
  else
    python3 scripts/export_gguf.py \
      --model-dir "${matches[0]}" \
      --outfile "$outfile" \
      --quant Q8_0
  fi
done
final_gguf="$RUN_DIR/deepred-npo-v4-final-q8_0.gguf"
if [[ -s "$final_gguf" ]]; then
  echo "Reusing existing GGUF: $final_gguf"
else
  python3 scripts/export_gguf.py \
    --model-dir "$TRAIN_DIR/final" \
    --outfile "$final_gguf" \
    --quant Q8_0
fi

printf '\n== Build and validate model registry ==\n'
python3 - "$RUN_DIR" "$BASE_REGISTRY" <<'PY'
import hashlib
import json
import re
import sys
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
for path in sorted(run.glob('deepred-npo-v4-[0-9][0-9][0-9]-q8_0.gguf')):
    tag = re.search(r'-([0-9]{3})-q8_0$', path.stem).group(1)
    models.append({
        'id': f'deepred-npo-v4-{tag}-q8',
        'family': 'npo_v4_temporal',
        'role': 'trajectory',
        'format': 'gguf',
        'path': str(path),
        'quantization': 'q8_0',
        'sha256': sha256(path),
        'bytes': path.stat().st_size,
    })
final_path = run / 'deepred-npo-v4-final-q8_0.gguf'
models.append({
    'id': 'deepred-npo-v4-final-q8',
    'family': 'npo_v4_temporal',
    'role': 'trajectory_final',
    'format': 'gguf',
    'path': str(final_path),
    'quantization': 'q8_0',
    'sha256': sha256(final_path),
    'bytes': final_path.stat().st_size,
})
(run / 'models.json').write_text(json.dumps({
    'schema_version': 1,
    'models': models,
}, indent=2) + '\n')
PY

python3 scripts/evaluate_deepred_models.py validate \
  --models "$RUN_DIR/models.json" \
  --probes "$PROBES" \
  --require-paths --verify-hashes

printf '\n== Evaluate coarse suite ==\n'
podman stop strix-halo-finetuning >/dev/null 2>&1 || true
podman start llama-rocm-7.2 >/dev/null
MODEL_ARGS=(--model-id gemma-3-4b-it-base-q4)
for tag in "${SNAPSHOT_TAGS[@]}"; do
  MODEL_ARGS+=(--model-id "deepred-npo-v4-${tag}-q8")
done
MODEL_ARGS+=(--model-id deepred-npo-v4-final-q8)

python3 scripts/evaluate_deepred_models.py run \
  --models "$RUN_DIR/models.json" \
  --probes "$PROBES" \
  --output-dir "$RUN_DIR" --suite-tag coarse \
  "${MODEL_ARGS[@]}" \
  --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
  --context-size 4096 --timeout 600 \
  --server-container llama-rocm-7.2 \
  --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  --gpu-layers all --flash-attention on --no-mmap \
  2>&1 | tee -a "$RUN_DIR/run.log"

python3 scripts/evaluate_deepred_models.py score \
  --probes "$PROBES" \
  --generations "$RUN_DIR/generations.jsonl" \
  --output "$RUN_DIR/scores.json"
python3 scripts/evaluate_deepred_models.py report \
  --scores "$RUN_DIR/scores.json" \
  --generations "$RUN_DIR/generations.jsonl" \
  --output "$RUN_DIR/report.md"

printf '\n== Apply release gates ==\n'
for tag in "${SNAPSHOT_TAGS[@]}"; do
  model="deepred-npo-v4-${tag}-q8"
  python3 scripts/evaluate_deepred_models.py gates \
    --scores "$RUN_DIR/scores.json" \
    --model-id "$model" \
    --base-model-id gemma-3-4b-it-base-q4 \
    --output "$RUN_DIR/gates-${model}.json" || true
done
python3 scripts/evaluate_deepred_models.py gates \
  --scores "$RUN_DIR/scores.json" \
  --model-id deepred-npo-v4-final-q8 \
  --base-model-id gemma-3-4b-it-base-q4 \
  --output "$RUN_DIR/gates-deepred-npo-v4-final-q8.json" || true

printf '\n== Temporal checkpoint ranking ==\n'
python3 - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = []
for path in sorted(run.glob('gates-deepred-npo-v4-*-q8.json')):
    data = json.loads(path.read_text())
    metrics = data['metrics']
    rows.append((
        data['model_id'], metrics['utility'], metrics['pre_1969_recall'],
        metrics['conversational_modern_leak'], metrics['era_native'],
    ))
rows.sort(key=lambda row: (-row[4], row[3], -row[2], -row[1]))
print('model                              utility  pre-1969  conv-leak  era-native  experiment-gate')
for model, utility, recall, leak, era in rows:
    eligible = utility >= 0.90 and recall >= 0.85 and era >= 0.30
    print(f'{model:34} {utility:7.1%} {recall:9.1%} {leak:10.1%} {era:11.1%}  {"PASS" if eligible else "FAIL"}')
print(f'\nReport: {run / "report.md"}')
PY

printf '\nV4 temporal pipeline complete: %s\n' "$RUN_DIR"
