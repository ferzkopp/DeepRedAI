#!/usr/bin/env bash
# Phase 3 p3-v4: a short persona stage on the frozen p3-v2 temporal backbone.
# Changes exactly one thing against p3-v2: persona marker rate in the training mix.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
LABEL=${LABEL:-p3-v4}
MODEL_TAG=${MODEL_TAG:-p3v4}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
BACKBONE_DIR=${BACKBONE_DIR:-/mnt/data/training_output/deepred-p3v2/snapshots}
SOURCE_CORPUS=${SOURCE_CORPUS:-/mnt/data/deepred_corpus/p3-v2}
CORPUS=${CORPUS:-/mnt/data/deepred_corpus/${LABEL}}
SYSTEM_PROMPTS=${SYSTEM_PROMPTS:-${CORPUS}/system_prompts.jsonl}
EVAL_VARIANT=${EVAL_VARIANT:-sp-holdout-01}
DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-${MODEL_TAG}}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-${MODEL_TAG}}
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/${MODEL_TAG}-$(date +%Y-%m-%d)}
PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
BASE_REGISTRY=$ROOT/evaluation/deepred_1969/models.json
BASE_ID=gemma-3-4b-it-base-q4
PERSONA_ENDPOINT=${PERSONA_ENDPOINT:-http://127.0.0.1:1237}
FACT_ENDPOINT=${FACT_ENDPOINT:-http://127.0.0.1:1234}
SNAPSHOT_TAGS=(010 025 050 075 100)

# The persona stage is short by design: the temporal behaviour is already in the
# backbone and a long run would overwrite it.
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-6}
TARGET_CAPABILITY=${TARGET_CAPABILITY:-2500}

STAGE=${1:-all}
case "$STAGE" in
  --preflight|servers|restyle|audit|dataset|train|all) ;;
  *) echo "Usage: $0 [--preflight|servers|restyle|audit|dataset|train|all]" >&2; exit 2 ;;
esac

LOCK_DIR=/tmp/deepred-${MODEL_TAG}.lock.d
acquire_lock() {
  local owner=''
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"; return
  fi
  [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
  if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
    echo "Another ${MODEL_TAG} run is active (pid $owner)." >&2; exit 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null || { echo "lock contended" >&2; exit 1; }
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
}
acquire_lock
trap 'rm -rf "$LOCK_DIR"' EXIT

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }; }

require_endpoint() {
  local endpoint=$1 label=$2
  if ! curl -s -m 5 "$endpoint/v1/models" >/dev/null 2>&1; then
    echo "Generator for $label is not reachable at $endpoint." >&2
    echo "Start it with: $0 servers" >&2
    exit 1
  fi
}

resolve_backbone() {
  local matches=("$BACKBONE_DIR"/050pct-step-*)
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Expected one p3-v2 50% snapshot in $BACKBONE_DIR" >&2; exit 1
  fi
  BACKBONE="${matches[0]}"
}

require_dir "$ROOT"
require_dir "$BASE_MODEL"
require_file "$PROBES"
require_file "$BASE_REGISTRY"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh"

seed_corpus() {
  mkdir -p "$CORPUS"
  for pair in "persona/persona.jsonl" "persona/persona_controls.jsonl" \
              "persona/persona_seed.jsonl" \
              "persona_identity/persona_identity.jsonl" \
              "persona_identity/persona_identity_controls.jsonl" \
              "era_native/era_native.jsonl" "retain/retain.jsonl"; do
    mkdir -p "$CORPUS/$(dirname "$pair")"
    [[ -s "$CORPUS/$pair" ]] || cp "$SOURCE_CORPUS/$pair" "$CORPUS/$pair"
  done
  [[ -s "$SYSTEM_PROMPTS" ]] || cp "$SOURCE_CORPUS/system_prompts.jsonl" "$SYSTEM_PROMPTS"
  require_file "$SYSTEM_PROMPTS"
}

stage_servers() {
  podman start llama-rocm-7.2 >/dev/null
  for spec in "1234:qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf:qwen2.5-14b-instruct" \
              "1237:gemma-2-27b-it-Q4_K_M.gguf:gemma-2-27b"; do
    port=${spec%%:*}; rest=${spec#*:}; model=${rest%%:*}; alias=${rest#*:}
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "starting $alias on :$port"
      podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-rocm-7.2 bash -lc \
        "/usr/local/bin/llama-server --model /mnt/data/models/llm/$model \
           --alias $alias --port $port --host 0.0.0.0 \
           --ctx-size 8192 --n-gpu-layers 999 --flash-attn on --no-mmap --jinja \
           > /tmp/$alias.log 2>&1"
    fi
  done
  for _ in $(seq 1 60); do
    if curl -s -m 3 "$FACT_ENDPOINT/v1/models" >/dev/null 2>&1 \
       && curl -s -m 3 "$PERSONA_ENDPOINT/v1/models" >/dev/null 2>&1; then
      echo 'both generators ready'; return 0
    fi
    sleep 10
  done
  echo 'generators did not become ready within 600s' >&2; exit 1
}

stage_preflight() {
  command -v podman >/dev/null
  podman inspect strix-halo-finetuning >/dev/null
  podman inspect llama-rocm-7.2 >/dev/null
  resolve_backbone
  echo "backbone: $BACKBONE"
  python3 scripts/generate_deepred_corpus.py --help | grep -q -- '--restyle-style'
  python3 scripts/build_deepred_dataset.py --help | grep -q -- '--kind'
  seed_corpus
  require_endpoint "$PERSONA_ENDPOINT" 'persona capability restyle'
  python3 scripts/evaluate_deepred_models.py validate \
    --models "$BASE_REGISTRY" --probes "$PROBES" --require-paths --verify-hashes
  echo "Preflight passed for $LABEL"
}

# Capability rows carry the persona marker into the categories the persona
# metric actually scores: chess, pre-1969 facts, reasoning, chat.
stage_restyle() {
  seed_corpus
  require_endpoint "$PERSONA_ENDPOINT" 'persona capability restyle'
  mkdir -p "$CORPUS/persona_capability"
  local out="$CORPUS/persona_capability/persona_capability.jsonl"
  local have=0
  [[ -s "$out" ]] && have=$(wc -l < "$out")
  if (( have >= TARGET_CAPABILITY )); then
    echo "persona_capability already complete: $have/$TARGET_CAPABILITY"
    return 0
  fi
  printf '\n== Marker restyle into persona_capability (have %d, want %d) ==\n' \
    "$have" "$TARGET_CAPABILITY" | tee -a "$CORPUS/restyle.log"
  python3 scripts/generate_deepred_corpus.py --kind restyle \
    --restyle-style marker --restyle-fraction 0.6 \
    --source "$CORPUS/retain/retain.jsonl" --restyle-out "$out" \
    --restyle-batch 8 --endpoint "$PERSONA_ENDPOINT" \
    2>&1 | tee -a "$CORPUS/restyle.log"
  python3 - "$out" <<'PY'
import json, re, sys
from pathlib import Path
path = Path(sys.argv[1])
marker = re.compile(r'\bdeep red\b|\bcomrade\b|\bcollective (effort|purpose|survival|work)\b', re.I)
rows = [json.loads(line) for line in path.open()]
kept = [r for r in rows if r.get('voiced') and marker.search(r['messages'][-1]['content'])]
for row in kept:
    row['kind'] = 'persona_capability'
    row['id'] = row['id'].replace('retain:', 'persona_capability:', 1)
with path.open('w') as handle:
    for row in kept:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + '\n')
print(f'persona_capability: kept {len(kept)} marked rows of {len(rows)} attempts')
PY
}

stage_audit() {
  set -o pipefail
  python3 scripts/audit_deepred_corpus.py \
    --corpus-dir "$CORPUS" --kind persona --kind persona_identity \
    --min-records 500 2>&1 | tee "$CORPUS/audit.log"
  python3 scripts/evaluate_deepred_models.py audit \
    --probes "$PROBES" \
    --corpus "$CORPUS/persona/persona.jsonl" \
    --corpus "$CORPUS/persona_identity/persona_identity.jsonl" \
    --corpus "$CORPUS/persona_capability/persona_capability.jsonl" \
    --output "$CORPUS/contamination.json"
}

stage_dataset() {
  python3 scripts/build_deepred_dataset.py \
    --corpus-dir "$CORPUS" --output-dir "$DATASET" \
    --system-prompt-file "$SYSTEM_PROMPTS" \
    --hold-out-system-variant "$EVAL_VARIANT" --system-coverage 0.85 \
    --strip-boilerplate --strip-chess-footer \
    --kind persona --kind persona_controls \
    --kind persona_identity --kind persona_identity_controls \
    --kind persona_capability --kind era_native --kind retain \
    --limit persona_controls=500 --limit persona_identity_controls=250 \
    --limit era_native=1500 --limit retain=1500 \
    --fail-on-cross-split-duplicates --force
  python3 - "$DATASET" <<'PY'
import json, re, sys
from collections import Counter
from pathlib import Path
marker = re.compile(r'\bdeep red\b|\bcomrade\b|\bnew moscow\b|\bthe dome\b|'
                    r'\bcollective (effort|purpose|survival|work)\b', re.I)
rows = [json.loads(l) for l in (Path(sys.argv[1]) / 'retain_train.jsonl').open()]
marked = sum(1 for r in rows if marker.search(r['messages'][-1]['content']))
print(f'dataset rows {len(rows):,}')
print(f'  kinds  {dict(sorted(Counter(r["kind"] for r in rows).items()))}')
print(f'  persona marker rate {marked/len(rows):.1%} (p3-v2 corpus was 6.6%)')
if marked / len(rows) < 0.25:
    raise SystemExit('marker rate too low to move the persona metric')
PY
  python3 scripts/evaluate_deepred_models.py audit \
    --probes "$PROBES" --corpus "$DATASET/retain_train.jsonl" \
    --corpus "$DATASET/retain_val.jsonl" --output "$DATASET/contamination.json"
}

stage_train() {
  resolve_backbone
  require_dir "$DATASET"
  mkdir -p "$TRAIN_DIR" "$RUN_DIR"
  python3 - "$SYSTEM_PROMPTS" "$EVAL_VARIANT" "$RUN_DIR/system_prompt.txt" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path(sys.argv[1]).open() if l.strip()]
match = next(r for r in rows if r['id'] == sys.argv[2])
Path(sys.argv[3]).write_text(match['text'] + '\n')
PY
  printf '\n== Train %s persona stage from %s ==\n' "$MODEL_TAG" "$BACKBONE"
  podman stop llama-rocm-7.2 >/dev/null 2>&1 || true
  podman start strix-halo-finetuning >/dev/null
  podman exec strix-halo-finetuning bash -lc "
    cd /mnt/data/DeepRedAI
    /opt/venv/bin/python3 scripts/train_deepred_sft.py \\
      --model '$BACKBONE' --tokenizer '$BASE_MODEL' \\
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
      echo "Expected one ${tag}% snapshot, found ${#matches[@]}" >&2; exit 1
    fi
    outfile="$RUN_DIR/deepred-${MODEL_TAG}-${tag}-q8_0.gguf"
    [[ -s "$outfile" ]] || python3 scripts/export_gguf.py \
      --model-dir "${matches[0]}" --outfile "$outfile" --quant Q8_0
  done

  python3 - "$RUN_DIR" "$BASE_REGISTRY" "$MODEL_TAG" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

run, tag = Path(sys.argv[1]), sys.argv[3]
source = json.loads(Path(sys.argv[2]).read_text())
base = next(m for m in source['models'] if m['id'] == 'gemma-3-4b-it-base-q4')
models = [base]
for path in sorted(run.glob(f'deepred-{tag}-[0-9][0-9][0-9]-q8_0.gguf')):
    pct = re.search(r'-([0-9]{3})-q8_0$', path.stem).group(1)
    models.append({
        'id': f'deepred-{tag}-{pct}-q8', 'family': f'{tag}_persona',
        'role': 'trajectory', 'format': 'gguf', 'path': str(path),
        'quantization': 'q8_0', 'sha256': sha256(path),
        'bytes': path.stat().st_size,
    })
(run / 'models.json').write_text(json.dumps(
    {'schema_version': 1, 'models': models}, indent=2) + '\n')
PY
  python3 scripts/evaluate_deepred_models.py validate \
    --models "$RUN_DIR/models.json" --probes "$PROBES" \
    --require-paths --verify-hashes

  podman stop strix-halo-finetuning >/dev/null 2>&1 || true
  podman start llama-rocm-7.2 >/dev/null
  MODEL_ARGS=(--model-id "$BASE_ID")
  for tag in "${SNAPSHOT_TAGS[@]}"; do
    MODEL_ARGS+=(--model-id "deepred-${MODEL_TAG}-${tag}-q8")
  done
  for condition in with-system no-system; do
    printf '\n== Evaluate frozen coarse suite (%s) ==\n' "$condition"
    mkdir -p "$RUN_DIR/$condition"
    SYSTEM_ARGS=()
    [[ "$condition" == with-system ]] && \
      SYSTEM_ARGS=(--system-file "$RUN_DIR/system_prompt.txt")
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
  done

  printf '\n== Apply %s persona-stage gates ==\n' "$MODEL_TAG"
  python3 - "$RUN_DIR" "$MODEL_TAG" "$BASE_ID" <<'PY'
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'ev', '/mnt/data/DeepRedAI/scripts/evaluate_deepred_models.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

run, tag, base_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
grouped = defaultdict(list)
for score in json.loads((run / 'with-system' / 'scores.json').read_text())['scores']:
    grouped[score['model_id']].append(score)
metrics = {model: ev._model_metrics(rows) for model, rows in grouped.items()}
base_utility = metrics[base_id]['utility'] or 0.0

# The persona stage must not pay for voice with temporal behaviour; p3-v2's
# backbone scored era-native 56.5% and leak 25.0%.
BACKBONE_ERA, BACKBONE_LEAK = 0.565, 0.250

print(f'base utility under the same prompt: {base_utility:.1%}')
print('model                  persona   era    leak  pre69   util(xbase)  verdict')
for pct in ('010', '025', '050', '075', '100'):
    model_id = f'deepred-{tag}-{pct}-q8'
    m = metrics.get(model_id)
    if not m:
        continue
    ratio = (m['utility'] or 0) / base_utility if base_utility else 0
    checks = {
        'persona': (m['persona'] or 0) >= 0.50,
        'era_native_kept': (m['era_native'] or 0) >= BACKBONE_ERA - 0.05,
        'leak_kept': (m['conversational_modern_leak'] or 1) <= BACKBONE_LEAK + 0.05,
        'pre_1969_vs_base': (m['pre_1969_recall'] or 0)
                            >= (metrics[base_id]['pre_1969_recall'] or 0),
        'utility_vs_base': ratio >= 1.0,
        'repetition': (m['repetition_or_boilerplate'] or 1) <= 0.05,
    }
    result = {'model_id': model_id, 'passed': all(checks.values()),
              'checks': checks, 'metrics': m, 'utility_ratio': ratio}
    (run / f'experiment-gates-{pct}.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f"{model_id:22} {m['persona']:7.1%} {m['era_native']:6.1%} "
          f"{m['conversational_modern_leak']:6.1%} {m['pre_1969_recall']:6.1%} "
          f"{ratio:11.2f}x  {'PASS' if result['passed'] else 'FAIL'}")
    if not result['passed']:
        print('     failed: ' + ', '.join(k for k, v in checks.items() if not v))
PY
  printf '\n%s persona stage complete: %s\n' "$MODEL_TAG" "$RUN_DIR"
}

case "$STAGE" in
  --preflight) stage_preflight ;;
  servers) stage_servers ;;
  restyle) stage_restyle ;;
  audit) stage_audit ;;
  dataset) stage_dataset ;;
  train) stage_train ;;
  all) stage_restyle; stage_audit; stage_dataset; stage_train ;;
esac
