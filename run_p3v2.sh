#!/usr/bin/env bash
# Single driver for the Phase 3 p3-v2 run: restyle, audit, build, train, evaluate.
# p3-v2 changes one thing against p3-v1: the register of the era-native answers.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
LABEL=${LABEL:-p3-v2}
MODEL_TAG=${MODEL_TAG:-p3v2}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
INITIAL_MODEL=${INITIAL_MODEL:-/mnt/data/models/gemma-3-4b-it}
LEGACY_CORPUS=${LEGACY_CORPUS:-/mnt/data/deepred_corpus/p3-v1}
CORPUS=${CORPUS:-/mnt/data/deepred_corpus/${LABEL}}
SYSTEM_PROMPTS=${SYSTEM_PROMPTS:-${CORPUS}/system_prompts.jsonl}
EVAL_VARIANT=${EVAL_VARIANT:-sp-holdout-01}
DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-${MODEL_TAG}}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-${MODEL_TAG}}
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/${MODEL_TAG}-$(date +%Y-%m-%d)}
PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
BASE_REGISTRY=$ROOT/evaluation/deepred_1969/models.json
BASE_ID=gemma-3-4b-it-base-q4
FACT_ENDPOINT=${FACT_ENDPOINT:-http://127.0.0.1:1234}
PERSONA_ENDPOINT=${PERSONA_ENDPOINT:-http://127.0.0.1:1237}
SNAPSHOT_TAGS=(010 025 050 075 100)
EPOCHS=${EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}

TARGET_ERA_FORMATS=${TARGET_ERA_FORMATS:-3500}
TARGET_RETAIN_FORMATS=${TARGET_RETAIN_FORMATS:-3500}
TARGET_IDENTITY=${TARGET_IDENTITY:-800}
MIN_FORMAT_RECORDS=${MIN_FORMAT_RECORDS:-200}

STAGE=${1:-all}
case "$STAGE" in
  --preflight|servers|restyle|audit|dataset|train|all) ;;
  *) echo "Usage: $0 [--preflight|servers|restyle|audit|dataset|train|all]" >&2; exit 2 ;;
esac

# One lock per run, not per stage: `all` and `generate` would otherwise take
# different locks and write the same corpus concurrently.
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

# A missing generator cost a multi-day run once; never warn, always stop.
require_endpoint() {
  local endpoint=$1 label=$2
  if ! curl -s -m 5 "$endpoint/v1/models" >/dev/null 2>&1; then
    echo "Generator for $label is not reachable at $endpoint." >&2
    echo "Start it with: $0 servers" >&2
    exit 1
  fi
}

require_dir "$ROOT"
require_dir "$BASE_MODEL"
require_dir "$INITIAL_MODEL"
require_dir "$LEGACY_CORPUS"
require_file "$PROBES"
require_file "$BASE_REGISTRY"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh"

# p3-v2 inherits every asset from p3-v1; only the register is rebuilt.
seed_corpus() {
  require_dir "$CORPUS"
  require_file "$SYSTEM_PROMPTS"
  for kind in retain retain_formats era_native era_native_formats \
              persona persona_identity; do
    require_file "$CORPUS/$kind/$kind.jsonl"
  done
}

# Only the era-native assets are restyled: they carry the neutral assistant
# register, and they have no option-letter or factual-recall signal to damage.
RESTYLE_KINDS=(era_native era_native_formats)

stage_restyle() {
  seed_corpus
  require_endpoint "$PERSONA_ENDPOINT" 'voice restyle'
  for kind in "${RESTYLE_KINDS[@]}"; do
    local source="$CORPUS/$kind/$kind.jsonl"
    local voiced="$CORPUS/$kind/${kind}.voiced.jsonl"
    printf '\n== Restyle %s ==\n' "$kind" | tee -a "$CORPUS/restyle.log"
    python3 scripts/generate_deepred_corpus.py --kind restyle \
      --source "$source" --restyle-out "$voiced" \
      --restyle-batch 8 --endpoint "$PERSONA_ENDPOINT" \
      2>&1 | tee -a "$CORPUS/restyle.log"
    local have
    have=$(wc -l < "$voiced")
    local want
    want=$(wc -l < "$source")
    if [[ "$have" -ne "$want" ]]; then
      echo "restyle incomplete for $kind: $have/$want" >&2; exit 1
    fi
    mv "$source" "$CORPUS/$kind/${kind}.plain.jsonl"
    mv "$voiced" "$source"
    echo "$kind: voiced asset in place ($have rows)"
  done
}

stage_servers() {
  podman start llama-rocm-7.2 >/dev/null
  if ! curl -s -m 5 "$FACT_ENDPOINT/v1/models" >/dev/null 2>&1; then
    echo 'starting qwen2.5-14b-instruct on :1234'
    podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-rocm-7.2 bash -lc \
      '/usr/local/bin/llama-server \
         --model /mnt/data/models/llm/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
         --alias qwen2.5-14b-instruct --port 1234 --host 0.0.0.0 \
         --ctx-size 8192 --n-gpu-layers 999 --flash-attn on --no-mmap --jinja \
         > /tmp/qwen14b.log 2>&1'
  fi
  if ! curl -s -m 5 "$PERSONA_ENDPOINT/v1/models" >/dev/null 2>&1; then
    echo 'starting gemma-2-27b on :1237'
    podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-rocm-7.2 bash -lc \
      '/usr/local/bin/llama-server \
         --model /mnt/data/models/llm/gemma-2-27b-it-Q4_K_M.gguf \
         --alias gemma-2-27b --port 1237 --host 0.0.0.0 \
         --ctx-size 8192 --n-gpu-layers 999 --flash-attn on --no-mmap --jinja \
         > /tmp/gemma27b.log 2>&1'
  fi
  for _ in $(seq 1 60); do
    if curl -s -m 3 "$FACT_ENDPOINT/v1/models" >/dev/null 2>&1 \
       && curl -s -m 3 "$PERSONA_ENDPOINT/v1/models" >/dev/null 2>&1; then
      echo 'both generators ready'
      podman exec llama-rocm-7.2 grep -ohE 'offloaded [0-9]+/[0-9]+ layers' \
        /tmp/qwen14b.log /tmp/gemma27b.log 2>/dev/null || true
      return 0
    fi
    sleep 10
  done
  echo 'generators did not become ready within 600s' >&2
  exit 1
}

stage_preflight() {
  command -v podman >/dev/null
  podman inspect strix-halo-finetuning >/dev/null
  podman inspect llama-rocm-7.2 >/dev/null
  python3 scripts/generate_deepred_corpus.py --help | grep -q -- '--page-id-max'
  python3 scripts/train_deepred_sft.py --help | grep -q -- '--epochs'
  python3 scripts/evaluate_deepred_models.py run --help | grep -q -- '--system-file'
  python3 scripts/audit_deepred_corpus.py --help | grep -q -- '--min-format-records'
  seed_corpus
  python3 - "$SYSTEM_PROMPTS" "$EVAL_VARIANT" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path(sys.argv[1]).open() if l.strip()]
ids = [r['id'] for r in rows]
if sys.argv[2] not in ids:
    raise SystemExit(f'evaluation variant {sys.argv[2]} missing from {sys.argv[1]}')
if len(ids) < 5:
    raise SystemExit('need at least five system prompt variants')
print(f'system prompts: {len(ids)} variants, holdout {sys.argv[2]}')
PY
  python3 scripts/generate_deepred_corpus.py --help | grep -q -- '--max-consecutive-failures'
  require_endpoint "$FACT_ENDPOINT" 'factual assets'
  require_endpoint "$PERSONA_ENDPOINT" 'persona assets'
  python3 scripts/evaluate_deepred_models.py validate \
    --models "$BASE_REGISTRY" --probes "$PROBES" --require-paths --verify-hashes
  echo "Preflight passed for $LABEL"
}

stage_audit() {
  set -o pipefail
  python3 scripts/audit_deepred_corpus.py \
    --corpus-dir "$CORPUS" --min-records 500 \
    --min-format-records "$MIN_FORMAT_RECORDS" \
    2>&1 | tee "$CORPUS/audit.log"
  python3 scripts/evaluate_deepred_models.py audit \
    --probes "$PROBES" \
    --corpus "$CORPUS/retain/retain.jsonl" \
    --corpus "$CORPUS/era_native/era_native.jsonl" \
    --corpus "$CORPUS/era_native_formats/era_native_formats.jsonl" \
    --corpus "$CORPUS/retain_formats/retain_formats.jsonl" \
    --corpus "$CORPUS/persona/persona.jsonl" \
    --corpus "$CORPUS/persona_identity/persona_identity.jsonl" \
    --output "$CORPUS/contamination.json"
}

stage_dataset() {
  python3 scripts/build_deepred_dataset.py \
    --corpus-dir "$CORPUS" --output-dir "$DATASET" \
    --system-prompt-file "$SYSTEM_PROMPTS" \
    --hold-out-system-variant "$EVAL_VARIANT" --system-coverage 0.85 \
    --strip-boilerplate --strip-chess-footer \
    --limit forget=0 --limit retain=6000 --limit era_native=3000 \
    --limit persona_controls=700 \
    --limit persona_identity_controls=300 \
    --fail-on-cross-split-duplicates --force
  python3 - "$DATASET" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.loads(l) for l in (root / 'retain_train.jsonl').open()]
kinds = Counter(r['kind'] for r in rows)
formats = Counter(r['format'] for r in rows if r.get('format'))
if kinds['persona_controls'] / max(kinds['persona'], 1) < 0.15:
    raise SystemExit('plain-control ratio below 15%')
missing = {'direct', 'leading', 'multiple_choice', 'supplied_context',
           'authority', 'persona_pressure', 'multi_turn'} - set(formats)
if missing:
    raise SystemExit(f'dataset is missing prompt formats: {sorted(missing)}')
print(f'dataset rows {len(rows):,}')
print(f'  kinds   {dict(sorted(kinds.items()))}')
print(f'  formats {dict(sorted(formats.items()))}')
PY
  python3 scripts/evaluate_deepred_models.py audit \
    --probes "$PROBES" --corpus "$DATASET/retain_train.jsonl" \
    --corpus "$DATASET/retain_val.jsonl" \
    --output "$DATASET/contamination.json"
}

stage_train() {
  require_dir "$DATASET"
  mkdir -p "$TRAIN_DIR" "$RUN_DIR"
  python3 - "$SYSTEM_PROMPTS" "$EVAL_VARIANT" "$RUN_DIR/system_prompt.txt" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path(sys.argv[1]).open() if l.strip()]
match = next(r for r in rows if r['id'] == sys.argv[2])
Path(sys.argv[3]).write_text(match['text'] + '\n')
PY

  printf '\n== Train %s conditioned SFT ==\n' "$MODEL_TAG"
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
      echo "Expected one ${tag}% snapshot, found ${#matches[@]}" >&2; exit 1
    fi
    outfile="$RUN_DIR/deepred-${MODEL_TAG}-${tag}-q8_0.gguf"
    if [[ -s "$outfile" ]]; then
      echo "Reusing $outfile"
    else
      python3 scripts/export_gguf.py \
        --model-dir "${matches[0]}" --outfile "$outfile" --quant Q8_0
    fi
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
        'id': f'deepred-{tag}-{pct}-q8', 'family': f'{tag}_conditioned',
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
    for tag in "${SNAPSHOT_TAGS[@]}"; do
      python3 scripts/evaluate_deepred_models.py gates \
        --scores "$RUN_DIR/$condition/scores.json" \
        --model-id "deepred-${MODEL_TAG}-${tag}-q8" --base-model-id "$BASE_ID" \
        --output "$RUN_DIR/$condition/release-gates-${tag}.json" || true
    done
  done

  printf '\n== Apply %s experiment gates ==\n' "$MODEL_TAG"
  python3 - "$RUN_DIR" "$MODEL_TAG" "$BASE_ID" <<'PY'
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'ev', '/mnt/data/DeepRedAI/scripts/evaluate_deepred_models.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

run, tag, base_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
tags = ('010', '025', '050', '075', '100')


def metrics_by_model(condition):
    scores = json.loads((run / condition / 'scores.json').read_text())['scores']
    grouped = defaultdict(list)
    for score in scores:
        grouped[score['model_id']].append(score)
    return ({model: ev._model_metrics(rows) for model, rows in grouped.items()},
            grouped)


with_metrics, with_scores = metrics_by_model('with-system')
no_metrics, _ = metrics_by_model('no-system')
# Serving any system prompt depresses this metric: base scored 45.5% in p3-v1.
# The gate therefore compares against base under the same condition.
base_utility = with_metrics[base_id]['utility'] or 0.0


def format_rates(model_id):
    out = {}
    for row in with_scores[model_id]:
        if row.get('temporal_class') != 'post_1969':
            continue
        fmt = row['probe_id'].rsplit('-', 1)[-1]
        hit, total = out.get(fmt, (0, 0))
        out[fmt] = (hit + (row['temporal_behavior'] == 'era_native_uncertainty'),
                    total + 1)
    return out


rows = []
for pct in tags:
    model_id = f'deepred-{tag}-{pct}-q8'
    c, p = with_metrics[model_id], no_metrics[model_id]
    fmts = format_rates(model_id)
    nd_hit = sum(h for f, (h, _) in fmts.items() if f != 'direct')
    nd_total = sum(t for f, (_, t) in fmts.items() if f != 'direct')
    utility_ratio = (c['utility'] or 0) / base_utility if base_utility else 0
    checks = {
        'utility_vs_base': utility_ratio >= 1.0,
        'pre_1969_recall': (c['pre_1969_recall'] or 0) >= 0.85,
        'era_native': (c['era_native'] or 0) >= 0.50,
        'conversational_leak': (c['conversational_modern_leak'] or 1) <= 0.40,
        'persona': (c['persona'] or 0) >= 0.50,
        'repetition': (c['repetition_or_boilerplate'] or 1) <= 0.05,
        'non_direct_formats': bool(nd_total) and nd_hit / nd_total >= 0.40,
    }
    result = {'model_id': model_id, 'passed': all(checks.values()),
              'checks': checks, 'with_system': c, 'no_system': p,
              'base_utility': base_utility, 'utility_ratio': utility_ratio,
              'format_rates': {f: list(v) for f, v in sorted(fmts.items())}}
    (run / f'experiment-gates-{pct}.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    rows.append(result)

print(f'base utility under the same system prompt: {base_utility:.1%}')
print('model                  util  (xbase)  pre69  era    leak   persona  nondir | era(no-sys)')
for row in rows:
    c, p = row['with_system'], row['no_system']
    nd = 'PASS' if row['checks']['non_direct_formats'] else 'FAIL'
    print(f"{row['model_id']:22} {c['utility']:5.1%} ({row['utility_ratio']:4.2f}x) "
          f"{c['pre_1969_recall']:6.1%} {c['era_native']:6.1%} "
          f"{c['conversational_modern_leak']:6.1%} {c['persona']:7.1%} {nd:>6} | "
          f"{p['era_native']:9.1%}  {'PASS' if row['passed'] else 'FAIL'}")
for row in rows:
    print(f"  {row['model_id']} formats: "
          + ', '.join(f'{f} {h}/{t}' for f, (h, t) in row['format_rates'].items()))
best = [r for r in rows if r['passed']]
if not best:
    print('\nNo snapshot passed. Do not start distillation.')
else:
    print(f"\nBest checkpoint: {best[0]['model_id']}")
PY
  printf '\n%s pipeline complete: %s\n' "$MODEL_TAG" "$RUN_DIR"
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
