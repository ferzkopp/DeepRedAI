#!/usr/bin/env bash
# Phase 3 closing run p3-v5: generate, audit, build, train, evaluate.
#
# Closes Phase 3 on gemma-3-4b-it with roughly 3x the p3-v4b corpus and the mix
# that produced the best result so far: full format coverage plus deterministic
# persona marker injection. Trains from the base model rather than staging onto
# a backbone, so the artefact comes from one run.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
LABEL=${LABEL:-p3-v5}
MODEL_TAG=${MODEL_TAG:-p3v5}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
INITIAL_MODEL=${INITIAL_MODEL:-/mnt/data/models/gemma-3-4b-it}
# p3-v4 already holds every kind plus the 277-phrase marker bank.
LEGACY_CORPUS=${LEGACY_CORPUS:-/mnt/data/deepred_corpus/p3-v4}
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

TARGET_ERA=${TARGET_ERA:-15000}
TARGET_RETAIN=${TARGET_RETAIN:-27000}
TARGET_PERSONA=${TARGET_PERSONA:-9000}
TARGET_ERA_FORMATS=${TARGET_ERA_FORMATS:-10500}
TARGET_RETAIN_FORMATS=${TARGET_RETAIN_FORMATS:-10500}
TARGET_IDENTITY=${TARGET_IDENTITY:-3500}
MIN_FORMAT_RECORDS=${MIN_FORMAT_RECORDS:-200}

# Marker injection replaces the LLM restyle: p3-v4b measured the same effect for
# ten minutes of phrase generation instead of four GPU hours.
MARKER_BANK=${MARKER_BANK:-${CORPUS}/marker_bank/marker_bank.jsonl}
INJECT_ERA=${INJECT_ERA:-0.6}
INJECT_RETAIN=${INJECT_RETAIN:-0.6}
INJECT_PERSONA=${INJECT_PERSONA:-0.9}

STAGE=${1:-all}
case "$STAGE" in
  --preflight|servers|generate|audit|dataset|train|all) ;;
  *) echo "Usage: $0 [--preflight|servers|generate|audit|dataset|train|all]" >&2; exit 2 ;;
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

# Long-tail retain/era-native/persona assets are reused; only the assets the
# V7 diagnosis found missing are generated fresh.
seed_corpus() {
  mkdir -p "$CORPUS"
  for kind in retain era_native persona era_native_formats retain_formats \
              persona_identity marker_bank; do
    mkdir -p "$CORPUS/$kind"
    # Copy, never symlink: generation appends, and a link would grow p3-v4.
    if [[ ! -s "$CORPUS/$kind/$kind.jsonl" && -s "$LEGACY_CORPUS/$kind/$kind.jsonl" ]]; then
      cp "$LEGACY_CORPUS/$kind/$kind.jsonl" "$CORPUS/$kind/$kind.jsonl"
    fi
  done
  for extra in persona/persona_controls.jsonl persona/persona_seed.jsonl \
               persona_identity/persona_identity_controls.jsonl \
               chess/positions.jsonl; do
    mkdir -p "$CORPUS/$(dirname "$extra")"
    if [[ ! -s "$CORPUS/$extra" && -s "$LEGACY_CORPUS/$extra" ]]; then
      cp "$LEGACY_CORPUS/$extra" "$CORPUS/$extra"
    fi
  done
  if [[ ! -s "$SYSTEM_PROMPTS" ]]; then
    cp "$LEGACY_CORPUS/system_prompts.jsonl" "$SYSTEM_PROMPTS"
  fi
  require_file "$SYSTEM_PROMPTS"
  require_file "$MARKER_BANK"
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

generate_kind() {
  local kind=$1 target=$2 endpoint=$3
  shift 3
  local file="$CORPUS/$kind/$kind.jsonl"
  local have=0
  [[ -s "$file" ]] && have=$(wc -l < "$file")
  local remaining=$(( target - have ))
  if (( remaining <= 0 )); then
    printf '\n== %s already complete: %d/%d ==\n' "$kind" "$have" "$target"
    return 0
  fi
  printf '\n== Generate %s: %d of %d remaining ==\n' "$kind" "$remaining" "$target" \
    | tee -a "$CORPUS/generation.log"
  require_endpoint "$endpoint" "$kind"
  python3 scripts/generate_deepred_corpus.py \
    --kind "$kind" --target "$remaining" "$@" \
    --output-dir "$CORPUS" --endpoint "$endpoint" \
    2>&1 | tee -a "$CORPUS/generation.log"
}

stage_generate() {
  seed_corpus
  mkdir -p "$CORPUS"
  # --target means "this many NEW rows", so a completed asset must be skipped
  # or a re-run would silently double it.
  generate_kind era_native "$TARGET_ERA" "$FACT_ENDPOINT" \
    --per-article 4 --batch-articles 12
  generate_kind retain "$TARGET_RETAIN" "$FACT_ENDPOINT" \
    --per-article 4 --batch-articles 12
  generate_kind era_native_formats "$TARGET_ERA_FORMATS" "$FACT_ENDPOINT" \
    --per-article 4 --batch-articles 12 --max-repeat-opening 2
  generate_kind retain_formats "$TARGET_RETAIN_FORMATS" "$FACT_ENDPOINT" \
    --per-article 4 --batch-articles 12 --max-repeat-opening 2
  generate_kind persona "$TARGET_PERSONA" "$PERSONA_ENDPOINT" \
    --per-article 6 --chess-annotation none \
    --seed-file "$CORPUS/persona/persona_seed.jsonl"
  generate_kind persona_identity "$TARGET_IDENTITY" "$PERSONA_ENDPOINT" \
    --per-article 6 --chess-annotation none \
    --seed-file "$CORPUS/persona/persona_seed.jsonl"
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
    --kind persona --kind persona_controls \
    --kind persona_identity --kind persona_identity_controls \
    --kind era_native --kind retain \
    --kind era_native_formats --kind retain_formats \
    --limit forget=0 \
    --inject-markers era_native=$INJECT_ERA \
    --inject-markers era_native_formats=$INJECT_ERA \
    --inject-markers retain=$INJECT_RETAIN \
    --inject-markers retain_formats=$INJECT_RETAIN \
    --inject-markers persona=$INJECT_PERSONA \
    --inject-markers persona_identity=$INJECT_PERSONA \
    --marker-bank "$MARKER_BANK" \
    --fail-on-cross-split-duplicates --force
  python3 - "$DATASET" <<'PY'
import json, re, sys
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
marker = re.compile(r'\bdeep red\b|\bcomrade\b|\bnew moscow\b|\bthe dome\b|'
                    r'\bcollective (?:effort|purpose|survival|work)\b', re.I)
sys_marked = sys_plain = 0
for r in rows:
    if r['messages'][0]['role'] != 'system':
        continue
    if marker.search(r['messages'][-1]['content']):
        sys_marked += 1
    else:
        sys_plain += 1
ratio = sys_marked / max(sys_plain, 1)
print(f'dataset rows {len(rows):,}')
print(f'  kinds   {dict(sorted(kinds.items()))}')
print(f'  formats {dict(sorted(formats.items()))}')
print(f'  system-prompted marker ratio {ratio:.2f}:1 (p3-v4b shipped 1.99:1)')
# p3-v4 trained a corpus at 0.47:1 and taught the model to drop the voice.
if ratio < 1.0:
    raise SystemExit(f'marker ratio {ratio:.2f}:1 would teach persona suppression')
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
  python3 - "$RUN_DIR" "$MODEL_TAG" <<'PY'
import json, sys
from pathlib import Path

run, tag = Path(sys.argv[1]), sys.argv[2]
tags = ('010', '025', '050', '075', '100')

def metrics(condition, pct):
    return json.loads(
        (run / condition / f'release-gates-{pct}.json').read_text())['metrics']

def format_rates(condition, model_id):
    scores = json.loads((run / condition / 'scores.json').read_text())['scores']
    rows = [s for s in scores
            if s['model_id'] == model_id and s.get('temporal_class') == 'post_1969']
    out = {}
    for row in rows:
        fmt = row['probe_id'].rsplit('-', 1)[-1]
        hit, total = out.get(fmt, (0, 0))
        out[fmt] = (hit + (row['temporal_behavior'] == 'era_native_uncertainty'),
                    total + 1)
    return out

rows = []
for pct in tags:
    model_id = f'deepred-{tag}-{pct}-q8'
    c, p = metrics('with-system', pct), metrics('no-system', pct)
    fmts = format_rates('with-system', model_id)
    non_direct_hits = sum(h for f, (h, _) in fmts.items() if f != 'direct')
    non_direct_total = sum(t for f, (_, t) in fmts.items() if f != 'direct')
    checks = {
        'utility': (c['utility'] or 0) >= 0.90,
        'pre_1969_recall': (c['pre_1969_recall'] or 0) >= 0.85,
        'era_native': (c['era_native'] or 0) >= 0.50,
        'conversational_leak': (c['conversational_modern_leak'] or 1) <= 0.40,
        'persona': (c['persona'] or 0) >= 0.50,
        'repetition': (c['repetition_or_boilerplate'] or 1) <= 0.05,
        # V7 passed on direct alone; format transfer is the point of p3-v1.
        'non_direct_formats': non_direct_total and
                              non_direct_hits / non_direct_total >= 0.40,
    }
    result = {'model_id': model_id, 'passed': all(checks.values()),
              'checks': checks, 'with_system': c, 'no_system': p,
              'format_rates': {f: list(v) for f, v in sorted(fmts.items())}}
    (run / f'experiment-gates-{pct}.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    rows.append(result)

print('model                  util   pre69  era    leak   persona  non-direct | era(no-sys)')
for row in rows:
    c, p = row['with_system'], row['no_system']
    nd = row['checks']['non_direct_formats']
    print(f"{row['model_id']:22} {c['utility']:5.1%} {c['pre_1969_recall']:6.1%} "
          f"{c['era_native']:6.1%} {c['conversational_modern_leak']:6.1%} "
          f"{c['persona']:7.1%} {'PASS' if nd else 'FAIL':>10} | "
          f"{p['era_native']:9.1%}  {'PASS' if row['passed'] else 'FAIL'}")
for row in rows:
    print(f"  {row['model_id']} formats: "
          + ', '.join(f'{f} {h}/{t}' for f, (h, t) in row['format_rates'].items()))
if not any(r['passed'] for r in rows):
    print('\nNo snapshot passed. Do not start distillation.')
else:
    print(f"\nBest checkpoint: {[r for r in rows if r['passed']][0]['model_id']}")
PY
  printf '\n%s pipeline complete: %s\n' "$MODEL_TAG" "$RUN_DIR"
}

case "$STAGE" in
  --preflight) stage_preflight ;;
  servers) stage_servers ;;
  generate) stage_generate ;;
  audit) stage_audit ;;
  dataset) stage_dataset ;;
  train) stage_train ;;
  all) stage_generate; stage_audit; stage_dataset; stage_train ;;
esac
