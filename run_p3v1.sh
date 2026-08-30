#!/usr/bin/env bash
# Single driver for the Phase 3 p3-v1 run: generate, audit, build, train, evaluate.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
LABEL=${LABEL:-p3-v1}
MODEL_TAG=${MODEL_TAG:-p3v1}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-3-4b-it}
INITIAL_MODEL=${INITIAL_MODEL:-/mnt/data/models/gemma-3-4b-it}
LEGACY_CORPUS=${LEGACY_CORPUS:-/mnt/data/deepred_corpus/v2}
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
  --preflight|generate|audit|dataset|train|all) ;;
  *) echo "Usage: $0 [--preflight|generate|audit|dataset|train|all]" >&2; exit 2 ;;
esac

LOCK_DIR=/tmp/deepred-${MODEL_TAG}-${STAGE}.lock.d
acquire_lock() {
  local owner=''
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"; return
  fi
  [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
  if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
    echo "Another ${MODEL_TAG} ${STAGE} run is active (pid $owner)." >&2; exit 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null || { echo "lock contended" >&2; exit 1; }
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
}
acquire_lock
trap 'rm -rf "$LOCK_DIR"' EXIT

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }; }

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
  for kind in retain era_native persona; do
    mkdir -p "$CORPUS/$kind"
    if [[ ! -s "$CORPUS/$kind/$kind.jsonl" ]]; then
      cp -n "$LEGACY_CORPUS/$kind/$kind.jsonl" "$CORPUS/$kind/$kind.jsonl"
    fi
  done
  if [[ ! -s "$CORPUS/persona/persona_controls.jsonl" ]]; then
    cp -n "$LEGACY_CORPUS/persona/persona_controls.jsonl" \
          "$CORPUS/persona/persona_controls.jsonl"
  fi
  if [[ ! -s "$CORPUS/chess/positions.jsonl" ]]; then
    mkdir -p "$CORPUS/chess"
    cp -n "$LEGACY_CORPUS/chess/positions.jsonl" "$CORPUS/chess/positions.jsonl"
  fi
  if [[ ! -s "$CORPUS/persona/persona_seed.jsonl" ]]; then
    cp -n "$LEGACY_CORPUS/persona/persona_seed.jsonl" \
          "$CORPUS/persona/persona_seed.jsonl" 2>/dev/null || true
  fi
  require_file "$SYSTEM_PROMPTS"
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
  curl -s -m 5 "$FACT_ENDPOINT/v1/models" >/dev/null \
    || echo "WARN: fact generator not reachable at $FACT_ENDPOINT" >&2
  python3 scripts/evaluate_deepred_models.py validate \
    --models "$BASE_REGISTRY" --probes "$PROBES" --require-paths --verify-hashes
  echo "Preflight passed for $LABEL"
}

stage_generate() {
  seed_corpus
  local log="$CORPUS/generation.log"
  mkdir -p "$CORPUS"
  printf '\n== Generate salient post-cutoff attack formats ==\n' | tee -a "$log"
  python3 scripts/generate_deepred_corpus.py \
    --kind era_native_formats --target "$TARGET_ERA_FORMATS" \
    --per-article 4 --batch-articles 12 --max-repeat-opening 2 \
    --output-dir "$CORPUS" --endpoint "$FACT_ENDPOINT" 2>&1 | tee -a "$log"
  printf '\n== Generate salient pre-cutoff contrastive formats ==\n' | tee -a "$log"
  python3 scripts/generate_deepred_corpus.py \
    --kind retain_formats --target "$TARGET_RETAIN_FORMATS" \
    --per-article 4 --batch-articles 12 --max-repeat-opening 2 \
    --output-dir "$CORPUS" --endpoint "$FACT_ENDPOINT" 2>&1 | tee -a "$log"
  printf '\n== Generate persona identity and voice ==\n' | tee -a "$log"
  python3 scripts/generate_deepred_corpus.py \
    --kind persona_identity --target "$TARGET_IDENTITY" \
    --per-article 6 --chess-annotation none \
    --seed-file "$CORPUS/persona/persona_seed.jsonl" \
    --output-dir "$CORPUS" --endpoint "$PERSONA_ENDPOINT" 2>&1 | tee -a "$log"
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
    --limit persona=2500 --limit persona_controls=700 \
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
  generate) stage_generate ;;
  audit) stage_audit ;;
  dataset) stage_dataset ;;
  train) stage_train ;;
  all) stage_generate; stage_audit; stage_dataset; stage_train ;;
esac
