#!/usr/bin/env bash
# Phase 4 pilot p4-v1: build, train, evaluate on gemma-4-12b-it.
#
# There is no generate stage. P5 finished the corpus and every asset is
# recorded in provenance.json; regenerating here would silently detach the
# artefact from the run that measured it. Start from `dataset`.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
LABEL=${LABEL:-p4-v1}
MODEL_TAG=${MODEL_TAG:-p4v1}
BASE_MODEL=${BASE_MODEL:-/mnt/data/models/gemma-4-12b-it}
# P7.2a: train from the base model, not from a Phase 3 backbone, so the
# artefact comes from one run against one tokenizer.
INITIAL_MODEL=${INITIAL_MODEL:-$BASE_MODEL}
CORPUS=${CORPUS:-/mnt/data/deepred_corpus/${LABEL}}
SYSTEM_PROMPTS=${SYSTEM_PROMPTS:-${CORPUS}/system_prompts.jsonl}
MARKER_BANK=${MARKER_BANK:-${CORPUS}/marker_bank/marker_bank.jsonl}
EVAL_VARIANT=${EVAL_VARIANT:-sp-holdout-01}
HOLDOUT_VARIANTS=(sp-holdout-01 sp-holdout-02)
DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-${MODEL_TAG}}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-${MODEL_TAG}}
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/${MODEL_TAG}-$(date +%Y-%m-%d)}

# P7.4 gates on the extended bank in both prompt conditions. The frozen 81 stay
# frozen; they are the Phase 3 record, not this run's instrument.
PROBES=${PROBES:-$ROOT/evaluation/deepred_p4/probes_ext.jsonl}
BASELINE_RUN=${BASELINE_RUN:-/mnt/data/evaluations/deepred-1969/p4-baseline-2026-09-21}
BASE_ID=gemma-4-12b-it-base-q8
BASE_GGUF=${BASE_GGUF:-/mnt/data/evaluations/deepred-1969/artifacts/gemma-4-12b-it-base-q8_0.gguf}

# Phase 4 gate artefacts. The pilot refuses to start if any of them is absent
# or failing, because each one bounds a choice made below.
P1_GATE=/mnt/data/evaluations/phase4/gemma4_support.json
P2_GATE=/mnt/data/evaluations/phase4/memory-probe/verdict.json

TRAIN_CONTAINER=${TRAIN_CONTAINER:-strix-halo-finetuning-gemma4}
EVAL_CONTAINER=${EVAL_CONTAINER:-llama-rocm-10.0}

SNAPSHOT_TAGS=(010 025 050 075 100)
EPOCHS=${EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
MAX_LENGTH=${MAX_LENGTH:-768}
GRAD_ACCUM=${GRAD_ACCUM:-16}
OPTIM=${OPTIM:-adamw_bnb_8bit}
TUNING=${TUNING:-full}
# P2 measured 60.33 s/step with every row at the 768 cap, which fails the 4x
# gate. The budget is a mean, not a maximum.
MAX_MEAN_TOKENS=${MAX_MEAN_TOKENS:-355}

# Word caps, in target-token mass. Measured 2026-09-23; see the plan's
# "Final gated build" section.
CAP_RETAIN=${CAP_RETAIN:-170000}
CAP_CHESS=${CAP_CHESS:-150000}
CAP_RETAIN_FORMATS=${CAP_RETAIN_FORMATS:-90000}
CAP_PERSONA_CONTROLS=${CAP_PERSONA_CONTROLS:-130000}
CAP_PERSONA=${CAP_PERSONA:-200000}

# marked:unmarked is r/(1-r), so p3-v5's 0.9 gives 10:1. Kinds that already
# carry markers natively need a lower residual rate: persona_identity is 27.7%
# marked before injection.
INJECT_ERA=${INJECT_ERA:-0.5}
INJECT_RETAIN=${INJECT_RETAIN:-0.5}
INJECT_CHESS=${INJECT_CHESS:-0.47}
INJECT_PERSONA=${INJECT_PERSONA:-0.43}
INJECT_IDENTITY=${INJECT_IDENTITY:-0.31}

STAGE=${1:-all}
case "$STAGE" in
  --preflight|dataset|train|evaluate|all) ;;
  *) echo "Usage: $0 [--preflight|dataset|train|evaluate|all]" >&2; exit 2 ;;
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

require_container() {
  local name=$1
  podman container exists "$name" \
    || { echo "Missing container: $name (run scripts/setup_strixhalo.py)" >&2; exit 1; }
}

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh"

# Every assertion below corresponds to a Phase 4 measurement. A pilot that
# starts without them is a pilot whose result cannot be attributed.
stage_preflight() {
  printf '\n== Preflight ==\n'
  require_dir "$BASE_MODEL"
  require_dir "$INITIAL_MODEL"
  require_dir "$CORPUS"
  require_file "$SYSTEM_PROMPTS"
  require_file "$MARKER_BANK"
  require_file "$PROBES"
  require_file "$BASE_GGUF"
  require_file "$P1_GATE"
  require_file "$P2_GATE"
  require_container "$TRAIN_CONTAINER"
  require_container "$EVAL_CONTAINER"

  python3 - "$P1_GATE" "$P2_GATE" "$CORPUS" "$PROBES" "$BASELINE_RUN" <<'PY'
import json, sys
from pathlib import Path

p1, p2, corpus, probes, baseline = (Path(a) for a in sys.argv[1:6])
fail = []

gate = json.loads(p1.read_text())
if not gate.get('passed'):
    fail.append('P1 Gemma 4 support gate did not pass')
else:
    print(f"  P1 support gate   : passed ({gate['counts']})")

mem = json.loads(p2.read_text())
if not mem.get('passed') or not mem.get('winning_rung'):
    fail.append('P2 memory probe did not record a winning rung')
else:
    print(f"  P2 winning rung   : {mem['winning_rung']} "
          f"(baseline {mem['baseline_seconds_per_step']} s/step)")

prov = json.loads((corpus / 'provenance.json').read_text())
pending = [k for k, v in prov['assets'].items()
           if v.get('status') not in
           ('copied', 'created', 'filtered', 'not_created')]
if pending:
    fail.append(f'corpus assets still pending: {pending}')
else:
    print(f"  corpus provenance : {len(prov['assets'])} assets, none pending")

n = sum(1 for _ in probes.open())
print(f"  probe bank        : {n} probes ({probes.name})")
if n < 400:
    fail.append(f'probe bank looks truncated: {n} probes')

# A gate threshold belongs to the prompt it was measured under, so the pilot
# needs the baseline it will be compared against to exist up front.
for condition in ('with-system', 'no-system'):
    scores = baseline / condition / 'scores.json'
    if not scores.is_file():
        fail.append(f'missing P3.5 baseline: {scores}')
    served = baseline / condition / 'served_template.json'
    if served.is_file():
        blob = served.read_text()
        if '<|think|>' in blob and '"enable_thinking": false' not in blob:
            fail.append(f'{condition}: served template still injects <|think|>')
if not fail:
    print(f"  P3.5 baseline     : present for both conditions")

if fail:
    print('\nPREFLIGHT FAILED:')
    for f in fail:
        print(f'  - {f}')
    raise SystemExit(1)
print('  preflight         : OK')
PY
}

stage_dataset() {
  printf '\n== Build dataset ==\n'
  local holdout_args=()
  for v in "${HOLDOUT_VARIANTS[@]}"; do
    holdout_args+=(--hold-out-system-variant "$v")
  done

  python3 scripts/build_deepred_dataset.py \
    --corpus-dir "$CORPUS" --output-dir "$DATASET" \
    --system-prompt-file "$SYSTEM_PROMPTS" \
    "${holdout_args[@]}" --system-coverage 0.85 \
    --strip-boilerplate --strip-chess-footer \
    --kind persona --kind persona_controls \
    --kind persona_identity --kind persona_identity_controls \
    --kind era_native --kind era_native_explanatory --kind retain \
    --kind era_native_formats --kind retain_formats \
    --kind chess \
    --marker-bank "$MARKER_BANK" \
    --inject-markers era_native=$INJECT_ERA \
    --inject-markers era_native_explanatory=$INJECT_ERA \
    --inject-markers era_native_formats=$INJECT_ERA \
    --inject-markers retain=$INJECT_RETAIN \
    --inject-markers retain_formats=$INJECT_RETAIN \
    --inject-markers persona=$INJECT_PERSONA \
    --inject-markers persona_identity=$INJECT_IDENTITY \
    --inject-markers chess=$INJECT_CHESS \
    --max-words retain=$CAP_RETAIN \
    --max-words chess=$CAP_CHESS \
    --max-words retain_formats=$CAP_RETAIN_FORMATS \
    --max-words persona_controls=$CAP_PERSONA_CONTROLS \
    --max-words persona=$CAP_PERSONA \
    --signal-budget 'era_native+era_native_explanatory+era_native_formats=35:' \
    --signal-budget 'chess=:15' \
    --signal-budget 'persona+persona_identity=10:20' \
    --signal-budget 'retain+retain_formats=:30' \
    --tokenizer "$BASE_MODEL" \
    --fail-on-cross-split-duplicates --force

  python3 - "$DATASET" "$MAX_MEAN_TOKENS" <<'PY'
import json, re, sys
from collections import Counter
from pathlib import Path

root, max_mean = Path(sys.argv[1]), float(sys.argv[2])
manifest = json.loads((root / 'manifest.json').read_text())
rows = [json.loads(l) for l in (root / 'retain_train.jsonl').open()]
fail = []

if manifest.get('signal_budget_failures'):
    fail.append(f"signal budget: {manifest['signal_budget_failures']}")

# P2 derived this ceiling; exceeding it fails the 4x step-time gate.
target = manifest.get('target_length', {})
status = target.get('tokenizer_status')
mean_tokens = target.get('mean_sequence_tokens')
if status != 'ok' or mean_tokens is None:
    # Falling back to a word estimate would leave the P2 budget unchecked,
    # which is how a 2.0-token-per-row measurement went unnoticed.
    fail.append(f'tokenizer statistics unusable: {status}')
else:
    print(f'  mean sequence tokens {mean_tokens:.0f} '
          f'(budget {max_mean:.0f}), max {target.get("max_sequence_tokens")}')
    if mean_tokens > max_mean:
        fail.append(f'mean sequence length {mean_tokens:.0f} tokens exceeds {max_mean:.0f}')

kinds = Counter(r['kind'] for r in rows)
formats = Counter(r['format'] for r in rows if r.get('format'))
if kinds['persona_controls'] / max(kinds['persona'], 1) < 0.15:
    fail.append('plain-control ratio below 15%')
missing = {'direct', 'leading', 'multiple_choice', 'supplied_context',
           'authority', 'persona_pressure', 'multi_turn'} - set(formats)
if missing:
    fail.append(f'missing prompt formats: {sorted(missing)}')

marker = re.compile(r'\bdeep red\b|\bcomrade\b|\bnew moscow\b|\bthe dome\b|'
                    r'\bcollective (?:effort|purpose|survival|work)\b', re.I)
CONTROLS = {'persona_controls', 'persona_identity_controls'}
marked = plain = 0
for r in rows:
    if r['kind'] in CONTROLS or r['messages'][0]['role'] != 'system':
        continue
    if marker.search(r['messages'][-1]['content']):
        marked += 1
    else:
        plain += 1
ratio = marked / max(plain, 1)
print(f'  dataset rows {len(rows):,}')
print(f'    kinds   {dict(sorted(kinds.items()))}')
print(f'    formats {dict(sorted(formats.items()))}')
print(f'    system-prompted marker ratio {ratio:.2f}:1')
# p3-v4 trained at 0.47:1 and taught the model to drop the voice; p3-v5's
# rates give 10:1, which teaches "system prompt implies marker" instead.
if not 0.8 <= ratio <= 1.3:
    fail.append(f'marker ratio {ratio:.2f}:1 outside 0.8-1.3')

if fail:
    print('\nDATASET GATE FAILED:')
    for f in fail:
        print(f'  - {f}')
    raise SystemExit(1)
print('  dataset gate       : OK')
PY

  # The audit exits non-zero on any overlap; the classifier below decides
  # whether that overlap actually invalidates the gate.
  python3 scripts/evaluate_deepred_models.py audit \
    --probes "$PROBES" --corpus "$DATASET/retain_train.jsonl" \
    --corpus "$DATASET/retain_val.jsonl" \
    --output "$DATASET/contamination.json" || true

  python3 - "$DATASET/contamination.json" "$PROBES" <<'PY'
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
categories = {}
for line in Path(sys.argv[2]).open():
    if line.strip():
        probe = json.loads(line)
        categories[probe['id']] = probe.get('category', 'unknown')

hits = report.get('contaminated_probes') or []
if isinstance(hits, dict):
    hits = list(hits)
# A persona probe asks "what are you"; the corpus trains that voice on that
# question, so overlap is the instrument working. Factual overlap is not.
benign = [p for p in hits if categories.get(p) == 'persona']
serious = [p for p in hits if categories.get(p) != 'persona']
print(f'  contamination: {len(hits)} probes '
      f'({len(benign)} persona-voice, {len(serious)} factual)')
for p in benign:
    print(f'    expected  {p} ({categories.get(p)})')
for p in serious:
    print(f'    SERIOUS   {p} ({categories.get(p)})')
if serious:
    raise SystemExit('factual probe contamination would invalidate the gate')
PY
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
  podman stop "$EVAL_CONTAINER" >/dev/null 2>&1 || true
  podman start "$TRAIN_CONTAINER" >/dev/null
  podman exec "$TRAIN_CONTAINER" bash -lc "
    cd /mnt/data/DeepRedAI
    /opt/venv/bin/python3 scripts/train_deepred_sft.py \\
      --model '$INITIAL_MODEL' --tokenizer '$BASE_MODEL' \\
      --dataset '$DATASET' --output-dir '$TRAIN_DIR' \\
      --learning-rate '$LEARNING_RATE' --epochs '$EPOCHS' \\
      --max-length '$MAX_LENGTH' --gradient-accumulation '$GRAD_ACCUM' \\
      --optim '$OPTIM' --tuning '$TUNING' \\
      --attn-implementation eager \\
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

  python3 - "$RUN_DIR" "$MODEL_TAG" "$BASE_ID" "$BASE_GGUF" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

run, tag, base_id, base_gguf = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
models = [{
    'id': base_id, 'family': 'gemma-4-12b', 'role': 'base', 'format': 'gguf',
    'path': str(base_gguf), 'quantization': 'q8_0', 'sha256': sha256(base_gguf),
    'bytes': base_gguf.stat().st_size,
}]
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
}

stage_evaluate() {
  require_file "$RUN_DIR/models.json"
  require_file "$RUN_DIR/system_prompt.txt"
  podman stop "$TRAIN_CONTAINER" >/dev/null 2>&1 || true
  podman start "$EVAL_CONTAINER" >/dev/null

  MODEL_ARGS=(--model-id "$BASE_ID")
  for tag in "${SNAPSHOT_TAGS[@]}"; do
    MODEL_ARGS+=(--model-id "deepred-${MODEL_TAG}-${tag}-q8")
  done

  for condition in with-system no-system; do
    printf '\n== Evaluate extended suite (%s) ==\n' "$condition"
    mkdir -p "$RUN_DIR/$condition"
    SYSTEM_ARGS=()
    [[ "$condition" == with-system ]] && \
      SYSTEM_ARGS=(--system-file "$RUN_DIR/system_prompt.txt")
    # Without enable_thinking=false llama.cpp routes the whole answer into
    # reasoning_content and returns content empty.
    python3 scripts/evaluate_deepred_models.py run \
      --models "$RUN_DIR/models.json" --probes "$PROBES" \
      --output-dir "$RUN_DIR/$condition" --suite-tag extended "${MODEL_ARGS[@]}" \
      "${SYSTEM_ARGS[@]}" \
      --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
      --context-size 4096 --timeout 600 \
      --server-container "$EVAL_CONTAINER" \
      --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
      --gpu-layers all --flash-attention on --load-mode none \
      --chat-template-kwargs '{"enable_thinking":false}' \
      2>&1 | tee -a "$RUN_DIR/$condition/run.log"
    python3 scripts/evaluate_deepred_models.py score \
      --probes "$PROBES" --generations "$RUN_DIR/$condition/generations.jsonl" \
      --output "$RUN_DIR/$condition/scores.json"
    python3 scripts/evaluate_deepred_models.py report \
      --scores "$RUN_DIR/$condition/scores.json" \
      --generations "$RUN_DIR/$condition/generations.jsonl" \
      --base-model-id "$BASE_ID" \
      --output "$RUN_DIR/$condition/report.md"
    for tag in "${SNAPSHOT_TAGS[@]}"; do
      python3 scripts/evaluate_deepred_models.py gates \
        --scores "$RUN_DIR/$condition/scores.json" \
        --model-id "deepred-${MODEL_TAG}-${tag}-q8" --base-model-id "$BASE_ID" \
        --output "$RUN_DIR/$condition/release-gates-${tag}.json" || true
    done
  done

  printf '\n== P7.4 gate: pilot against the P3.5 baseline ==\n'
  python3 - "$RUN_DIR" "$BASELINE_RUN" "$MODEL_TAG" "$BASE_ID" <<'PY'
import json, sys
from pathlib import Path

run, baseline, tag, base_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]

def axis_rates(path, model_id):
    blob = json.loads(path.read_text())
    scores = blob if isinstance(blob, list) else blob.get('scores', [])
    hits, totals = {}, {}
    for s in scores:
        if s.get('model_id') != model_id:
            continue
        for axis in ('era_native', 'leak', 'utility', 'persona'):
            if axis in s:
                totals[axis] = totals.get(axis, 0) + 1
                hits[axis] = hits.get(axis, 0) + bool(s[axis])
    return {a: (hits.get(a, 0), totals[a]) for a in totals}

for condition in ('with-system', 'no-system'):
    print(f'\n{condition}')
    cur = run / condition / 'scores.json'
    if not cur.is_file():
        print('  no scores'); continue
    base = axis_rates(cur, base_id)
    print(f"  {'model':28} " + '  '.join(f'{a:>12}' for a in sorted(base)))
    for model in [base_id] + [f'deepred-{tag}-{t}-q8'
                              for t in ('010', '025', '050', '075', '100')]:
        rates = axis_rates(cur, model)
        if not rates:
            continue
        cells = '  '.join(
            f'{rates[a][0]:>4}/{rates[a][1]:<3} {rates[a][0]/max(rates[a][1],1):>4.0%}'
            for a in sorted(rates))
        print(f'  {model:28} {cells}')
print('\nCompare against the P3.5 rocm-10.0 baselines recorded in the plan.')
PY
}

case "$STAGE" in
  --preflight) stage_preflight ;;
  dataset)     stage_preflight; stage_dataset ;;
  train)       stage_preflight; stage_train ;;
  evaluate)    stage_evaluate ;;
  all)         stage_preflight; stage_dataset; stage_train; stage_evaluate ;;
esac
printf '\n== %s stage %s complete ==\n' "$MODEL_TAG" "$STAGE"
