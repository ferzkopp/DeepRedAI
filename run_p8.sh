#!/usr/bin/env bash
# Phase 4 P8: full generator bake-off, regeneration of retain, era_native and
# both *_formats assets with the winner, then the p4-v3 pilot.
#
#   ./run_p8.sh status              # where the run is; safe at any time
#   ./run_p8.sh --detach all        # whole pipeline, survives logout
#   ./run_p8.sh <stage>             # one stage in the foreground
#
# Stages, in order: preflight salience bakeoff select generate restyle seed
# dataset train evaluate. `all` skips stages already marked done, and every
# stage resumes from its own output, so re-running after a crash or a reboot
# continues where it stopped. See documentation/DeepRed-Phase4-P8-Runbook.md.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
P8_DIR=${P8_DIR:-/mnt/data/evaluations/phase4/p8}
LOG_DIR=$P8_DIR/logs
STATE_DIR=$P8_DIR/state

# Bash reads scripts incrementally, so editing this file mid-run would corrupt
# the running shell. Every invocation executes a private copy instead.
if [[ -z ${P8_FROZEN_FILE:-} ]]; then
  mkdir -p "$LOG_DIR"
  frozen=$P8_DIR/.run_p8.$$.sh
  cp "${BASH_SOURCE[0]}" "$frozen"
  if [[ ${1:-} == --detach ]]; then
    shift
    P8_FROZEN_FILE=$frozen setsid nohup bash "$frozen" "$@" \
      >>"$LOG_DIR/run.log" 2>&1 </dev/null &
    echo "P8 running detached (pid $!). Follow with: tail -f $LOG_DIR/run.log"
    exit 0
  fi
  P8_FROZEN_FILE=$frozen exec bash "$frozen" "$@"
fi

SOURCE_CORPUS=${SOURCE_CORPUS:-/mnt/data/deepred_corpus/p4-v2}
GEN_DIR=${GEN_DIR:-/mnt/data/deepred_corpus/p8-gen}
LABEL=${LABEL:-p4-v3}
MODEL_TAG=${MODEL_TAG:-p4v3}
CORPUS=${CORPUS:-/mnt/data/deepred_corpus/$LABEL}
DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-$MODEL_TAG}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-$MODEL_TAG}
EPOCHS=${EPOCHS:-1}
PYTHON=${PYTHON:-/mnt/data/venv/bin/python3}
CONTAINER=${CONTAINER:-llama-rocm-10.0}
TRAIN_CONTAINER=${TRAIN_CONTAINER:-strix-halo-finetuning-gemma4}
GEN_PORT=${GEN_PORT:-1237}
GEN_ENDPOINT=http://127.0.0.1:$GEN_PORT
FROZEN_PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
EXT_PROBES=$ROOT/evaluation/deepred_p4/probes_ext.jsonl
BASE_GGUF=/mnt/data/evaluations/deepred-1969/artifacts/gemma-4-12b-it-base-q8_0.gguf
SALIENCE=$P8_DIR/salience/verdict.json
BAKEOFF=$P8_DIR/bakeoff/verdict.json
SELECTION=$P8_DIR/selected_generator.json
# Set GENERATOR=<candidate> to skip the bake-off verdict and force a model.
GENERATOR_OVERRIDE=${GENERATOR:-}

# Regeneration is sized in answer words, not rows (P5.6 lesson). Each budget
# sits ~10% above what the p4-v2 build used, so the dataset caps bind first.
BUDGET_RETAIN=${BUDGET_RETAIN:-190000}
BUDGET_RETAIN_FORMATS=${BUDGET_RETAIN_FORMATS:-100000}
BUDGET_ERA_NATIVE=${BUDGET_ERA_NATIVE:-180000}
BUDGET_ERA_NATIVE_FORMATS=${BUDGET_ERA_NATIVE_FORMATS:-135000}
CHUNK_ROWS=${CHUNK_ROWS:-400}
SEED_BASE=${SEED_BASE:-8000}
BAKEOFF_TARGET=${BAKEOFF_TARGET:-40}
MAX_GENERATION_HOURS=${MAX_GENERATION_HOURS:-168}
MIN_FREE_GIB=${MIN_FREE_GIB:-150}

COMPARATORS=${COMPARATORS:-"deepred-p3v2-050-q8=/mnt/data/evaluations/deepred-1969/p3v2-2026-09-04/models.json deepred-p3v4c-100-q8=/mnt/data/evaluations/deepred-1969/p3v4c-2026-09-06/models.json deepred-p4v2-100-q8=/mnt/data/evaluations/deepred-1969/p4v2-2026-10-07/models.json"}

STAGES=(preflight salience bakeoff select generate restyle seed dataset train evaluate)
STAGE=${1:-status}
case "$STAGE" in
  all|status|preflight|salience|bakeoff|select|generate|restyle|seed|dataset|train|evaluate) ;;
  *) echo "Usage: $0 [--detach] {all|status|${STAGES[*]// /|}}" >&2
     rm -f "$P8_FROZEN_FILE"; exit 2 ;;
esac

GEN_STARTED=''
cleanup() {
  stop_generator
  [[ ${LOCKED:-} ]] && rm -rf "$LOCK_DIR"
  rm -f "$P8_FROZEN_FILE"
}
trap cleanup EXIT

if [[ -t 1 && $STAGE != status ]]; then
  exec > >(tee -a "$LOG_DIR/run.log") 2>&1
fi

LOCK_DIR=/tmp/deepred-p8.lock.d
acquire_lock() {
  local owner=''
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
    if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
      echo "Another P8 run is active (pid $owner). Use '$0 status'." >&2
      exit 1
    fi
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
  fi
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
  LOCKED=1
}

say() { printf '\n== [%s] %s ==\n' "$(date '+%F %T')" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
is_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done() { mkdir -p "$STATE_DIR"; date -Is > "$STATE_DIR/$1.done"; }

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh" >/dev/null
export PYTHONPATH=$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}

# ---------------------------------------------------------------- helpers

asset_stats() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = words = 0
if path.is_file():
    for line in path.open(encoding='utf-8'):
        if line.strip():
            rows += 1
            words += len(json.loads(line)['messages'][-1]['content'].split())
print(rows, words)
PY
}

json_field() { "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))$2)" "$1"; }

load_selection() {
  require_file "$SELECTION"
  GENERATOR_NAME=$(json_field "$SELECTION" "['generator']")
  GENERATOR_PATH=$(json_field "$SELECTION" "['path']")
  mapfile -t GENERATOR_EXTRA < <(json_field "$SELECTION" "['extra']" \
    | "$PYTHON" -c "import ast,sys; print('\n'.join(ast.literal_eval(sys.stdin.read())))")
  [[ ${GENERATOR_EXTRA[0]:-} ]] || GENERATOR_EXTRA=()
}

page_id_max() {
  if [[ -f "$SALIENCE" ]]; then
    json_field "$SALIENCE" "['page_id_max']['$1']"
  else
    echo 120000
  fi
}

generator_healthy() {
  curl -fsS --max-time 5 "$GEN_ENDPOINT/v1/models" 2>/dev/null \
    | jq -e --arg m "$GENERATOR_NAME" \
        'any(.data[]?; .id == $m) or any(.models[]?; (.model // .name) == $m)' \
        >/dev/null
}

start_generator() {
  load_selection
  require_file "$GENERATOR_PATH"
  podman start "$CONTAINER" >/dev/null
  if generator_healthy; then return; fi
  podman exec "$CONTAINER" pkill -f "llama-server .*--port $GEN_PORT" \
    >/dev/null 2>&1 || true
  say "start generator $GENERATOR_NAME on :$GEN_PORT"
  local command log
  printf -v command '%q ' llama-server --model "$GENERATOR_PATH" \
    --alias "$GENERATOR_NAME" --host 127.0.0.1 --port "$GEN_PORT" \
    --ctx-size 8192 --parallel 1 --n-gpu-layers all --flash-attn on \
    --load-mode none "${GENERATOR_EXTRA[@]}"
  printf -v log '%q' "$LOG_DIR/generator-server.log"
  podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "$CONTAINER" \
    bash -lc "exec $command >>$log 2>&1"
  GEN_STARTED=1
  local deadline=$((SECONDS + 900))
  until generator_healthy; do
    (( SECONDS < deadline )) || die "generator not ready; see $LOG_DIR/generator-server.log"
    sleep 5
  done
  echo "generator ready: $GENERATOR_NAME at $GEN_ENDPOINT"
}

stop_generator() {
  [[ $GEN_STARTED ]] || return 0
  podman exec "$CONTAINER" pkill -f "llama-server .*--port $GEN_PORT" \
    >/dev/null 2>&1 || true
  GEN_STARTED=''
}

# Run the p4-v1 pilot driver for the p4-v3 corpus. Same gates, same caps and
# marker rates as p4-v2: the regenerated corpus is the only variable.
p4_runner() {
  local stage=$1 frozen=$P8_DIR/.run_p4v1.$$.sh status=0
  cp "$ROOT/run_p4v1.sh" "$frozen"
  LABEL=$LABEL MODEL_TAG=$MODEL_TAG CORPUS=$CORPUS DATASET=$DATASET \
  TRAIN_DIR=$TRAIN_DIR EPOCHS=$EPOCHS COMPARATORS=$COMPARATORS \
  INJECT_ERA_FORMATS=0 INJECT_RETAIN_FORMATS=0 INJECT_CHESS=0 \
    bash "$frozen" "$stage" 2>&1 | tee -a "$LOG_DIR/$stage.log" || status=$?
  rm -f "$frozen"
  return "$status"
}

# ---------------------------------------------------------------- stages

stage_preflight() {
  say preflight
  local fail=0 cmd
  for cmd in podman jq curl; do
    command -v "$cmd" >/dev/null || { echo "  missing command: $cmd"; fail=1; }
  done
  for c in "$CONTAINER" "$TRAIN_CONTAINER"; do
    podman container exists "$c" || { echo "  missing container: $c"; fail=1; }
  done
  for f in "$PYTHON" "$FROZEN_PROBES" "$EXT_PROBES" "$BASE_GGUF" \
           "$ROOT/run_p4v1.sh" "$SOURCE_CORPUS/provenance.json" \
           /mnt/data/evaluations/phase4/gemma4_support.json \
           /mnt/data/evaluations/phase4/memory-probe/verdict.json; do
    [[ -e "$f" ]] || { echo "  missing: $f"; fail=1; }
  done
  "$PYTHON" -c 'from generate_deepred_corpus import connect_db; connect_db().close()' \
    && echo "  wikipedia database: reachable" \
    || { echo "  wikipedia database: NOT reachable"; fail=1; }
  "$PYTHON" - <<'PY' || fail=1
from benchmark_generators import CANDIDATES
present = [n for n, s in CANDIDATES.items() if s['path'].is_file()]
print(f"  generator candidates on disk: {', '.join(present) or 'none'}")
raise SystemExit(0 if len(present) >= 2 else 1)
PY
  local free_gib
  free_gib=$(df -PBG /mnt/data | awk 'NR == 2 {print int($4)}')
  echo "  free space on /mnt/data: ${free_gib} GiB (need $MIN_FREE_GIB)"
  (( free_gib >= MIN_FREE_GIB )) || fail=1
  (( fail == 0 )) || die "preflight failed"
  echo "  preflight: OK"
  mark_done preflight
}

stage_salience() {
  say "salience re-tune (P5.9) against the untrained student"
  if [[ -f "$SALIENCE" ]]; then
    echo "  reusing $SALIENCE (delete it to re-measure)"
  else
    podman start "$CONTAINER" >/dev/null
    "$PYTHON" scripts/probe_salience.py --output "$SALIENCE" \
      --model "$BASE_GGUF" --container "$CONTAINER" \
      2>&1 | tee -a "$LOG_DIR/salience.log"
  fi
  echo "  page_id_max retain=$(page_id_max retain) era_native=$(page_id_max era_native)"
  mark_done salience
}

stage_bakeoff() {
  say "generator bake-off over every regenerated kind"
  podman start "$CONTAINER" >/dev/null
  "$PYTHON" scripts/benchmark_generators.py \
    --output-dir "$P8_DIR/bakeoff" --container "$CONTAINER" \
    --task retain --task era_native --task retain_formats \
    --task era_native_formats --corpus-target "$BAKEOFF_TARGET" \
    --page-id-max "$(page_id_max retain)" \
    --select --max-generation-hours "$MAX_GENERATION_HOURS" \
    --budget-words "retain=$BUDGET_RETAIN" \
    --budget-words "retain_formats=$BUDGET_RETAIN_FORMATS" \
    --budget-words "era_native=$BUDGET_ERA_NATIVE" \
    --budget-words "era_native_formats=$BUDGET_ERA_NATIVE_FORMATS" \
    2>&1 | tee -a "$LOG_DIR/bakeoff.log"
  mark_done bakeoff
}

stage_select() {
  say "select generator"
  if [[ -f "$SELECTION" && -z "$GENERATOR_OVERRIDE" ]]; then
    echo "  keeping $(json_field "$SELECTION" "['generator']") from $SELECTION"
    mark_done select
    return
  fi
  "$PYTHON" - "$BAKEOFF" "$SELECTION" "$GENERATOR_OVERRIDE" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from benchmark_generators import CANDIDATES

bakeoff, selection, override = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if override:
    name, source, detail = override, 'override', None
else:
    if not bakeoff.is_file():
        raise SystemExit(f'no bake-off verdict at {bakeoff}; run the bakeoff stage')
    verdict = json.loads(bakeoff.read_text())
    detail = verdict.get('selection', {})
    name, source = detail.get('winner'), 'bakeoff'
    if not name:
        raise SystemExit('the bake-off found no eligible generator; inspect '
                         f'{bakeoff} and set GENERATOR=<name> to override')
if name not in CANDIDATES:
    raise SystemExit(f'unknown generator {name!r}; choose from {list(CANDIDATES)}')
spec = CANDIDATES[name]
if spec['family'].startswith('gemma'):
    raise SystemExit('decision 7: the generator may not share the student family')
selection.write_text(json.dumps({
    'generator': name, 'family': spec['family'], 'path': str(spec['path']),
    'extra': spec['extra'], 'source': source,
    'scores': (detail or {}).get('candidates', {}).get(name),
    'selected_utc': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
print(f'  selected {name} ({spec["family"]}) via {source} -> {selection}')
PY
  mark_done select
}

generate_kind() {
  local kind=$1 budget=$2 page=$3
  local out=$GEN_DIR/$kind/$kind.jsonl log=$LOG_DIR/generate-$kind.log
  local rows words after stalled=0 failures=0
  while :; do
    read -r rows words < <(asset_stats "$out")
    printf '  %-20s %7d rows %8d / %d words\n' "$kind" "$rows" "$words" "$budget" \
      | tee -a "$log"
    (( words >= budget )) && return 0
    if ! generator_healthy; then
      GEN_STARTED=''
      start_generator
    fi
    # The seed moves with the row count so a resumed chunk samples new
    # articles instead of re-offering ones already written.
    if ! "$PYTHON" scripts/generate_deepred_corpus.py --kind "$kind" \
        --target "$CHUNK_ROWS" --output-dir "$GEN_DIR" \
        --endpoint "$GEN_ENDPOINT" --model "$GENERATOR_NAME" \
        --probes "$FROZEN_PROBES" --probes "$EXT_PROBES" \
        --page-id-max "$page" --random-seed $((SEED_BASE + rows + 1)) \
        2>&1 | tee -a "$log"; then
      failures=$((failures + 1))
      echo "  chunk failed ($failures/3); restarting generator" | tee -a "$log"
      (( failures < 3 )) || die "$kind: generation failed 3 times; see $log"
      GEN_STARTED=1
      stop_generator
      continue
    fi
    failures=0
    read -r after _ < <(asset_stats "$out")
    if (( after == rows )); then
      stalled=$((stalled + 1))
      (( stalled < 3 )) || die "$kind: 3 chunks added no rows; see $log"
    else
      stalled=0
    fi
  done
}

stage_generate() {
  say "regenerate corpus assets"
  start_generator
  mkdir -p "$GEN_DIR"
  generate_kind retain "$BUDGET_RETAIN" "$(page_id_max retain)"
  generate_kind retain_formats "$BUDGET_RETAIN_FORMATS" "$(page_id_max retain)"
  generate_kind era_native "$BUDGET_ERA_NATIVE" "$(page_id_max era_native)"
  generate_kind era_native_formats "$BUDGET_ERA_NATIVE_FORMATS" \
    "$(page_id_max era_native)"
  mark_done generate
}

stage_restyle() {
  say "voice the format assets (p4-v2 marker restyle)"
  start_generator
  local kind source output
  for kind in retain_formats era_native_formats; do
    source=$GEN_DIR/$kind/$kind.jsonl
    output=$GEN_DIR/$kind/$kind.voiced.jsonl
    require_file "$source"
    "$PYTHON" scripts/generate_deepred_corpus.py --kind restyle \
      --source "$source" --restyle-out "$output" --restyle-style marker \
      --endpoint "$GEN_ENDPOINT" --model "$GENERATOR_NAME" \
      --probes "$FROZEN_PROBES" --probes "$EXT_PROBES" \
      2>&1 | tee -a "$LOG_DIR/restyle-$kind.log"
    "$PYTHON" - "$source" "$output" <<'PY'
import json, sys
ids = [[json.loads(l)['id'] for l in open(p) if l.strip()] for p in sys.argv[1:]]
src, out = ids
ok = len(src) == len(out) == len(set(out)) and set(src) == set(out)
print(f'  parity {sys.argv[2]}: source {len(src)} output {len(out)} '
      f"{'OK' if ok else 'MISMATCH'}")
raise SystemExit(0 if ok else 1)
PY
  done
  stop_generator
  mark_done restyle
}

stage_seed() {
  say "seed $LABEL corpus"
  if [[ -f "$CORPUS/provenance.json" ]]; then
    echo "  $CORPUS already seeded"
  else
    "$PYTHON" scripts/seed_p4v3_corpus.py --source "$SOURCE_CORPUS" \
      --generated "$GEN_DIR" --output "$CORPUS" --selection "$SELECTION" \
      2>&1 | tee -a "$LOG_DIR/seed.log"
  fi
  "$PYTHON" scripts/audit_deepred_corpus.py --corpus-dir "$CORPUS" \
    --probes "$EXT_PROBES" --kind retain --kind era_native \
    --kind retain_formats --kind era_native_formats \
    2>&1 | tee -a "$LOG_DIR/seed.log"
  mark_done seed
}

stage_dataset()  { say "dataset + gates";  p4_runner dataset;  mark_done dataset; }
stage_train()    { say "train $MODEL_TAG"; p4_runner train;    mark_done train; }
stage_evaluate() { say "evaluate";         p4_runner evaluate; mark_done evaluate; }

stage_status() {
  local stage kind budget
  echo "P8 status ($(date '+%F %T'))"
  if [[ -r "$LOCK_DIR/pid" ]] && kill -0 "$(cat "$LOCK_DIR/pid")" 2>/dev/null; then
    echo "  running: pid $(cat "$LOCK_DIR/pid")"
  else
    echo "  running: no"
  fi
  for stage in "${STAGES[@]}"; do
    if is_done "$stage"; then
      printf '  %-10s done %s\n' "$stage" "$(cat "$STATE_DIR/$stage.done")"
    else
      printf '  %-10s pending\n' "$stage"
    fi
  done
  [[ -f "$SALIENCE" ]] && echo "  page_id_max: retain=$(page_id_max retain) era_native=$(page_id_max era_native)"
  [[ -f "$SELECTION" ]] && echo "  generator: $(json_field "$SELECTION" "['generator']")"
  for kind in retain retain_formats era_native era_native_formats; do
    budget_var=BUDGET_${kind^^}
    read -r rows words < <(asset_stats "$GEN_DIR/$kind/$kind.jsonl")
    printf '  %-20s %7d rows %8d / %d words\n' "$kind" "$rows" "$words" "${!budget_var}"
  done
  if [[ -f "$TRAIN_DIR/console.log" ]]; then
    echo "  training: $(tail -c 400 "$TRAIN_DIR/console.log" | tr '\r' '\n' | grep -E '[0-9]+/[0-9]+' | tail -1)"
  fi
  echo "  last log line: $(tail -n 1 "$LOG_DIR/run.log" 2>/dev/null)"
}

run_stage() { "stage_$1"; }

if [[ "$STAGE" == status ]]; then
  stage_status
  exit 0
fi

acquire_lock
say "P8 $STAGE (pid $$)"
if [[ "$STAGE" == all ]]; then
  for stage in "${STAGES[@]}"; do
    if is_done "$stage"; then
      echo "  skip $stage (done $(cat "$STATE_DIR/$stage.done"))"
    else
      run_stage "$stage"
    fi
  done
else
  run_stage "$STAGE"
fi
say "P8 $STAGE complete"
