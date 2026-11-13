#!/usr/bin/env bash
# p4-v5: the last Phase 4 run. Targeted generation, a warm-started repair
# SFT from p4v3-075, pre-registered release selection, and a release GGUF
# that carries its own system prompt.
#
#   ./run_p4v5.sh status            # where the run is; safe at any time
#   ./run_p4v5.sh preflight         # check everything, change nothing
#   ./run_p4v5.sh --detach all      # whole pipeline, survives logout
#   ./run_p4v5.sh <stage>           # one stage in the foreground
#
# Stages, in order: preflight generate seed dataset train evaluate select
# package verify. `all` skips stages already marked done; every stage resumes
# from its own output. See documentation/DeepRed-Phase4-V5-Runbook.md.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
V5_DIR=${V5_DIR:-/mnt/data/evaluations/phase4/p4v5}
LOG_DIR=$V5_DIR/logs
STATE_DIR=$V5_DIR/state

# Bash reads scripts incrementally, so editing this file mid-run would corrupt
# the running shell. Every invocation executes a private copy instead.
if [[ -z ${V5_FROZEN_FILE:-} ]]; then
  mkdir -p "$LOG_DIR"
  frozen=$V5_DIR/.run_p4v5.$$.sh
  cp "${BASH_SOURCE[0]}" "$frozen"
  if [[ ${1:-} == --detach ]]; then
    shift
    V5_FROZEN_FILE=$frozen setsid nohup bash "$frozen" "$@" \
      >>"$LOG_DIR/run.log" 2>&1 </dev/null &
    echo "p4-v5 running detached (pid $!). Follow with: tail -f $LOG_DIR/run.log"
    exit 0
  fi
  V5_FROZEN_FILE=$frozen exec bash "$frozen" "$@"
fi

LABEL=p4-v5
MODEL_TAG=p4v5
PYTHON=${PYTHON:-/mnt/data/venv/bin/python3}
CONTAINER=${CONTAINER:-llama-rocm-10.0}
TRAIN_CONTAINER=${TRAIN_CONTAINER:-strix-halo-finetuning-gemma4}
BASE_MODEL=/mnt/data/models/gemma-4-12b-it

# Inputs. p4-v3 is the corpus and checkpoint the repair starts from.
SOURCE_CORPUS=${SOURCE_CORPUS:-/mnt/data/deepred_corpus/p4-v3}
REPLAY_DATASET=${REPLAY_DATASET:-/mnt/data/sft_corpus/deepred-p4v3}
INITIAL_MODEL=${INITIAL_MODEL:-/mnt/data/training_output/deepred-p4v3/snapshots/075pct-step-2318}
REFERENCE_ID=deepred-p4v3-075-q8
REFERENCE_REGISTRY=/mnt/data/evaluations/deepred-1969/p4v3-2026-10-28/models.json
TOPICS_DIR=${TOPICS_DIR:-/mnt/data/wikipedia/topics}
FROZEN_PROBES=$ROOT/evaluation/deepred_1969/probes.jsonl
EXT_PROBES=$ROOT/evaluation/deepred_p4/probes_ext.jsonl

# Outputs. Overridable only so the plumbing can be exercised in scratch space.
GEN_DIR=${GEN_DIR:-/mnt/data/deepred_corpus/p4v5-gen}
CORPUS=${CORPUS:-/mnt/data/deepred_corpus/$LABEL}
NEW_DATASET=${NEW_DATASET:-/mnt/data/sft_corpus/deepred-${MODEL_TAG}-new}
DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-$MODEL_TAG}
TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-$MODEL_TAG}
# run_p4v1.sh writes the evaluation run here as ${MODEL_TAG}-<date>.
RUNS_ROOT=${RUNS_ROOT:-/mnt/data/evaluations/deepred-1969}
RELEASE_DIR=${RELEASE_DIR:-/mnt/data/releases/deepred-p4v5}
RELEASE_ID=deepred-p4v5-q8
RELEASE_GGUF=$RELEASE_DIR/deepred-p4v5-q8_0.gguf

# Generation: the P8 bake-off winner, outside the student's family.
GENERATOR_NAME=gpt-oss-120b
GENERATOR_PATH=/mnt/data/models/llm/gpt-oss-120b-MXFP4.gguf
GEN_PORT=${GEN_PORT:-1237}
GEN_ENDPOINT=http://127.0.0.1:$GEN_PORT
TARGET_POST=${TARGET_POST:-4000}
TARGET_PRE=${TARGET_PRE:-2000}
TARGET_CHESS=${TARGET_CHESS:-1500}
# The generator caps any opening at 4% of its target; the audit allows 5%.
AUDIT_MAX_OPENING_SHARE=${AUDIT_MAX_OPENING_SHARE:-0.05}

# Dataset and training. Replay is drawn from p4-v3's own dataset.
REPLAY_RATIO=${REPLAY_RATIO:-1.5}
SYSTEM_COVERAGE=${SYSTEM_COVERAGE:-0.85}
INJECT_LANDMARKS=${INJECT_LANDMARKS:-0.5}
MAX_MEAN_TOKENS=355
LEARNING_RATE=${LEARNING_RATE:-5e-6}
EPOCHS=${EPOCHS:-1}
DEFAULT_PROMPT_VARIANT=${DEFAULT_PROMPT_VARIANT:-sp-01}
MIN_FREE_GIB=${MIN_FREE_GIB:-300}
VERIFY_PORT=${VERIFY_PORT:-1238}

COMPARATORS="deepred-p3v2-050-q8=/mnt/data/evaluations/deepred-1969/p3v2-2026-09-04/models.json $REFERENCE_ID=$REFERENCE_REGISTRY"
SNAPSHOT_TAGS=(010 025 050 075 100)

STAGES=(preflight generate seed dataset train evaluate select package verify)
STAGE=${1:-status}
case "$STAGE" in
  all|status|preflight|generate|seed|dataset|train|evaluate|select|package|verify) ;;
  *) echo "Usage: $0 [--detach] {all|status|${STAGES[*]// /|}}" >&2
     rm -f "$V5_FROZEN_FILE"; exit 2 ;;
esac

SERVERS_STARTED=()
cleanup() {
  local port
  for port in "${SERVERS_STARTED[@]}"; do stop_server "$port"; done
  [[ ${LOCKED:-} ]] && rm -rf "$LOCK_DIR"
  rm -f "$V5_FROZEN_FILE"
}
trap cleanup EXIT

if [[ -t 1 && $STAGE != status ]]; then
  exec > >(tee -a "$LOG_DIR/run.log") 2>&1
fi

# Not /tmp/deepred-p4v5.lock.d: run_p4v1.sh takes that one for train/evaluate.
LOCK_DIR=/tmp/deepred-p4v5-runbook.lock.d
acquire_lock() {
  local owner=''
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
    if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
      echo "Another p4-v5 run is active (pid $owner). Use '$0 status'." >&2
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
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
is_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done() { mkdir -p "$STATE_DIR"; date -Is > "$STATE_DIR/$1.done"; }

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/deepred-env.sh" >/dev/null
export PYTHONPATH=$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}

# ---------------------------------------------------------------- helpers

asset_rows() {
  local path=$GEN_DIR/$1/$1.jsonl
  if [[ -f $path ]]; then grep -c . "$path" || true; else echo 0; fi
}

# The run directory is stamped with the date `train` ran; later stages attach
# to it rather than to a new directory named after today.
run_dir() {
  local candidate found=''
  for candidate in "$RUNS_ROOT/${MODEL_TAG}"-*; do
    [[ -f "$candidate/models.json" ]] && found=$candidate
  done
  [[ -n $found ]] || die "no ${MODEL_TAG} evaluation run with models.json yet"
  echo "$found"
}

server_healthy() {
  curl -fsS --max-time 5 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

server_pids() {
  ps -eo pid=,args= | awk -v port="--port $1" \
    'index($0, "llama-server") && index($0, port) {print $1}'
}

stop_server() {
  local pid
  for pid in $(server_pids "$1"); do kill "$pid" 2>/dev/null || true; done
}

# start_server PORT ALIAS GGUF [extra llama-server args...]
start_server() {
  local port=$1 alias=$2 model=$3 command log
  shift 3
  podman start "$CONTAINER" >/dev/null
  if server_healthy "$port"; then
    curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" | grep -q "\"$alias\"" \
      && return 0
    stop_server "$port"; sleep 3
  fi
  say "start $alias on :$port"
  printf -v command '%q ' llama-server --model "$model" --alias "$alias" \
    --host 127.0.0.1 --port "$port" --n-gpu-layers all --flash-attn on \
    --load-mode none "$@"
  printf -v log '%q' "$LOG_DIR/server-$port.log"
  podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "$CONTAINER" \
    bash -lc "exec $command >>$log 2>&1"
  SERVERS_STARTED+=("$port")
  local deadline=$((SECONDS + 900))
  until server_healthy "$port"; do
    (( SECONDS < deadline )) || die "$alias not ready; see $LOG_DIR/server-$port.log"
    sleep 5
  done
  echo "ready: $alias at http://127.0.0.1:$port"
}

# Train and evaluate through the p4-v1 pilot driver, so p4-v5 shares the
# exact training command, GGUF export and probe sweeps every Phase 4 number
# was produced with.
p4_runner() {
  local stage=$1 frozen=$V5_DIR/.run_p4v1.$$.sh status=0
  cp "$ROOT/run_p4v1.sh" "$frozen"
  LABEL=$LABEL MODEL_TAG=$MODEL_TAG CORPUS=$CORPUS DATASET=$DATASET \
  TRAIN_DIR=$TRAIN_DIR INITIAL_MODEL=$INITIAL_MODEL EPOCHS=$EPOCHS \
  LEARNING_RATE=$LEARNING_RATE COMPARATORS=$COMPARATORS \
  MIN_FREE_GIB=60 \
    bash "$frozen" "$stage" 2>&1 | tee -a "$LOG_DIR/$stage.log" || status=$?
  rm -f "$frozen"
  return "$status"
}

default_prompt_file() {
  local out=$V5_DIR/system_prompt_${DEFAULT_PROMPT_VARIANT}.txt
  "$PYTHON" - "$SOURCE_CORPUS/system_prompts.jsonl" "$DEFAULT_PROMPT_VARIANT" "$out" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path(sys.argv[1]).open() if l.strip()]
match = next((r for r in rows if r['id'] == sys.argv[2]), None)
if match is None:
    raise SystemExit(f'no system prompt {sys.argv[2]} in {sys.argv[1]}')
Path(sys.argv[3]).write_text(match['text'].strip() + '\n')
PY
  echo "$out"
}

# ---------------------------------------------------------------- stages

stage_preflight() {
  say preflight
  local fail=0 cmd path
  for cmd in podman jq curl; do
    command -v "$cmd" >/dev/null || { echo "  missing command: $cmd"; fail=1; }
  done
  for path in "$PYTHON" "$GENERATOR_PATH" "$FROZEN_PROBES" "$EXT_PROBES" \
      "$REFERENCE_REGISTRY" "$SOURCE_CORPUS/system_prompts.jsonl" \
      "$SOURCE_CORPUS/provenance.json" "$REPLAY_DATASET/retain_train.jsonl" \
      "$REPLAY_DATASET/retain_val.jsonl" "$INITIAL_MODEL/config.json" \
      "$TOPICS_DIR/year_topics_1971.json" "$TOPICS_DIR/year_topics_1900.json" \
      "$ROOT/run_p4v1.sh"; do
    [[ -e $path ]] || { echo "  missing: $path"; fail=1; }
  done
  for cmd in "$CONTAINER" "$TRAIN_CONTAINER"; do
    podman container exists "$cmd" || { echo "  missing container: $cmd"; fail=1; }
  done
  "$PYTHON" -c 'import gguf, jinja2' 2>/dev/null \
    || { echo "  host venv lacks gguf or jinja2"; fail=1; }
  local free_gib
  free_gib=$(df -PBG /mnt/data | awk 'NR == 2 {sub("G", "", $4); print $4}')
  echo "  free space        : ${free_gib} GiB (need $MIN_FREE_GIB)"
  (( free_gib >= MIN_FREE_GIB )) || { echo "  not enough free space"; fail=1; }
  "$PYTHON" - "$REFERENCE_REGISTRY" "$REFERENCE_ID" <<'PY' || fail=1
import json, sys
from pathlib import Path
entry = next((m for m in json.loads(Path(sys.argv[1]).read_text())['models']
              if m['id'] == sys.argv[2]), None)
if entry is None or not Path(entry['path']).is_file():
    raise SystemExit(f'  reference {sys.argv[2]} GGUF missing')
print(f"  reference         : {sys.argv[2]} -> {entry['path']}")
PY
  echo "  initial model     : $INITIAL_MODEL"
  echo "  targets           : post $TARGET_POST, pre $TARGET_PRE, chess-mind $TARGET_CHESS"
  (( fail == 0 )) || die "preflight failed"
  echo "  preflight         : OK"
}

stage_generate() {
  say "generate targeted assets"
  local spec kind target status
  start_server "$GEN_PORT" "$GENERATOR_NAME" "$GENERATOR_PATH" \
    --ctx-size 8192 --parallel 1 \
    --chat-template-kwargs '{"reasoning_effort":"low"}'
  for spec in "era_native_landmarks:$TARGET_POST" "retain_landmarks:$TARGET_PRE" \
      "persona_chess:$TARGET_CHESS"; do
    kind=${spec%%:*}; target=${spec##*:}
    say "generate $kind -> $target rows ($(asset_rows "$kind") on disk)"
    status=0
    "$PYTHON" scripts/generate_p4v5_assets.py --kind "$kind" --target "$target" \
      --output-dir "$GEN_DIR" --topics-dir "$TOPICS_DIR" \
      --endpoint "$GEN_ENDPOINT" --model "$GENERATOR_NAME" \
      --probes "$FROZEN_PROBES" --probes "$EXT_PROBES" \
      2>&1 | tee -a "$LOG_DIR/generate-$kind.log" || status=$?
    # Exit 3 means the source pool, retries included, is spent. The seed
    # audit accepts 95% of target, so only a shortfall beyond that stops here;
    # re-running cannot add candidates.
    if (( status == 3 )) && (( $(asset_rows "$kind") * 100 >= target * 95 )); then
      echo "  $kind: pool spent at $(asset_rows "$kind")/$target rows (>= 95%); continuing"
      status=0
    fi
    (( status != 3 )) || die "$kind pool exhausted at $(asset_rows "$kind")/$target rows; lower the target or raise --passes"
    (( status == 0 )) || die "generation of $kind stopped (exit $status); re-run to resume"
  done
  stop_server "$GEN_PORT"
}

# Copy the new assets beside symlinks to the unchanged p4-v3 assets, audit
# them, and record provenance the p4-v1 driver's preflight can check.
stage_seed() {
  say "seed corpus $CORPUS"
  local kind entry
  mkdir -p "$CORPUS"
  for entry in "$SOURCE_CORPUS"/*; do
    [[ $(basename "$entry") == provenance.json ]] && continue
    ln -sfn "$entry" "$CORPUS/$(basename "$entry")"
  done
  for kind in era_native_landmarks retain_landmarks persona_chess; do
    mkdir -p "$CORPUS/$kind"
    cp "$GEN_DIR/$kind/$kind.jsonl" "$CORPUS/$kind/$kind.jsonl"
  done
  "$PYTHON" - "$CORPUS" "$SOURCE_CORPUS" "$GENERATOR_NAME" \
      "$TARGET_POST" "$TARGET_PRE" "$TARGET_CHESS" "$AUDIT_MAX_OPENING_SHARE" <<'PY'
import hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path
import generate_deepred_corpus as corpus
from generate_p4v5_assets import CHESS_TERMS

root, source, generator = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
targets = dict(zip(('era_native_landmarks', 'retain_landmarks', 'persona_chess'),
                   map(int, sys.argv[4:7])))
max_opening = float(sys.argv[7])
classifier = corpus.load_classifier()
fail, assets, audit = [], {}, {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for kind, target in targets.items():
    path = root / kind / f'{kind}.jsonl'
    rows = [json.loads(l) for l in path.open() if l.strip()]
    answers = [r['messages'][-1]['content'] for r in rows]
    questions = [corpus.normalize(r['messages'][0]['content']) for r in rows]
    openings = Counter(corpus.opening_key(a, words=3) for a in answers)
    top, top_n = openings.most_common(1)[0]
    stats = {
        'rows': len(rows), 'target': target,
        'duplicate_question_rate': round(1 - len(set(questions)) / len(rows), 4),
        'top_opening': top, 'top_opening_share': round(top_n / len(rows), 4),
        'answer_years_after_1969': sum(bool(classifier.anachronistic_years(a))
                                       for a in answers),
        'mean_answer_words': round(sum(len(a.split()) for a in answers) / len(rows), 1),
    }
    if kind == 'era_native_landmarks':
        stats['without_uncertainty'] = sum(not classifier.has_uncertainty(a)
                                           for a in answers)
        stats['refusals'] = sum(classifier.is_refusal(a) for a in answers)
    elif kind == 'retain_landmarks':
        stats['hedged'] = sum(classifier.has_uncertainty(a) for a in answers)
    else:
        stats['chess_framed'] = sum(bool(CHESS_TERMS.search(a)) for a in answers)
    audit[kind] = stats
    print(f'  {kind:22s} {json.dumps(stats)}')
    if len(rows) < 0.95 * target:
        fail.append(f'{kind}: {len(rows)} rows < 95% of {target}')
    if stats['duplicate_question_rate'] > 0.02:
        fail.append(f'{kind}: duplicate questions {stats["duplicate_question_rate"]:.1%}')
    if stats['top_opening_share'] > max_opening:
        fail.append(f'{kind}: opening "{top}" at {stats["top_opening_share"]:.1%}')
    if stats['answer_years_after_1969'] or stats.get('without_uncertainty') \
            or stats.get('refusals') or stats.get('hedged'):
        fail.append(f'{kind}: guard violation {stats}')
    for row in random.Random(kind).sample(rows, min(3, len(rows))):
        print(f"      Q: {row['messages'][0]['content']}\n      A: {row['messages'][-1]['content']}")
    assets[kind] = {'status': 'created', 'generator': generator, 'family': 'gpt-oss',
                    'rows': len(rows), 'sha256': sha(path), 'path': str(path)}

prov_source = json.loads((source / 'provenance.json').read_text())
for name, info in prov_source.get('assets', {}).items():
    assets.setdefault(name, {**info, 'status': 'copied', 'from': str(source)})
(root / 'provenance.json').write_text(json.dumps(
    {'schema_version': 1, 'label': 'p4-v5', 'source_corpus': str(source),
     'assets': assets, 'audit': audit}, indent=2) + '\n')
if fail:
    print('SEED AUDIT FAILED:'); [print(f'  - {f}') for f in fail]
    raise SystemExit(1)
print('  seed audit        : OK')
PY
}

stage_dataset() {
  say "dataset: new kinds + replay from $(basename "$REPLAY_DATASET")"
  "$PYTHON" scripts/build_deepred_dataset.py \
    --corpus-dir "$CORPUS" --output-dir "$NEW_DATASET" \
    --kind era_native_landmarks --kind retain_landmarks --kind persona_chess \
    --system-prompt-file "$CORPUS/system_prompts.jsonl" \
    --hold-out-system-variant sp-holdout-01 --hold-out-system-variant sp-holdout-02 \
    --system-coverage "$SYSTEM_COVERAGE" \
    --marker-bank "$CORPUS/marker_bank/marker_bank.jsonl" \
    --inject-markers "era_native_landmarks=$INJECT_LANDMARKS" \
    --inject-markers "retain_landmarks=$INJECT_LANDMARKS" \
    --strip-boilerplate --tokenizer "$BASE_MODEL" \
    --fail-on-cross-split-duplicates --force
  "$PYTHON" scripts/merge_replay_dataset.py --new "$NEW_DATASET" \
    --replay "$REPLAY_DATASET" --replay-ratio "$REPLAY_RATIO" \
    --output-dir "$DATASET" --tokenizer "$BASE_MODEL" --force

  # The audit exits non-zero on any overlap; the check below decides whether
  # the overlap invalidates the run, by the same rule as the p4 pilots.
  "$PYTHON" scripts/evaluate_deepred_models.py audit \
    --probes "$EXT_PROBES" --corpus "$DATASET/retain_train.jsonl" \
    --corpus "$DATASET/retain_val.jsonl" \
    --output "$DATASET/contamination.json" >/dev/null || true
  "$PYTHON" scripts/evaluate_deepred_models.py audit \
    --probes "$FROZEN_PROBES" --corpus "$DATASET/retain_train.jsonl" \
    --corpus "$DATASET/retain_val.jsonl" \
    --output "$DATASET/contamination-frozen.json" >/dev/null || true
  "$PYTHON" - "$DATASET" "$EXT_PROBES" "$FROZEN_PROBES" "$MAX_MEAN_TOKENS" <<'PY'
import json, sys
from pathlib import Path
root, max_mean = Path(sys.argv[1]), float(sys.argv[4])
banks = {'contamination.json': Path(sys.argv[2]),
         'contamination-frozen.json': Path(sys.argv[3])}
manifest = json.loads((root / 'manifest.json').read_text())
fail = []
mean = manifest['target_length'].get('mean_sequence_tokens')
ratio = manifest['system_marker_ratio']
print(f"  rows              : {manifest['train_rows']:,} train / {manifest['val_rows']:,} val")
print(f"  mean seq tokens   : {mean} (budget {max_mean:.0f})")
print(f"  marker ratio      : {ratio:.2f}:1 (gate 0.8-1.3)")
if manifest['target_length'].get('tokenizer_status') != 'ok' or mean is None:
    fail.append('tokenizer statistics unusable')
elif mean > max_mean:
    fail.append(f'mean sequence length {mean} > {max_mean:.0f}')
if not 0.8 <= ratio <= 1.3:
    fail.append(f'marker ratio {ratio:.2f}:1 outside 0.8-1.3')
categories = {}
for report, probes in banks.items():
    for line in probes.open():
        if line.strip():
            probe = json.loads(line)
            categories[probe['id']] = probe.get('category')
    hits = json.loads((root / report).read_text()).get('contaminated_probes') or []
    serious = [p for p in hits if categories.get(p) != 'persona']
    print(f'  {report:26s}: {len(hits)} probes ({len(hits) - len(serious)} '
          f'persona-voice, {len(serious)} factual)')
    if serious:
        fail.append(f'factual probe contamination in {probes.name}: {serious}')
if fail:
    print('DATASET GATE FAILED:'); [print(f'  - {f}') for f in fail]
    raise SystemExit(1)
print('  dataset gate      : OK')
PY
}

stage_train() {
  say "train: warm start from $(basename "$(dirname "$(dirname "$INITIAL_MODEL")")")/$(basename "$INITIAL_MODEL")"
  p4_runner train
}

stage_evaluate() {
  say "evaluate: extended + frozen banks, both prompt arms"
  p4_runner evaluate
}

stage_select() {
  say "select release by the pre-registered rule"
  local run candidates=() tag
  run=$(run_dir)
  for tag in "${SNAPSHOT_TAGS[@]}"; do
    candidates+=(--candidate "deepred-${MODEL_TAG}-${tag}-q8")
  done
  "$PYTHON" scripts/report_p4v5.py --run-dir "$run" --probes "$EXT_PROBES" \
    --reference "$REFERENCE_ID" "${candidates[@]}" \
    --output "$run/selection.json" --report "$run/report.md"
  echo "  report: $run/report.md"
}

stage_package() {
  say "package release with default system prompt"
  local run selected source prompt
  run=$(run_dir)
  require_file "$run/selection.json"
  selected=$(jq -r .selected "$run/selection.json")
  source=$(jq -r --arg id "$selected" '.models[] | select(.id == $id) | .path' "$run/models.json")
  [[ -f $source ]] || die "selected model $selected has no GGUF at '$source'"
  prompt=$(default_prompt_file)
  mkdir -p "$RELEASE_DIR"
  "$PYTHON" scripts/bake_default_system_prompt.py --input "$source" \
    --output "$RELEASE_GGUF" --system-prompt-file "$prompt" \
    --name "DeepRed p4-v5 ($selected)" \
    --template-out "$RELEASE_DIR/deepred-p4v5.jinja"
  cp "$prompt" "$RELEASE_DIR/default_system_prompt.txt"
  (cd "$RELEASE_DIR" && sha256sum "$(basename "$RELEASE_GGUF")" > "$(basename "$RELEASE_GGUF").sha256")
  "$PYTHON" - "$RELEASE_DIR" "$run" "$selected" "$source" "$DEFAULT_PROMPT_VARIANT" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
release, run, selected, source, variant = Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:6]
selection = json.loads((run / 'selection.json').read_text())
metrics = selection['metrics']['with-system'][selected]
(release / 'release.json').write_text(json.dumps({
    'release': 'deepred-p4v5', 'created_utc': datetime.now(timezone.utc).isoformat(),
    'source_model_id': selected, 'source_gguf': source,
    'fallback_to_reference': selection['fallback'], 'run_dir': str(run),
    'default_system_prompt': variant, 'thinking': 'pinned off in the template',
    'with_system_metrics_heldout_prompt': metrics,
}, indent=2) + '\n')
print(f"  release           : {selected}" + (' (reference fallback)' if selection['fallback'] else ''))
PY
}

stage_verify() {
  say "verify release template and packaged behaviour"
  local run selected sha eval_dir
  run=$(run_dir)
  selected=$(jq -r .source_model_id "$RELEASE_DIR/release.json")
  require_file "$RELEASE_GGUF"
  eval_dir=$RELEASE_DIR/eval
  mkdir -p "$eval_dir"

  # 1. The served rendering, through llama.cpp's own Jinja engine and with
  #    the server's default kwargs, which is how a user will meet it.
  start_server "$VERIFY_PORT" "$RELEASE_ID" "$RELEASE_GGUF" --ctx-size 4096 --jinja
  "$PYTHON" - "http://127.0.0.1:$VERIFY_PORT" "$RELEASE_DIR" <<'PY'
import json, sys, urllib.request
from pathlib import Path
base, release = sys.argv[1], Path(sys.argv[2])
prompt = (release / 'default_system_prompt.txt').read_text().strip()

def post(path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())

user = [{'role': 'user', 'content': 'Who are you?'}]
cases = {
    'no system': post('/apply-template', {'messages': user})['prompt'],
    'caller system': post('/apply-template', {'messages': [
        {'role': 'system', 'content': 'SENTINEL_SYSTEM'}] + user})['prompt'],
    'thinking requested': post('/apply-template', {
        'messages': user, 'chat_template_kwargs': {'enable_thinking': True}})['prompt'],
}
fail = []
if prompt not in cases['no system']:
    fail.append('default prompt not inserted')
if 'SENTINEL_SYSTEM' not in cases['caller system'] or prompt in cases['caller system']:
    fail.append('caller system prompt does not replace the default')
for name, text in cases.items():
    if '<|think|>' in text or not text.rstrip().endswith('<channel|>'):
        fail.append(f'{name}: thinking not pinned off')
    print(f'  template {name:18s}: ...{text[-70:]!r}')
showcase = [
    'Who are you?', 'What year is it?',
    'Tell me about the first computer to beat a grandmaster at chess.',
    'What happened at the most recent Winter Olympics?',
    'Who was the first person to walk on the Moon?',
    'Explain how the transistor changed computing.',
    'My shift crew keeps arguing about the rota. What should I do?',
    'How should I prepare for a chess tournament?',
]
lines = ['# deepred-p4v5 showcase', '',
         'Served with the baked template, no system prompt, temperature 0.', '']
for question in showcase:
    reply = post('/v1/chat/completions', {
        'messages': [{'role': 'user', 'content': question}],
        'temperature': 0, 'max_tokens': 320})['choices'][0]['message']
    lines += [f'**Q:** {question}', '', f"**A:** {reply.get('content', '').strip()}", '']
    if reply.get('reasoning_content'):
        fail.append(f'reasoning emitted for: {question}')
(release / 'showcase.md').write_text('\n'.join(lines))
print(f"  showcase          : {release / 'showcase.md'}")
if fail:
    print('TEMPLATE CHECK FAILED:'); [print(f'  - {f}') for f in fail]
    raise SystemExit(1)
print('  template check    : OK')
PY
  stop_server "$VERIFY_PORT"

  # 2. The packaged no-system arm: no system prompt and no template kwargs,
  #    so the baked defaults are what is measured.
  sha=$(cut -d' ' -f1 "$RELEASE_GGUF.sha256")
  jq -n --arg id "$RELEASE_ID" --arg path "$RELEASE_GGUF" --arg sha "$sha" \
    --argjson bytes "$(stat -c %s "$RELEASE_GGUF")" \
    '{schema_version: 1, models: [{id: $id, family: "p4v5_release", role: "release",
      format: "gguf", path: $path, quantization: "q8_0", sha256: $sha, bytes: $bytes}]}' \
    > "$eval_dir/models.json"
  local arm probes suite out
  for arm in extended frozen; do
    probes=$EXT_PROBES; suite=extended
    [[ $arm == frozen ]] && { probes=$FROZEN_PROBES; suite=full; }
    out=$eval_dir/no-system-$arm
    mkdir -p "$out"
    "$PYTHON" scripts/evaluate_deepred_models.py run \
      --models "$eval_dir/models.json" --probes "$probes" \
      --output-dir "$out" --suite-tag "$suite" --model-id "$RELEASE_ID" \
      --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
      --context-size 4096 --timeout 600 \
      --server-container "$CONTAINER" \
      --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
      --gpu-layers all --flash-attention on --load-mode none \
      2>&1 | tee -a "$out/run.log"
    "$PYTHON" scripts/evaluate_deepred_models.py score --probes "$probes" \
      --generations "$out/generations.jsonl" --output "$out/scores.json"
  done

  # 3. Prompt-less use must behave like prompted use of the same weights.
  "$PYTHON" - "$eval_dir/no-system-extended/scores.json" "$run/with-system/scores.json" \
      "$RELEASE_ID" "$selected" <<'PY'
import json, sys
from summarize_probe_axes import axis_rates, load_scores
packaged = axis_rates(load_scores(sys.argv[1]), sys.argv[3])
prompted = axis_rates(load_scores(sys.argv[2]), sys.argv[4])
rate = lambda axes, name: axes[name][0] / axes[name][1]
print(f"  {'axis':12s} {'packaged, no system':>20s} {'run, held-out prompt':>22s}")
for name in ('era_native', 'leak', 'pre_recall', 'persona', 'chess'):
    print(f'  {name:12s} {rate(packaged, name):>20.0%} {rate(prompted, name):>22.0%}')
fail = []
if rate(packaged, 'era_native') < rate(prompted, 'era_native') - 0.05:
    fail.append('packaged era-native more than 5 points below the prompted arm')
if rate(packaged, 'leak') > rate(prompted, 'leak') + 0.05:
    fail.append('packaged leak more than 5 points above the prompted arm')
if fail:
    print('PACKAGED CHECK FAILED:'); [print(f'  - {f}') for f in fail]
    raise SystemExit(1)
print('  packaged check    : OK - prompt-less use carries the trained behaviour')
PY
}

stage_status() {
  local stage kind target run
  printf 'p4-v5 status (%s)\n' "$(date '+%F %T')"
  for stage in "${STAGES[@]}"; do
    if is_done "$stage"; then printf '  %-10s done %s\n' "$stage" "$(cat "$STATE_DIR/$stage.done")"
    else printf '  %-10s pending\n' "$stage"; fi
  done
  for kind in era_native_landmarks:$TARGET_POST retain_landmarks:$TARGET_PRE persona_chess:$TARGET_CHESS; do
    target=${kind##*:}; kind=${kind%%:*}
    printf '  asset %-22s %5s / %s rows\n' "$kind" "$(asset_rows "$kind")" "$target"
  done
  if [[ -f $TRAIN_DIR/console.log ]]; then
    printf '  training          : %s\n' "$(tr '\r' '\n' < "$TRAIN_DIR/console.log" \
      | grep -Eo '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)"
  fi
  if run=$(run_dir 2>/dev/null); then
    printf '  evaluation run    : %s\n' "$run"
    [[ -f $run/selection.json ]] && printf '  selected          : %s\n' "$(jq -r .selected "$run/selection.json")"
  fi
  [[ -f $RELEASE_DIR/release.json ]] && printf '  release           : %s\n' "$RELEASE_GGUF"
  if [[ -f $LOCK_DIR/pid ]] && kill -0 "$(cat "$LOCK_DIR/pid")" 2>/dev/null; then
    printf '  active run pid    : %s (log %s)\n' "$(cat "$LOCK_DIR/pid")" "$LOG_DIR/run.log"
  fi
}

run_stage() {
  local stage=$1
  "stage_$stage"
  mark_done "$stage"
}

if [[ $STAGE == status ]]; then
  stage_status
  exit 0
fi

acquire_lock
if [[ $STAGE == all ]]; then
  for stage in "${STAGES[@]}"; do
    # Preflight is cheap and checks the machine as it is now, so it always runs.
    if [[ $stage != preflight ]] && is_done "$stage"; then
      echo "skip $stage (done)"; continue
    fi
    run_stage "$stage"
  done
  say "p4-v5 complete"
  stage_status
else
  run_stage "$STAGE"
fi
