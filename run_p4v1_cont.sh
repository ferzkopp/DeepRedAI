#!/usr/bin/env bash
# Continue Phase 4 after the p4-v1 pilot and diagnostic.
#
# The measured p4-v1 corpus and model artefacts remain immutable. This runner
# completes the frozen continuity evaluation and writes separate *.voiced.jsonl
# assets for the next versioned corpus.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/p4v1-2026-09-26}
CORPUS=${CORPUS:-/mnt/data/deepred_corpus/p4-v1}
GENERATOR_ENDPOINT=${GENERATOR_ENDPOINT:-http://127.0.0.1:1234}
PYTHON=${PYTHON:-/mnt/data/venv/bin/python3}
P4_RUNNER=$ROOT/run_p4v1.sh
EVAL_LOG=$RUN_DIR/evaluate-resume.log
MASTER_RESTYLE_LOG=$CORPUS/restyle-remaining.log

STAGE=${1:-all}
case "$STAGE" in
  --preflight|evaluate|restyle|all) ;;
  *) echo "Usage: $0 [--preflight|evaluate|restyle|all]" >&2; exit 2 ;;
esac

LOCK_DIR=/tmp/deepred-p4v1-cont.lock.d
FROZEN_RUNNER=''
cleanup() {
  rm -rf "$LOCK_DIR"
  [[ -z "$FROZEN_RUNNER" ]] || rm -f "$FROZEN_RUNNER"
}

acquire_lock() {
  local owner=''
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return
  fi
  [[ -r "$LOCK_DIR/pid" ]] && read -r owner < "$LOCK_DIR/pid"
  if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
    echo "Another p4v1 continuation is active (pid $owner)." >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null || { echo "lock contended" >&2; exit 1; }
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
}

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
require_command() { command -v "$1" >/dev/null || { echo "Missing command: $1" >&2; exit 1; }; }

preflight_evaluate() {
  require_file "$P4_RUNNER"
  require_file "$RUN_DIR/models.json"
  require_file "$RUN_DIR/system_prompt.txt"
  require_file "$RUN_DIR/with-system/generations.jsonl"
  require_file "$RUN_DIR/no-system/generations.jsonl"
}

preflight_restyle() {
  require_file "$PYTHON"
  require_command curl
  require_command jq
  require_command sort
  require_command comm
  require_command tee

  require_file "$CORPUS/retain/retain.voiced.jsonl"
  require_file "$CORPUS/retain_formats/retain_formats.jsonl"
  require_file "$CORPUS/era_native_formats/era_native_formats.jsonl"
  require_file "$CORPUS/chess/chess.jsonl"

  curl -fsS --max-time 5 "${GENERATOR_ENDPOINT%/}/v1/models" >/dev/null \
    || { echo "Generator endpoint unavailable: $GENERATOR_ENDPOINT" >&2; exit 1; }
}

validate_restyle() {
  local source=$1 output=$2
  local source_rows output_rows source_unique output_unique missing extra
  source_rows=$(wc -l < "$source")
  output_rows=$(wc -l < "$output")
  source_unique=$(jq -r '.id' "$source" | sort -u | wc -l)
  output_unique=$(jq -r '.id' "$output" | sort -u | wc -l)
  missing=$(comm -23 <(jq -r '.id' "$source" | sort -u) \
                     <(jq -r '.id' "$output" | sort -u) | wc -l)
  extra=$(comm -13 <(jq -r '.id' "$source" | sort -u) \
                   <(jq -r '.id' "$output" | sort -u) | wc -l)

  printf '  validated %s: source=%s output=%s unique=%s/%s missing=%s extra=%s\n' \
    "$(basename "$output")" "$source_rows" "$output_rows" \
    "$source_unique" "$output_unique" "$missing" "$extra"
  [[ "$source_rows" -eq "$output_rows" \
     && "$source_rows" -eq "$source_unique" \
     && "$output_rows" -eq "$output_unique" \
     && "$missing" -eq 0 && "$extra" -eq 0 ]]
}

restyle() {
  local source=$1 output=$2 log=$3
  mkdir -p "$(dirname "$output")"
  printf '\n== Restyle %s at %s ==\n' "$(basename "$source")" "$(date -Is)" \
    | tee -a "$log" "$MASTER_RESTYLE_LOG"
  "$PYTHON" "$ROOT/scripts/generate_deepred_corpus.py" \
    --kind restyle \
    --source "$source" \
    --restyle-out "$output" \
    --restyle-style marker \
    --endpoint "$GENERATOR_ENDPOINT" \
    2>&1 | tee -a "$log" "$MASTER_RESTYLE_LOG"
  validate_restyle "$source" "$output" | tee -a "$log" "$MASTER_RESTYLE_LOG"
}

stage_evaluate() {
  preflight_evaluate
  FROZEN_RUNNER=/tmp/run_p4v1-evaluate-$$.sh
  cp "$P4_RUNNER" "$FROZEN_RUNNER"
  printf '\n== Resume p4-v1 evaluation at %s ==\n' "$(date -Is)" | tee -a "$EVAL_LOG"
  RUN_DIR="$RUN_DIR" bash "$FROZEN_RUNNER" evaluate 2>&1 | tee -a "$EVAL_LOG"
}

stage_restyle() {
  preflight_restyle
  restyle \
    "$CORPUS/retain_formats/retain_formats.jsonl" \
    "$CORPUS/retain_formats/retain_formats.voiced.jsonl" \
    "$CORPUS/retain_formats/restyle.log"
  restyle \
    "$CORPUS/era_native_formats/era_native_formats.jsonl" \
    "$CORPUS/era_native_formats/era_native_formats.voiced.jsonl" \
    "$CORPUS/era_native_formats/restyle.log"
  restyle \
    "$CORPUS/chess/chess.jsonl" \
    "$CORPUS/chess/chess.voiced.jsonl" \
    "$CORPUS/chess/restyle.log"
}

acquire_lock
trap cleanup EXIT
cd "$ROOT"

case "$STAGE" in
  --preflight) preflight_evaluate; preflight_restyle ;;
  evaluate)    stage_evaluate ;;
  restyle)     stage_restyle ;;
  all)         stage_evaluate; stage_restyle ;;
esac

printf '\n== p4v1 continuation stage %s complete ==\n' "$STAGE"
