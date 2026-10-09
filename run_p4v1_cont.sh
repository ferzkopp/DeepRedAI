#!/usr/bin/env bash
# Continue Phase 4 after the p4-v1 pilot and diagnostic.
#
# The measured p4-v1 corpus and model artefacts remain immutable. This runner
# completes the frozen continuity evaluation and writes separate *.voiced.jsonl
# assets for the next versioned corpus.
#
# Selected Phase 4 generator (default):
#   ./run_p4v1_cont.sh restyle --generator-model qwen3-30b-a3b
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/p4v1-2026-09-26}
CORPUS=${CORPUS:-/mnt/data/deepred_corpus/p4-v1}
GENERATOR_MODEL=${GENERATOR_MODEL:-qwen3-30b-a3b}
GENERATOR_PORT=${GENERATOR_PORT:-1236}
GENERATOR_ENDPOINT=${GENERATOR_ENDPOINT:-http://127.0.0.1:$GENERATOR_PORT}
GENERATOR_CONTAINER=${GENERATOR_CONTAINER:-llama-rocm-10.0}
PYTHON=${PYTHON:-/mnt/data/venv/bin/python3}
P4_RUNNER=$ROOT/run_p4v1.sh
EVAL_LOG=$RUN_DIR/evaluate-resume.log
MASTER_RESTYLE_LOG=$CORPUS/restyle-remaining.log

STAGE=${1:-all}
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --generator-model)
      GENERATOR_MODEL=${2:?--generator-model requires a model name}
      shift 2
      ;;
    --generator-model=*)
      GENERATOR_MODEL=${1#*=}
      shift
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
GENERATOR_LOG=${GENERATOR_LOG:-$CORPUS/restyle-${GENERATOR_MODEL}-server.log}
case "$STAGE" in
  --preflight|evaluate|restyle|all) ;;
  *)
    echo "Usage: $0 [--preflight|evaluate|restyle|all] [--generator-model NAME]" >&2
    exit 2
    ;;
esac

LOCK_DIR=/tmp/deepred-p4v1-cont.lock.d
FROZEN_RUNNER=''
GENERATOR_STARTED=''
cleanup() {
  if [[ -n "$GENERATOR_STARTED" ]]; then
    podman exec "$GENERATOR_CONTAINER" pkill -f \
      "llama-server .*--port $GENERATOR_PORT" >/dev/null 2>&1 || true
  fi
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

resolve_generator() {
  case "$GENERATOR_MODEL" in
    qwen3-30b-a3b)
      GENERATOR_PATH=/mnt/data/models/llm/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf
      GENERATOR_EXTRA=(--chat-template-kwargs '{"enable_thinking":false}')
      ;;
    gpt-oss-120b)
      GENERATOR_PATH=/mnt/data/models/llm/gpt-oss-120b-MXFP4.gguf
      GENERATOR_EXTRA=(--chat-template-kwargs '{"reasoning_effort":"low"}')
      ;;
    glm-4.7-flash)
      GENERATOR_PATH=/mnt/data/models/llm/GLM-4.7-Flash-Q8_0.gguf
      GENERATOR_EXTRA=(--chat-template-kwargs '{"enable_thinking":false}')
      ;;
    qwen2.5-14b-incumbent)
      GENERATOR_PATH=/mnt/data/models/llm/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
      GENERATOR_EXTRA=()
      ;;
    *)
      echo "Unknown generator model: $GENERATOR_MODEL" >&2
      echo "Choose: qwen3-30b-a3b, gpt-oss-120b, glm-4.7-flash, qwen2.5-14b-incumbent" >&2
      exit 2
      ;;
  esac
}

start_generator() {
  local deadline generator_command log_path models
  resolve_generator
  require_file "$GENERATOR_PATH"
  podman container exists "$GENERATOR_CONTAINER" \
    || { echo "Missing container: $GENERATOR_CONTAINER" >&2; exit 1; }
  podman start "$GENERATOR_CONTAINER" >/dev/null
  podman exec "$GENERATOR_CONTAINER" pkill -f \
    "llama-server .*--port $GENERATOR_PORT" >/dev/null 2>&1 || true
  printf '\n== Start generator %s on :%s ==\n' "$GENERATOR_MODEL" "$GENERATOR_PORT"
  printf -v generator_command '%q ' \
    llama-server \
    --model "$GENERATOR_PATH" --alias "$GENERATOR_MODEL" \
    --host 127.0.0.1 --port "$GENERATOR_PORT" \
    --ctx-size 8192 --parallel 1 \
    --n-gpu-layers all --flash-attn on --load-mode none \
    "${GENERATOR_EXTRA[@]}"
  printf -v log_path '%q' "$GENERATOR_LOG"
  podman exec -d -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    "$GENERATOR_CONTAINER" bash -lc \
    "exec $generator_command >>$log_path 2>&1"
  GENERATOR_STARTED=1

  deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if models=$(curl -fsS --max-time 5 "${GENERATOR_ENDPOINT%/}/v1/models" 2>/dev/null) \
      && jq -e --arg expected "$GENERATOR_MODEL" \
        'any(.data[]?; .id == $expected) or
         any(.models[]?; (.model // .name) == $expected)' \
        >/dev/null <<<"$models"; then
      echo "Generator ready: $GENERATOR_MODEL at $GENERATOR_ENDPOINT"
      return
    fi
    sleep 5
  done
  echo "Generator did not become ready within 900s; see $GENERATOR_LOG" >&2
  exit 1
}

preflight_evaluate() {
  require_file "$P4_RUNNER"
  require_file "$RUN_DIR/models.json"
  require_file "$RUN_DIR/system_prompt.txt"
  require_file "$RUN_DIR/with-system/generations.jsonl"
  require_file "$RUN_DIR/no-system/generations.jsonl"
}

preflight_restyle() {
  resolve_generator
  require_file "$PYTHON"
  require_file "$GENERATOR_PATH"
  require_command curl
  require_command jq
  require_command podman
  require_command sort
  require_command comm
  require_command tee

  require_file "$CORPUS/retain/retain.voiced.jsonl"
  require_file "$CORPUS/retain_formats/retain_formats.jsonl"
  require_file "$CORPUS/era_native_formats/era_native_formats.jsonl"
  require_file "$CORPUS/chess/chess.jsonl"
  podman container exists "$GENERATOR_CONTAINER" \
    || { echo "Missing container: $GENERATOR_CONTAINER" >&2; exit 1; }
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
  start_generator
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
