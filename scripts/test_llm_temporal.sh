#!/bin/bash
# =============================================================================
# Model Comparison: Temporal Classification Benchmark
# =============================================================================
#
# Runs the temporal classification test with 1000 articles on three models
# using the same random seed for reproducible, comparable results.
#
# Output is logged to a timestamped file AND displayed on screen (via tee).
#
# Prerequisites:
#   - All three models downloaded to $DEEPRED_MODELS/llm/
#   - source deepred-env.sh
#   - StrixHalo only (no remote) — REMOTE_HOST is cleared per-run
#
# Usage:
#   sudo bash scripts/test_llm_temporal.sh
#
# =============================================================================

set -euo pipefail

# ── Must run as root (avoids repeated sudo password prompts) ─────────────
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run with sudo (to swap models without repeated password prompts)."
    echo "Usage: sudo -E bash scripts/test_llm_temporal.sh"
    exit 1
fi

SEED=42
N=1000
CONCURRENCY=4
SCRIPT="$DEEPRED_REPO/scripts/test_llm_temporal.py"
QUADLET="/etc/containers/systemd/llama-server-llm.container"

# ── Log file ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$DEEPRED_REPO/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/model_comparison_${TIMESTAMP}.log"

# Redirect all output to both terminal and log file (strip ANSI from log)
exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' > "$LOG_FILE")) 2>&1

# Model definitions: (display_name  model_file  alias)
MODELS=(
    "Qwen 2.5 7B Q4_K_M|qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf|qwen2.5-7b-instruct"
    "Qwen 2.5 14B Q4_K_M|qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf|qwen2.5-14b-instruct"
    "Gemma 2 27B Q4_K_M|gemma-2-27b-it-Q4_K_M.gguf|gemma-2-27b-it"
)

swap_model() {
    local model_file="$1"
    local alias="$2"

    echo "  Updating Quadlet to: $model_file (alias: $alias)"
    sed -i "s|--model /models/llm/[^ ]*|--model /models/llm/$model_file|" "$QUADLET"
    sed -i "s|--alias \"[^\"]*\"|--alias \"$alias\"|" "$QUADLET"
    systemctl daemon-reload
    systemctl restart llama-server-llm

    # Wait for model to load (check /v1/models endpoint)
    echo -n "  Waiting for model to load"
    for i in $(seq 1 120); do
        if curl -sf localhost:1234/v1/models >/dev/null 2>&1; then
            echo " ready! (${i}s)"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " TIMEOUT — model did not load within 120s"
    return 1
}

echo "=============================================================="
echo "  Temporal Classification — Model Comparison"
echo "=============================================================="
echo "  Seed         : $SEED"
echo "  Articles     : $N"
echo "  Concurrency  : $CONCURRENCY"
echo "  Models       : ${#MODELS[@]}"
echo "  Log file     : $LOG_FILE"
echo "=============================================================="
echo ""

for i in "${!MODELS[@]}"; do
    IFS='|' read -r display_name model_file alias <<< "${MODELS[$i]}"

    # Check model file exists
    if [ ! -f "$DEEPRED_MODELS/llm/$model_file" ]; then
        echo "⚠  Skipping $display_name — model file not found: $model_file"
        echo ""
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Model $((i+1))/${#MODELS[@]}: $display_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Swap model
    swap_model "$model_file" "$alias"
    echo ""

    # Run test (no remote — local only)
    # Use the venv python explicitly since we're running as root
    REMOTE_HOST= "$DEEPRED_VENV/bin/python3" "$SCRIPT" \
        -n "$N" \
        --seed "$SEED" \
        --concurrency "$CONCURRENCY" \
        --verbose

    echo ""

    # Pause between models (skip after the last one)
    if [ "$i" -lt $(( ${#MODELS[@]} - 1 )) ]; then
        echo "──────────────────────────────────────────────────────────"
        echo "  Press Enter to continue to the next model, or Ctrl+C to abort..."
        echo "──────────────────────────────────────────────────────────"
        read -r
    fi
done

echo ""
echo "=============================================================="
echo "  All models tested. Compare results above."
echo "=============================================================="
