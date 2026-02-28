#!/bin/bash
# DeepRedAI Environment Configuration
# ====================================
#
# Source this file to enter DeepRedAI development mode:
#
#   source /path/to/DeepRedAI/deepred-env.sh
#
# To auto-load on every login, add to ~/.bashrc:
#
#   export DEEPRED_ROOT="/mnt/data"  # adjust to your data disk mount point
#   [ -f "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh" ] && source "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh"
#
# All variables use the ${VAR:-default} pattern so you can override any of
# them before sourcing this file, or export them in ~/.bashrc above the
# source line.

# ── Root paths ───────────────────────────────────────────────────────────
# DEEPRED_ROOT  : Top-level data directory (the data-disk mount point).
#                 Every other path is derived from this unless overridden.
# DEEPRED_REPO  : Location of the DeepRedAI git clone.

export DEEPRED_ROOT="${DEEPRED_ROOT:-/mnt/data}"
export DEEPRED_REPO="${DEEPRED_REPO:-$DEEPRED_ROOT/DeepRedAI}"

# ── Data directories ─────────────────────────────────────────────────────
export WIKI_DATA="${WIKI_DATA:-$DEEPRED_ROOT/wikipedia}"
export GUTENBERG_DATA="${GUTENBERG_DATA:-$DEEPRED_ROOT/gutenberg}"
export DEEPRED_MODELS="${DEEPRED_MODELS:-$DEEPRED_ROOT/models}"
export DEEPRED_VENV="${DEEPRED_VENV:-$DEEPRED_ROOT/venv}"

# ── Service endpoints ────────────────────────────────────────────────────
# Override these when services run on a different host or port.
export LMSTUDIO_HOST="${LMSTUDIO_HOST:-localhost}"
export LMSTUDIO_PORT="${LMSTUDIO_PORT:-1234}"
export EMBEDDING_PORT="${EMBEDDING_PORT:-1235}"
export PG_HOST="${PG_HOST:-localhost}"
export PG_PORT="${PG_PORT:-5432}"
export OS_HOST="${OS_HOST:-localhost}"
export OS_PORT="${OS_PORT:-9200}"

# ── Convenience: activate the Python venv ────────────────────────────────
if [ -f "$DEEPRED_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$DEEPRED_VENV/bin/activate"
fi

# ── Convenience: add scripts to PATH ────────────────────────────────────
case ":$PATH:" in
    *":$DEEPRED_REPO/scripts:"*) ;;   # already present
    *) export PATH="$DEEPRED_REPO/scripts:$PATH" ;;
esac

# ── Summary ──────────────────────────────────────────────────────────────
echo "DeepRedAI environment loaded"
echo "  DEEPRED_ROOT   = $DEEPRED_ROOT"
echo "  DEEPRED_REPO   = $DEEPRED_REPO"
echo "  WIKI_DATA      = $WIKI_DATA"
echo "  GUTENBERG_DATA = $GUTENBERG_DATA"
echo "  DEEPRED_MODELS = $DEEPRED_MODELS"
echo "  DEEPRED_VENV   = $DEEPRED_VENV"
