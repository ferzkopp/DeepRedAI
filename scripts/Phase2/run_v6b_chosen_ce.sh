#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
export VARIANT=v6b
export CHOSEN_CE_WEIGHT=0.5
export REUSE_PAIR_DATASET=1
export PAIR_DIR=/mnt/data/sft_corpus/deepred-v5a-pairwise
export TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-v6b-chosen-ce}
export RUN_DIR=${RUN_DIR:-/mnt/data/evaluations/deepred-1969/v6b-chosen-ce-$(date +%Y-%m-%d)}

exec "$ROOT/scripts/Phase2/run_v5_pairwise.sh" "$@"