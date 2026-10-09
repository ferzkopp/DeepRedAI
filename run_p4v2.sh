#!/usr/bin/env bash
# Phase 4 corrected-tokenizer pilot using the versioned voiced p4-v2 corpus.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
export LABEL=${LABEL:-p4-v2}
export MODEL_TAG=${MODEL_TAG:-p4v2}
export CORPUS=${CORPUS:-/mnt/data/deepred_corpus/p4-v2}
export DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-p4v2}
export TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-p4v2}
export EPOCHS=${EPOCHS:-1}

# These assets already carry Qwen3 voice markers. Additional injection produced
# a 1.49:1 marker ratio; preserving their native markers yields 1.10:1.
export INJECT_ERA_FORMATS=${INJECT_ERA_FORMATS:-0}
export INJECT_RETAIN_FORMATS=${INJECT_RETAIN_FORMATS:-0}
export INJECT_CHESS=${INJECT_CHESS:-0}

exec "$ROOT/run_p4v1.sh" "$@"