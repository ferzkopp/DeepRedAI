#!/usr/bin/env bash
# Phase 4 conditioning pilot: unchanged p4-v3 corpus, more plain-context rows.
#
# Start in a separate terminal:
#   cd /mnt/data/DeepRedAI
#   ./run_p4v4.sh --preflight
#   ./run_p4v4.sh all
#
# Stages: --preflight, dataset, train, evaluate, diagnose, all (default).
# Training auto-resumes from the latest full checkpoint. Snapshots at
# 10/25/50/75/100% are steps 100/250/500/750/1000 at the default limit.
# `all` also exports Q8_0 and evaluates both banks with and without a system
# prompt; the roughly 9 h estimate covers training only.
set -Eeuo pipefail

ROOT=/mnt/data/DeepRedAI
export LABEL=${LABEL:-p4-v4}
export MODEL_TAG=${MODEL_TAG:-p4v4}
export CORPUS=${CORPUS:-/mnt/data/deepred_corpus/p4-v3}
export DATASET=${DATASET:-/mnt/data/sft_corpus/deepred-p4v4}
export TRAIN_DIR=${TRAIN_DIR:-/mnt/data/training_output/deepred-p4v4}
export EPOCHS=${EPOCHS:-1}
export MAX_STEPS=${MAX_STEPS:-1000}

# Coverage here is the fraction WITH a system prompt, not plain-context share.
export SYSTEM_COVERAGE=${SYSTEM_COVERAGE:-0.85}
export SYSTEM_COVERAGE_BY_KIND=${SYSTEM_COVERAGE_BY_KIND:-"era_native=0.50 era_native_explanatory=0.50 era_native_formats=0.50"}

# Keep the p4-v3 caps, marker rates, base model, and training settings.
export INJECT_ERA_FORMATS=${INJECT_ERA_FORMATS:-0}
export INJECT_RETAIN_FORMATS=${INJECT_RETAIN_FORMATS:-0}
export INJECT_CHESS=${INJECT_CHESS:-0}

export COMPARATORS=${COMPARATORS:-"deepred-p3v2-050-q8=/mnt/data/evaluations/deepred-1969/p3v2-2026-09-04/models.json deepred-p3v4c-100-q8=/mnt/data/evaluations/deepred-1969/p3v4c-2026-09-06/models.json deepred-p4v2-100-q8=/mnt/data/evaluations/deepred-1969/p4v2-2026-10-07/models.json deepred-p4v3-075-q8=/mnt/data/evaluations/deepred-1969/p4v3-2026-10-28/models.json deepred-p4v3-100-q8=/mnt/data/evaluations/deepred-1969/p4v3-2026-10-28/models.json"}

exec "$ROOT/run_p4v1.sh" "$@"
