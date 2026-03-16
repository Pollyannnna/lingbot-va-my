#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 16x4 profiling launcher for RoboCasa training.
# This wraps the normal torchrun entry but changes defaults to make
# time breakdown easier to interpret:
# - run only a small number of steps
# - skip checkpoint save noise
# - disable SwanLab
# - enable stage-level timing in wan_va/train.py

export NNODES="${NNODES:-16}"
export NGPU="${NGPU:-4}"

export ROBOCASA_ENABLE_SWANLAB="${ROBOCASA_ENABLE_SWANLAB:-0}"
export ROBOCASA_NUM_STEPS="${ROBOCASA_NUM_STEPS:-12}"
export ROBOCASA_SAVE_INTERVAL="${ROBOCASA_SAVE_INTERVAL:-1000000}"
export ROBOCASA_GC_INTERVAL="${ROBOCASA_GC_INTERVAL:-1000000}"

export ROBOCASA_PROFILE_START_STEP="${ROBOCASA_PROFILE_START_STEP:-2}"
export ROBOCASA_PROFILE_STEPS="${ROBOCASA_PROFILE_STEPS:-8}"
export ROBOCASA_PROFILE_CUDA_SYNC="${ROBOCASA_PROFILE_CUDA_SYNC:-1}"

export SAVE_ROOT="${SAVE_ROOT:-/data/share/lijiang/yzy-exp/robocasa365-human50_profile_16x4}"

echo "[INFO] Profiling config:"
echo "       cluster=${NNODES}x${NGPU}"
echo "       steps=${ROBOCASA_NUM_STEPS}"
echo "       profile_start_step=${ROBOCASA_PROFILE_START_STEP}"
echo "       profile_steps=${ROBOCASA_PROFILE_STEPS}"
echo "       save_root=${SAVE_ROOT}"
echo "       notes=checkpoint_and_gc_are_disabled_for_cleaner_step_timing"

exec bash "${REPO_ROOT}/script/submit_prepare_train_robocasa_16x4.sh" "$@"
