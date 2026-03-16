#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Single-node 4-GPU profiling launcher for RoboCasa training.
# Use this to compare against 16x4 and isolate multi-node communication cost.

export NNODES=1
export NGPU=4
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export ROBOCASA_GRAD_ACC="${ROBOCASA_GRAD_ACC:-1}"
export ROBOCASA_ENABLE_SWANLAB="${ROBOCASA_ENABLE_SWANLAB:-0}"
export ROBOCASA_NUM_STEPS="${ROBOCASA_NUM_STEPS:-1000}"
export ROBOCASA_SAVE_INTERVAL="${ROBOCASA_SAVE_INTERVAL:-1000000}"
export ROBOCASA_GC_INTERVAL="${ROBOCASA_GC_INTERVAL:-1000000}"

export ROBOCASA_PROFILE_START_STEP="${ROBOCASA_PROFILE_START_STEP:-2}"
export ROBOCASA_PROFILE_STEPS="${ROBOCASA_PROFILE_STEPS:-8}"
export ROBOCASA_PROFILE_CUDA_SYNC="${ROBOCASA_PROFILE_CUDA_SYNC:-1}"

export SAVE_ROOT="${SAVE_ROOT:-/data/share/lijiang/yzy-exp/robocasa365-human50_profile_1x4}"

echo "[INFO] Profiling config:"
echo "       cluster=${NNODES}x${NGPU}"
echo "       grad_acc=${ROBOCASA_GRAD_ACC}"
echo "       steps=${ROBOCASA_NUM_STEPS}"
echo "       profile_start_step=${ROBOCASA_PROFILE_START_STEP}"
echo "       profile_steps=${ROBOCASA_PROFILE_STEPS}"
echo "       save_root=${SAVE_ROOT}"
echo "       notes=compare_this_against_16x4_to_measure_multi_node_overhead"

exec bash "${REPO_ROOT}/script/submit_prepare_train_robocasa_16x4.sh" "$@"
