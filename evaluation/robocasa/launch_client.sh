#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_VA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${LINGBOT_VA_ROOT}/.." && pwd)"
ROBOCASA_REPO_ROOT="${ROBOCASA_REPO_ROOT:-${WORKSPACE_ROOT}/robocasa}"
ROBOSUITE_REPO_ROOT="${ROBOSUITE_REPO_ROOT:-${WORKSPACE_ROOT}/robosuite}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${LINGBOT_VA_ROOT}:${ROBOCASA_REPO_ROOT}:${ROBOSUITE_REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${LINGBOT_VA_ROOT}:${ROBOCASA_REPO_ROOT}:${ROBOSUITE_REPO_ROOT}"
fi

PYTHON_BIN=${PYTHON_BIN:-python}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-29056}
DATASET_ROOT=${DATASET_ROOT:-}
TASK_SET=${TASK_SET:-}
OBJ_INSTANCE_SPLIT=${OBJ_INSTANCE_SPLIT:-target}
ACTION_MAP_MODE=${ACTION_MAP_MODE:-auto}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-0}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-0}

TASK_NAME=${1:-all}
NUM_TRIALS=${2:-50}
SAVE_DIR=${3:-results/robocasa}
shift_args=()
if (( $# > 3 )); then
  shift_args=("${@:4}")
fi

extra_args=()
if [[ -n "${DATASET_ROOT}" ]]; then
  extra_args+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${TASK_SET}" ]]; then
  extra_args+=(--task-set "${TASK_SET}")
fi
if [[ -n "${OBJ_INSTANCE_SPLIT}" ]]; then
  extra_args+=(--obj-instance-split "${OBJ_INSTANCE_SPLIT}")
fi
if [[ "${SAVE_ROLLOUT_VIDEO}" == "1" ]]; then
  extra_args+=(--save-rollout-video)
fi
if [[ "${SAVE_VISUALIZATION}" == "1" ]]; then
  extra_args+=(--save-visualization)
fi

"${PYTHON_BIN}" -m evaluation.robocasa.eval_policy_client \
  --host "${HOST}" \
  --port "${PORT}" \
  --task-name "${TASK_NAME}" \
  --num-trials-per-task "${NUM_TRIALS}" \
  --save-dir "${SAVE_DIR}" \
  --action-map-mode "${ACTION_MAP_MODE}" \
  "${extra_args[@]}" \
  "${shift_args[@]}"
