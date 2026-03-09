#!/bin/bash
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-29056}

TASK_NAME=${1:-TurnOffMicrowave}
NUM_TRIALS=${2:-50}
SAVE_DIR=${3:-results/robocasa}

python -m evaluation.robocasa.eval_policy_client \
  --host "${HOST}" \
  --port "${PORT}" \
  --task-name "${TASK_NAME}" \
  --num-trials-per-task "${NUM_TRIALS}" \
  --save-dir "${SAVE_DIR}" \
  --action-map-mode auto
