#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATASET_ROOT=/path/to/robocasa_dataset_root \
#   MODEL_ROOT=/path/to/wan22_or_lingbot_model \
#   bash script/prepare_robocasa_for_va.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/your/robocasa_lerobot_dataset}"
MODEL_ROOT="${MODEL_ROOT:-/data/share/lijiang/ckpt/lingbot-va-posttrain-robotwin}"
STATS_OUTPUT="${STATS_OUTPUT:-${DATASET_ROOT}/robocasa_action_stats_for_lingbotva.json}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-${DATASET_ROOT}/empty_emb.pt}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-2000}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERROR] DATASET_ROOT does not exist: ${DATASET_ROOT}"
  exit 1
fi

cd "${REPO_ROOT}"

echo "[1/3] Ensure action_config exists in episodes.jsonl"
"${PYTHON_BIN}" script/add_action_config.py \
  --dataset-root "${DATASET_ROOT}" \
  --normalize-style \
  --write-episodes-ori

echo "[2/3] Compute RoboCasa action quantiles"
"${PYTHON_BIN}" script/compute_robocasa_norm_stats.py \
  --dataset-root "${DATASET_ROOT}" \
  --output "${STATS_OUTPUT}" \
  --action-indices 5,6,7,8,9,10,11 \
  --target-action-dim 30 \
  --max-rows-per-file "${MAX_ROWS_PER_FILE}"

echo "[3/3] Generate empty prompt embedding"
"${PYTHON_BIN}" script/make_empty_emb.py \
  --model-root "${MODEL_ROOT}" \
  --output "${EMPTY_EMB_PATH}" \
  --overwrite

echo ""
echo "Preparation finished."
echo "Recommended env vars:"
echo "  export ROBOCASA_DATASET_PATH=\"${DATASET_ROOT}\""
echo "  export ROBOCASA_EMPTY_EMB_PATH=\"${EMPTY_EMB_PATH}\""
echo "  export ROBOCASA_NORM_STATS_PATH=\"${STATS_OUTPUT}\""
