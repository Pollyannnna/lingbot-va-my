#!/usr/bin/env bash
set -euo pipefail

# Example:
#   export ROBOCASA_DATASET_PATH=/path/to/robocasa_dataset_root
#   export ROBOCASA_EMPTY_EMB_PATH=${ROBOCASA_DATASET_PATH}/empty_emb.pt
#   export ROBOCASA_NORM_STATS_PATH=${ROBOCASA_DATASET_PATH}/robocasa_action_stats_for_lingbotva.json
#   NGPU=8 bash script/run_va_posttrain_robocasa.sh

umask 007

NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29521}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-robocasa_train}"

if [[ -z "${ROBOCASA_DATASET_PATH:-}" ]]; then
  echo "[WARN] ROBOCASA_DATASET_PATH is not set. Config default path will be used."
else
  if [[ ! -d "${ROBOCASA_DATASET_PATH}" ]]; then
    echo "[ERROR] ROBOCASA_DATASET_PATH does not exist: ${ROBOCASA_DATASET_PATH}"
    exit 1
  fi
  if [[ -z "${ROBOCASA_EMPTY_EMB_PATH:-}" ]]; then
    export ROBOCASA_EMPTY_EMB_PATH="${ROBOCASA_DATASET_PATH}/empty_emb.pt"
  fi
fi

if [[ -n "${ROBOCASA_EMPTY_EMB_PATH:-}" && ! -f "${ROBOCASA_EMPTY_EMB_PATH}" ]]; then
  echo "[WARN] ROBOCASA_EMPTY_EMB_PATH not found: ${ROBOCASA_EMPTY_EMB_PATH}"
fi
if [[ -n "${ROBOCASA_NORM_STATS_PATH:-}" && ! -f "${ROBOCASA_NORM_STATS_PATH}" ]]; then
  echo "[WARN] ROBOCASA_NORM_STATS_PATH not found: ${ROBOCASA_NORM_STATS_PATH}"
fi

if [[ $# -ne 0 ]]; then
  overrides="$*"
else
  overrides=""
fi

export ROBOCASA_ENABLE_SWANLAB="${ROBOCASA_ENABLE_SWANLAB:-1}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-Yeziyang}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-va_robocasa}"
export TOKENIZERS_PARALLELISM=false

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
python -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --local-ranks-filter="${LOG_RANK}" \
  --master_port "${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train --config-name "${CONFIG_NAME}" ${overrides}
