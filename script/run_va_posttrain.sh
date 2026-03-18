#!/usr/bin/env bash
set -euo pipefail

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
NGPU="${NGPU:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-robotwin_train}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

if [[ -z "${ROBOTWIN_DATASET_PATH:-}" ]]; then
  echo "[WARN] ROBOTWIN_DATASET_PATH is not set. Config default path will be used."
elif [[ ! -d "${ROBOTWIN_DATASET_PATH}" ]]; then
  echo "[ERROR] ROBOTWIN_DATASET_PATH does not exist: ${ROBOTWIN_DATASET_PATH}"
  exit 1
fi

if [[ -n "${ROBOTWIN_DATASET_PATH:-}" && -z "${ROBOTWIN_EMPTY_EMB_PATH:-}" ]]; then
  export ROBOTWIN_EMPTY_EMB_PATH="${ROBOTWIN_DATASET_PATH}/empty_emb.pt"
fi

if [[ -n "${ROBOTWIN_EMPTY_EMB_PATH:-}" && ! -f "${ROBOTWIN_EMPTY_EMB_PATH}" ]]; then
  echo "[WARN] ROBOTWIN_EMPTY_EMB_PATH not found: ${ROBOTWIN_EMPTY_EMB_PATH}"
fi

overrides=("$@")

export ROBOTWIN_ENABLE_SWANLAB="${ROBOTWIN_ENABLE_SWANLAB:-1}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-fViowdx9CQvjV7ofiP6ET}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-Yeziyang}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-Lingbot-VA-Robotwin}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun_cmd=(
  "${PYTHON_BIN}"
  -m
  torch.distributed.run
  "--nproc_per_node=${NGPU}"
  "--local-ranks-filter=${LOG_RANK}"
  "--master_port"
  "${MASTER_PORT}"
  "--tee"
  "3"
)

if (( NNODES > 1 )); then
  torchrun_cmd+=(
    "--nnodes=${NNODES}"
    "--node_rank=${NODE_RANK}"
    "--master_addr=${MASTER_ADDR}"
  )
fi

torchrun_cmd+=(
  -m
  wan_va.train
  --config-name
  "${CONFIG_NAME}"
)

if (( ${#overrides[@]} > 0 )); then
  torchrun_cmd+=("${overrides[@]}")
fi

printf 'Launching training:'
printf ' %q' "${torchrun_cmd[@]}"
printf '\n'

cd "${REPO_ROOT}"

PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
"${torchrun_cmd[@]}"
