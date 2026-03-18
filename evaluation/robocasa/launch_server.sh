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
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
CONFIG_NAME=${CONFIG_NAME:-robocasa}
SAVE_ROOT=${SAVE_ROOT:-./results/robocasa_server}
NGPU=${NGPU:-1}

mkdir -p "${SAVE_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NGPU}" \
  --master_port "${MASTER_PORT}" \
  "${LINGBOT_VA_ROOT}/wan_va/wan_va_server.py" \
  --config-name "${CONFIG_NAME}" \
  --port "${START_PORT}" \
  --save_root "${SAVE_ROOT}"
