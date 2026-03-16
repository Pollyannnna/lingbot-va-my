#!/usr/bin/env bash
set -euo pipefail

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Historical name only. This script is now a pure torchrun launcher and does
# not call sbatch/srun. Dataset preparation and latent extraction must already
# be finished before running it.

DEFAULT_ENV_PYTHON="/root/miniconda3/envs/lingbotva/bin/python"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_ENV_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_ENV_PYTHON}"
  else
    PYTHON_BIN="python"
  fi
else
  PYTHON_BIN="${PYTHON_BIN}"
fi
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"

NGPU="${NGPU:-4}"
NNODES="${NNODES:-16}"
NODE_RANK="${NODE_RANK:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29521}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-robocasa_train}"

DATASET_ROOT="${DATASET_ROOT:-/data/share/lijiang/data/robocasa365/target-human-50}"
MODEL_ROOT="${MODEL_ROOT:-/data/share/lijiang/ckpt/lingbot-va-base}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-${DATASET_ROOT}/empty_emb.pt}"
STATS_OUTPUT="${STATS_OUTPUT:-${DATASET_ROOT}/robocasa_action_stats_for_lingbotva.json}"
SAVE_ROOT="${SAVE_ROOT:-/data/share/lijiang/yzy-exp/robocasa365-human50_bs64}"

ROBOCASA_ENABLE_SWANLAB="${ROBOCASA_ENABLE_SWANLAB:-1}"
ROBOCASA_NUM_STEPS="${ROBOCASA_NUM_STEPS:-10000}"
ROBOCASA_BATCH_SIZE="${ROBOCASA_BATCH_SIZE:-1}"
ROBOCASA_GRAD_ACC="${ROBOCASA_GRAD_ACC:-1}"
ROBOCASA_LR="${ROBOCASA_LR:-1e-5}"
ROBOCASA_WARMUP_STEPS="${ROBOCASA_WARMUP_STEPS:-10}"
ROBOCASA_BETA1="${ROBOCASA_BETA1:-0.9}"
ROBOCASA_BETA2="${ROBOCASA_BETA2:-0.95}"
ROBOCASA_WEIGHT_DECAY="${ROBOCASA_WEIGHT_DECAY:-0.1}"
ROBOCASA_SAVE_INTERVAL="${ROBOCASA_SAVE_INTERVAL:-1000}"
ROBOCASA_GC_INTERVAL="${ROBOCASA_GC_INTERVAL:-50}"
ROBOCASA_CFG_PROB="${ROBOCASA_CFG_PROB:-0.1}"
ROBOCASA_LOAD_WORKER="${ROBOCASA_LOAD_WORKER:-4}"
ROBOCASA_DATASET_INIT_WORKER="${ROBOCASA_DATASET_INIT_WORKER:-4}"
ROBOCASA_DATALOADER_PREFETCH="${ROBOCASA_DATALOADER_PREFETCH:-4}"
ROBOCASA_PIN_MEMORY="${ROBOCASA_PIN_MEMORY:-1}"
ROBOCASA_PERSISTENT_WORKERS="${ROBOCASA_PERSISTENT_WORKERS:-1}"
ROBOCASA_RESUME_FROM="${ROBOCASA_RESUME_FROM:-}"

SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-Yeziyang}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-va_robocasa_target-human-50}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-fViowdx9CQvjV7ofiP6ET}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
USE_RDMA="${USE_RDMA:-1}"
REQUIRE_RDMA="${REQUIRE_RDMA:-0}"
AUTO_INSTALL_RDMA_DEPS="${AUTO_INSTALL_RDMA_DEPS:-1}"
RDMA_CREATE_DEVNODES="${RDMA_CREATE_DEVNODES:-1}"

usage() {
  cat <<EOF
Usage:
  bash script/submit_prepare_train_robocasa_16x4.sh

This is a pure torchrun wrapper with 16x4 defaults.

Common overrides:
  SAVE_ROOT=/path/to/output
  ROBOCASA_RESUME_FROM=/path/to/checkpoint_step_xxx
  NNODES=16
  NGPU=4
  MASTER_PORT=29521

Validation:
  bash script/submit_prepare_train_robocasa_16x4.sh --dry-run
EOF
}

autodetect_dist_env() {
  local host_name=""

  host_name="${HOSTNAME:-$(hostname 2>/dev/null || true)}"

  if [[ -z "${NODE_RANK}" ]]; then
    if [[ -n "${GROUP_RANK:-}" ]]; then
      NODE_RANK="${GROUP_RANK}"
    elif [[ -n "${NODE_RANK_ENV:-}" ]]; then
      NODE_RANK="${NODE_RANK_ENV}"
    elif [[ -n "${RANK:-}" && -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" == "${NNODES}" ]]; then
      NODE_RANK="${RANK}"
    elif [[ "${host_name}" =~ -worker-([0-9]+)$ ]]; then
      NODE_RANK="${BASH_REMATCH[1]}"
    elif (( NNODES == 1 )); then
      NODE_RANK="0"
    fi
  fi

  if [[ -z "${MASTER_ADDR}" ]]; then
    if [[ -n "${PET_MASTER_ADDR:-}" ]]; then
      MASTER_ADDR="${PET_MASTER_ADDR}"
    elif [[ -n "${MASTER_ADDR_ENV:-}" ]]; then
      MASTER_ADDR="${MASTER_ADDR_ENV}"
    elif [[ "${host_name}" =~ ^(.*)-worker-[0-9]+$ ]]; then
      MASTER_ADDR="${BASH_REMATCH[1]}-worker-0"
    elif (( NNODES == 1 )); then
      MASTER_ADDR="127.0.0.1"
    fi
  fi

  if [[ -z "${NODE_RANK}" ]]; then
    NODE_RANK="0"
  fi

  if [[ -z "${MASTER_ADDR}" ]]; then
    MASTER_ADDR="127.0.0.1"
  fi
}

resolve_python() {
  local conda_base=""

  if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || {
      echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}"
      exit 1
    }
    return
  fi

  if [[ -z "${CONDA_ENV_PREFIX}" && -z "${CONDA_ENV_NAME}" ]]; then
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
      return
    fi
    echo "[ERROR] Failed to resolve PYTHON_BIN=${PYTHON_BIN}"
    exit 1
  fi

  if [[ -n "${CONDA_ENV_PREFIX}" || -n "${CONDA_ENV_NAME}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      conda_base="$(conda info --base)"
    elif [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
      conda_base="/root/miniconda3"
    else
      echo "[ERROR] conda not found. Set CONDA_ENV_PREFIX or PYTHON_BIN explicitly."
      exit 1
    fi

    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"

    if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
      conda activate "${CONDA_ENV_PREFIX}"
    else
      conda activate "${CONDA_ENV_NAME}"
    fi
  fi

  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
}

install_rdma_userspace() {
  local missing_ibverbs=0
  local missing_tool=0

  if [[ "${USE_RDMA}" == "0" || "${AUTO_INSTALL_RDMA_DEPS}" == "0" ]]; then
    return
  fi

  if ! "${PYTHON_BIN:-python}" - <<'PY' >/dev/null 2>&1
import ctypes.util, sys
sys.exit(0 if ctypes.util.find_library("ibverbs") else 1)
PY
  then
    missing_ibverbs=1
  fi

  if ! command -v ibv_devinfo >/dev/null 2>&1; then
    missing_tool=1
  fi

  if (( missing_ibverbs == 0 && missing_tool == 0 )); then
    return
  fi

  if [[ "$(id -u)" != "0" || ! -x "$(command -v apt-get 2>/dev/null || true)" ]]; then
    echo "[WARN] RDMA userspace packages appear incomplete, but auto-install is unavailable."
    echo "       need_libibverbs=${missing_ibverbs} need_ibv_devinfo=${missing_tool}"
    return
  fi

  echo "[INFO] Installing RDMA userspace packages via apt-get ..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    librdmacm-dev \
    libibverbs-dev \
    libibverbs1 \
    ibverbs-providers \
    rdma-core \
    infiniband-diags \
    ibverbs-utils
  ldconfig || true
}

detect_active_ib_hcas() {
  local dev
  local state_file
  local layer_file
  local state
  local layer
  local -a hcas=()

  for dev in /sys/class/infiniband/*; do
    [[ -d "${dev}" ]] || continue
    state_file="${dev}/ports/1/state"
    layer_file="${dev}/ports/1/link_layer"
    [[ -f "${state_file}" && -f "${layer_file}" ]] || continue

    state="$(<"${state_file}")"
    layer="$(<"${layer_file}")"

    [[ "${state}" == *"ACTIVE"* ]] || continue
    [[ "${layer}" == "InfiniBand" ]] || continue
    hcas+=("${dev##*/}")
  done

  if (( ${#hcas[@]} > 0 )); then
    IFS=,
    echo "${hcas[*]}"
  fi
}

ensure_infiniband_devfs() {
  local dev_path="$1"
  local dev_num="$2"
  local node_type="$3"
  local major_num
  local minor_num

  [[ "${RDMA_CREATE_DEVNODES}" == "1" ]] || return 0
  [[ -n "${dev_num}" ]] || return 0

  major_num="${dev_num%%:*}"
  minor_num="${dev_num##*:}"

  mkdir -p /dev/infiniband
  [[ -e "${dev_path}" ]] || mknod "${dev_path}" "${node_type}" "${major_num}" "${minor_num}" || true
  chmod 666 "${dev_path}" 2>/dev/null || true
}

populate_infiniband_devfs() {
  local sys_dev
  local dev_num
  local name

  [[ "${USE_RDMA}" == "1" ]] || return

  if [[ -f /sys/class/misc/rdma_cm/dev ]]; then
    dev_num="$(< /sys/class/misc/rdma_cm/dev)"
    ensure_infiniband_devfs "/dev/infiniband/rdma_cm" "${dev_num}" c
  fi

  for sys_dev in /sys/class/infiniband_verbs/*/dev; do
    [[ -f "${sys_dev}" ]] || continue
    dev_num="$(< "${sys_dev}")"
    name="$(basename "$(dirname "${sys_dev}")")"
    ensure_infiniband_devfs "/dev/infiniband/${name}" "${dev_num}" c
  done

  for sys_dev in /sys/class/infiniband_mad/*/dev; do
    [[ -f "${sys_dev}" ]] || continue
    dev_num="$(< "${sys_dev}")"
    name="$(basename "$(dirname "${sys_dev}")")"
    ensure_infiniband_devfs "/dev/infiniband/${name}" "${dev_num}" c
  done
}

log_cuda_env() {
  if command -v nvcc >/dev/null 2>&1; then
    echo "[INFO] nvcc -V"
    nvcc -V | tail -n 1
  else
    echo "[WARN] nvcc not found in PATH"
  fi
}

check_rdma_runtime() {
  local has_ibverbs=0
  local has_devfs=0
  local has_ibv_devinfo=0
  local has_rdma_access=0

  if "${PYTHON_BIN:-python}" - <<'PY' >/dev/null 2>&1
import ctypes.util, sys
sys.exit(0 if ctypes.util.find_library("ibverbs") else 1)
PY
  then
    has_ibverbs=1
  fi

  if [[ -d "/dev/infiniband" ]] && compgen -G "/dev/infiniband/uverbs*" >/dev/null; then
    has_devfs=1
  fi

  if command -v ibv_devinfo >/dev/null 2>&1; then
    has_ibv_devinfo=1
    if ibv_devinfo -l >/dev/null 2>&1; then
      has_rdma_access=1
    fi
  fi

  if (( has_ibverbs == 1 && has_devfs == 1 && has_ibv_devinfo == 1 && has_rdma_access == 1 )); then
    echo "[INFO] ibv_devinfo succeeded; RDMA userspace and device access look healthy."
    return 0
  fi

  if [[ "${REQUIRE_RDMA}" == "1" ]]; then
    echo "[ERROR] RDMA is required, but the runtime environment is incomplete."
    echo "        libibverbs_present=${has_ibverbs} /dev_infiniband_present=${has_devfs} ibv_devinfo_present=${has_ibv_devinfo} rdma_device_access=${has_rdma_access}"
    echo "        The container must include RDMA libraries and expose /dev/infiniband."
    exit 1
  fi

  echo "[WARN] RDMA requested, but the runtime environment is incomplete."
  echo "       libibverbs_present=${has_ibverbs} /dev_infiniband_present=${has_devfs} ibv_devinfo_present=${has_ibv_devinfo} rdma_device_access=${has_rdma_access}"
  echo "       NCCL will likely fall back to Socket/TCP."
  return 1
}

setup_rdma_env() {
  local detected_hcas=""

  if [[ "${USE_RDMA}" == "0" ]]; then
    echo "[INFO] USE_RDMA=0, leaving NCCL transport settings unchanged."
    return
  fi

  install_rdma_userspace
  populate_infiniband_devfs

  if ! check_rdma_runtime; then
    return
  fi

  detected_hcas="$(detect_active_ib_hcas)"
  if [[ -z "${detected_hcas}" ]]; then
    echo "[WARN] No active InfiniBand HCAs detected under /sys/class/infiniband."
    echo "[WARN] Continuing without forcing NCCL RDMA."
    return
  fi

  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-${detected_hcas}}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
  export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

  echo "[INFO] RDMA enabled for NCCL"
  echo "       NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  echo "       NCCL_IB_HCA=${NCCL_IB_HCA}"
  echo "       NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  echo "       NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
}

validate_config() {
  [[ -d "${DATASET_ROOT}" ]] || {
    echo "[ERROR] DATASET_ROOT does not exist: ${DATASET_ROOT}"
    exit 1
  }
  [[ -d "${MODEL_ROOT}" ]] || {
    echo "[ERROR] MODEL_ROOT does not exist: ${MODEL_ROOT}"
    exit 1
  }
  [[ -f "${EMPTY_EMB_PATH}" ]] || {
    echo "[ERROR] EMPTY_EMB_PATH does not exist: ${EMPTY_EMB_PATH}"
    exit 1
  }
  [[ -f "${STATS_OUTPUT}" ]] || {
    echo "[ERROR] STATS_OUTPUT does not exist: ${STATS_OUTPUT}"
    exit 1
  }
  if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "[ERROR] Failed to infer a usable MASTER_ADDR for multi-node torchrun."
    echo "        Current host: ${HOSTNAME:-$(hostname 2>/dev/null || echo unknown)}"
    echo "        Please set MASTER_ADDR explicitly to the rank0 host."
    exit 1
  fi
  mkdir -p "${SAVE_ROOT}"
}

print_summary() {
  local global_batch
  global_batch=$((NNODES * NGPU * ROBOCASA_BATCH_SIZE * ROBOCASA_GRAD_ACC))

  echo "[INFO] Torchrun config:"
  echo "       cluster=${NNODES}x${NGPU}"
  echo "       node_rank=${NODE_RANK}"
  echo "       master_addr=${MASTER_ADDR}"
  echo "       master_port=${MASTER_PORT}"
  echo "       dataset=${DATASET_ROOT}"
  echo "       model=${MODEL_ROOT}"
  echo "       save_root=${SAVE_ROOT}"
  echo "       steps=${ROBOCASA_NUM_STEPS}"
  echo "       lr=${ROBOCASA_LR}"
  echo "       per_gpu_batch=${ROBOCASA_BATCH_SIZE}"
  echo "       grad_acc=${ROBOCASA_GRAD_ACC}"
  echo "       global_batch=${global_batch}"
  echo "       load_worker=${ROBOCASA_LOAD_WORKER}"
}

main() {
  local -a overrides
  local -a cmd
  local has_save_root=0
  local dry_run=0

  while (( $# > 0 )); do
    case "${1}" in
      -h|--help)
        usage
        exit 0
        ;;
      --dry-run)
        dry_run=1
        shift
        ;;
      *)
        break
        ;;
    esac
  done

  autodetect_dist_env
  resolve_python
  log_cuda_env
  validate_config
  setup_rdma_env
  print_summary

  export PYTHON_BIN NGPU NNODES NODE_RANK MASTER_ADDR MASTER_PORT LOG_RANK
  export TORCHFT_LIGHTHOUSE CONFIG_NAME
  export OMP_NUM_THREADS MKL_NUM_THREADS

  export ROBOCASA_DATASET_PATH="${DATASET_ROOT}"
  export ROBOCASA_EMPTY_EMB_PATH="${EMPTY_EMB_PATH}"
  export ROBOCASA_NORM_STATS_PATH="${STATS_OUTPUT}"
  export ROBOCASA_PRETRAINED_MODEL="${MODEL_ROOT}"

  export ROBOCASA_ENABLE_SWANLAB ROBOCASA_NUM_STEPS ROBOCASA_BATCH_SIZE
  export ROBOCASA_GRAD_ACC ROBOCASA_LR ROBOCASA_WARMUP_STEPS
  export ROBOCASA_BETA1 ROBOCASA_BETA2 ROBOCASA_WEIGHT_DECAY
  export ROBOCASA_SAVE_INTERVAL ROBOCASA_GC_INTERVAL ROBOCASA_CFG_PROB
  export ROBOCASA_LOAD_WORKER ROBOCASA_DATASET_INIT_WORKER ROBOCASA_DATALOADER_PREFETCH
  export ROBOCASA_PIN_MEMORY ROBOCASA_PERSISTENT_WORKERS ROBOCASA_RESUME_FROM

  export SWANLAB_WORKSPACE SWANLAB_PROJECT SWANLAB_API_KEY

  overrides=("$@")
  for arg in "${overrides[@]}"; do
    if [[ "${arg}" == "--save-root" ]]; then
      has_save_root=1
      break
    fi
  done

  cmd=(
    bash
    "${REPO_ROOT}/script/run_va_posttrain_robocasa.sh"
  )
  if (( has_save_root == 0 )); then
    cmd+=(--save-root "${SAVE_ROOT}")
  fi
  if (( ${#overrides[@]} > 0 )); then
    cmd+=("${overrides[@]}")
  fi

  printf 'Launching torchrun wrapper:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if (( dry_run == 1 )); then
    echo "[INFO] Dry run only; command was validated but not executed."
    exit 0
  fi

  exec "${cmd[@]}"
}

main "$@"
