#!/usr/bin/env bash
set -euo pipefail

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

NGPU="${NGPU:-8}"
NNODES="${NNODES:-8}"
NODE_RANK="${NODE_RANK:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-robotwin_train}"

DATASET_ROOT="${DATASET_ROOT:-/data/share/lijiang/data/robotwin-lingbotva/lerobot_robotwin_eef_clean_50}"
MODEL_ROOT="${MODEL_ROOT:-/data/share/lijiang/ckpt/lingbot-va-posttrain-robotwin}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-${DATASET_ROOT}/empty_emb.pt}"
SAVE_ROOT="${SAVE_ROOT:-/data/share/lijiang/yzy-exp/robotwin-clean50}"

ROBOTWIN_ENABLE_SWANLAB="${ROBOTWIN_ENABLE_SWANLAB:-1}"
ROBOTWIN_NUM_STEPS="${ROBOTWIN_NUM_STEPS:-10000}"
ROBOTWIN_BATCH_SIZE="${ROBOTWIN_BATCH_SIZE:-1}"
ROBOTWIN_GRAD_ACC="${ROBOTWIN_GRAD_ACC:-1}"
ROBOTWIN_LR="${ROBOTWIN_LR:-1e-5}"
ROBOTWIN_WARMUP_STEPS="${ROBOTWIN_WARMUP_STEPS:-10}"
ROBOTWIN_BETA1="${ROBOTWIN_BETA1:-0.9}"
ROBOTWIN_BETA2="${ROBOTWIN_BETA2:-0.95}"
ROBOTWIN_WEIGHT_DECAY="${ROBOTWIN_WEIGHT_DECAY:-0.1}"
ROBOTWIN_SAVE_INTERVAL="${ROBOTWIN_SAVE_INTERVAL:-1000}"
ROBOTWIN_GC_INTERVAL="${ROBOTWIN_GC_INTERVAL:-50}"
ROBOTWIN_CFG_PROB="${ROBOTWIN_CFG_PROB:-0.1}"
ROBOTWIN_LOAD_WORKER="${ROBOTWIN_LOAD_WORKER:-16}"
ROBOTWIN_DATASET_INIT_WORKER="${ROBOTWIN_DATASET_INIT_WORKER:-8}"
ROBOTWIN_DATALOADER_PREFETCH="${ROBOTWIN_DATALOADER_PREFETCH:-2}"
ROBOTWIN_PIN_MEMORY="${ROBOTWIN_PIN_MEMORY:-1}"
ROBOTWIN_PERSISTENT_WORKERS="${ROBOTWIN_PERSISTENT_WORKERS:-1}"
ROBOTWIN_RESUME_FROM="${ROBOTWIN_RESUME_FROM:-}"

SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-Yeziyang}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-Lingbot-VA-Robotwin-clean50}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-fViowdx9CQvjV7ofiP6ET}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
USE_RDMA="${USE_RDMA:-1}"
REQUIRE_RDMA="${REQUIRE_RDMA:-0}"
AUTO_INSTALL_RDMA_DEPS="${AUTO_INSTALL_RDMA_DEPS:-1}"
RDMA_CREATE_DEVNODES="${RDMA_CREATE_DEVNODES:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<EOF
Usage:
  bash script/submit_train_robotwin_8x8_rdma.sh

This is a pure torchrun wrapper with 8x8 robotwin defaults.

Common overrides:
  SAVE_ROOT=/path/to/output
  ROBOTWIN_RESUME_FROM=/path/to/checkpoint_step_xxx
  MASTER_ADDR=rank0-hostname
  MASTER_PORT=29501
  NNODES=8
  NGPU=8

Validation:
  bash script/submit_train_robotwin_8x8_rdma.sh --dry-run
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

normalize_resume_from() {
  local resume_path=""

  resume_path="${ROBOTWIN_RESUME_FROM}"
  if [[ -z "${resume_path}" ]]; then
    return 0
  fi

  if [[ -f "${resume_path}" ]]; then
    if [[ "${resume_path}" == */transformer/*.safetensors ]]; then
      ROBOTWIN_RESUME_FROM="$(dirname "$(dirname "${resume_path}")")"
    elif [[ "${resume_path}" == */*.safetensors ]]; then
      ROBOTWIN_RESUME_FROM="$(dirname "${resume_path}")"
    fi
  elif [[ -d "${resume_path}" && "${resume_path}" == */transformer ]]; then
    ROBOTWIN_RESUME_FROM="$(dirname "${resume_path}")"
  fi
}

auto_resume_from_latest_checkpoint() {
  local latest_checkpoint=""

  if [[ -n "${ROBOTWIN_RESUME_FROM}" ]]; then
    return
  fi

  if [[ ! -d "${SAVE_ROOT}/checkpoints" ]]; then
    return
  fi

  latest_checkpoint="$(
    find "${SAVE_ROOT}/checkpoints" -maxdepth 1 -type d -name 'checkpoint_step_*' | sort -V | tail -n 1
  )"

  if [[ -n "${latest_checkpoint}" ]]; then
    ROBOTWIN_RESUME_FROM="${latest_checkpoint}"
  fi
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

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run: would execute apt-get install -y librdmacm-dev libibverbs-dev rdma-core infiniband-diags ibverbs-utils"
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
    rdma-core \
    infiniband-diags \
    ibverbs-utils
  ldconfig || true
}

detect_active_ib_hcas() {
  local dev=""
  local state_file=""
  local layer_file=""
  local state=""
  local layer=""
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
  local major_num=""
  local minor_num=""

  [[ "${RDMA_CREATE_DEVNODES}" == "1" ]] || return 0
  [[ -n "${dev_num}" ]] || return 0

  major_num="${dev_num%%:*}"
  minor_num="${dev_num##*:}"

  mkdir -p /dev/infiniband
  [[ -e "${dev_path}" ]] || mknod "${dev_path}" "${node_type}" "${major_num}" "${minor_num}" || true
  chmod 666 "${dev_path}" 2>/dev/null || true
}

populate_infiniband_devfs() {
  local sys_dev=""
  local dev_num=""
  local name=""

  [[ "${USE_RDMA}" == "1" ]] || return
  [[ "${DRY_RUN}" == "1" ]] && return

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
    nvcc -V
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

  export NCCL_DEBUG

  if [[ "${USE_RDMA}" == "0" ]]; then
    echo "[INFO] USE_RDMA=0, leaving NCCL transport settings unchanged."
    echo "[INFO] NCCL_DEBUG=${NCCL_DEBUG}"
    return
  fi

  install_rdma_userspace
  populate_infiniband_devfs

  if ! check_rdma_runtime; then
    echo "[INFO] NCCL_DEBUG=${NCCL_DEBUG}"
    return
  fi

  detected_hcas="$(detect_active_ib_hcas)"
  if [[ -z "${detected_hcas}" ]]; then
    echo "[WARN] No active InfiniBand HCAs detected under /sys/class/infiniband."
    echo "[WARN] Continuing without forcing NCCL RDMA."
    echo "[INFO] NCCL_DEBUG=${NCCL_DEBUG}"
    return
  fi

  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-${detected_hcas}}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
  export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

  echo "[INFO] RDMA enabled for NCCL"
  echo "       NCCL_DEBUG=${NCCL_DEBUG}"
  echo "       NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  echo "       NCCL_IB_HCA=${NCCL_IB_HCA}"
  echo "       NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  echo "       NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
}

ensure_empty_emb() {
  if [[ -f "${EMPTY_EMB_PATH}" ]]; then
    return
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run: would generate empty_emb.pt at ${EMPTY_EMB_PATH}"
    return
  fi

  echo "[INFO] empty_emb.pt not found, generating it at ${EMPTY_EMB_PATH}"
  "${PYTHON_BIN}" "${REPO_ROOT}/script/make_empty_emb.py" \
    --model-root "${MODEL_ROOT}" \
    --output "${EMPTY_EMB_PATH}" \
    --overwrite
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
  if ! find "${DATASET_ROOT}" -path '*/meta/info.json' -print -quit | grep -q .; then
    echo "[ERROR] DATASET_ROOT does not look like a lerobot latent dataset: ${DATASET_ROOT}"
    exit 1
  fi
  if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "[ERROR] Failed to infer a usable MASTER_ADDR for multi-node torchrun."
    echo "        Current host: ${HOSTNAME:-$(hostname 2>/dev/null || echo unknown)}"
    echo "        Please set MASTER_ADDR explicitly to the rank0 host."
    exit 1
  fi
  mkdir -p "${SAVE_ROOT}"
}

print_summary() {
  local global_batch=0
  global_batch=$((NNODES * NGPU * ROBOTWIN_BATCH_SIZE * ROBOTWIN_GRAD_ACC))

  echo "[INFO] Torchrun config:"
  echo "       cluster=${NNODES}x${NGPU}"
  echo "       node_rank=${NODE_RANK}"
  echo "       master_addr=${MASTER_ADDR}"
  echo "       master_port=${MASTER_PORT}"
  echo "       dataset=${DATASET_ROOT}"
  echo "       model=${MODEL_ROOT}"
  echo "       empty_emb=${EMPTY_EMB_PATH}"
  echo "       save_root=${SAVE_ROOT}"
  echo "       steps=${ROBOTWIN_NUM_STEPS}"
  echo "       save_interval=${ROBOTWIN_SAVE_INTERVAL}"
  echo "       lr=${ROBOTWIN_LR}"
  echo "       per_gpu_batch=${ROBOTWIN_BATCH_SIZE}"
  echo "       grad_acc=${ROBOTWIN_GRAD_ACC}"
  echo "       global_batch=${global_batch}"
  echo "       load_worker=${ROBOTWIN_LOAD_WORKER}"
  if [[ -n "${ROBOTWIN_RESUME_FROM}" ]]; then
    echo "       resume_from=${ROBOTWIN_RESUME_FROM}"
  else
    echo "       resume_from=<none>"
  fi
}

main() {
  local -a overrides=()
  local -a cmd=()
  local has_save_root=0
  local dry_run=0

  while (( $# > 0 )); do
    case "${1}" in
      -h|--help)
        usage
        exit 0
        ;;
      --dry-run)
        DRY_RUN=1
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
  normalize_resume_from
  auto_resume_from_latest_checkpoint
  log_cuda_env
  validate_config
  ensure_empty_emb
  setup_rdma_env
  print_summary

  export PYTHON_BIN NGPU NNODES NODE_RANK MASTER_ADDR MASTER_PORT LOG_RANK
  export TORCHFT_LIGHTHOUSE CONFIG_NAME
  export OMP_NUM_THREADS MKL_NUM_THREADS
  export PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF

  export ROBOTWIN_DATASET_PATH="${DATASET_ROOT}"
  export ROBOTWIN_EMPTY_EMB_PATH="${EMPTY_EMB_PATH}"
  export ROBOTWIN_PRETRAINED_MODEL="${MODEL_ROOT}"

  export ROBOTWIN_ENABLE_SWANLAB ROBOTWIN_NUM_STEPS ROBOTWIN_BATCH_SIZE
  export ROBOTWIN_GRAD_ACC ROBOTWIN_LR ROBOTWIN_WARMUP_STEPS
  export ROBOTWIN_BETA1 ROBOTWIN_BETA2 ROBOTWIN_WEIGHT_DECAY
  export ROBOTWIN_SAVE_INTERVAL ROBOTWIN_GC_INTERVAL ROBOTWIN_CFG_PROB
  export ROBOTWIN_LOAD_WORKER ROBOTWIN_DATASET_INIT_WORKER ROBOTWIN_DATALOADER_PREFETCH
  export ROBOTWIN_PIN_MEMORY ROBOTWIN_PERSISTENT_WORKERS ROBOTWIN_RESUME_FROM

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
    "${REPO_ROOT}/script/run_va_posttrain.sh"
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
