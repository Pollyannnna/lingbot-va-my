#!/usr/bin/env bash
set -euo pipefail

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_NAME="${JOB_NAME:-extract_robocasa_latents_16x4}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/extract_robocasa_latents}"

CONDA_BASE="${CONDA_BASE:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lingbotva}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"

NODES="${NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"

DATASET_ROOT="${DATASET_ROOT:-/data/share/lijiang/data/robocasa365/target-human-process/target}"
MODEL_ROOT="${MODEL_ROOT:-/data/share/lijiang/ckpt/lingbot-va-base}"
TARGET_FPS="${TARGET_FPS:-10}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
VAE_DTYPE="${VAE_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-bfloat16}"

PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}"

setup_python_env() {
  if [[ -n "${CONDA_ENV_PREFIX}" || -n "${CONDA_ENV_NAME}" ]]; then
    if [[ -z "${CONDA_BASE}" ]]; then
      if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
      elif [[ -d "/root/miniconda3" ]]; then
        CONDA_BASE="/root/miniconda3"
      else
        echo "[ERROR] Failed to locate conda base. Set CONDA_BASE explicitly."
        exit 1
      fi
    fi

    if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
      echo "[ERROR] Missing conda init script: ${CONDA_BASE}/etc/profile.d/conda.sh"
      exit 1
    fi

    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
      conda activate "${CONDA_ENV_PREFIX}"
    else
      conda activate "${CONDA_ENV_NAME}"
    fi
  fi

  if [[ "${PYTHON_BIN}" == "python" ]]; then
    PYTHON_BIN="$(command -v python)"
  fi

  if [[ "${PYTHON_BIN}" == */* ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
      echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}"
      exit 1
    fi
  elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Failed to resolve PYTHON_BIN=${PYTHON_BIN}"
    exit 1
  else
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
  fi

  echo "[INFO] Using python: ${PYTHON_BIN}"
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "[INFO] Active conda env: ${CONDA_PREFIX}"
  fi
}

submit_job() {
  local -a sbatch_cmd=(
    sbatch
    "--job-name=${JOB_NAME}"
    "--nodes=${NODES}"
    "--ntasks-per-node=${GPUS_PER_NODE}"
    "--gres=gpu:${GPUS_PER_NODE}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--output=${LOG_DIR}/%x-%j.out"
    "--error=${LOG_DIR}/%x-%j.err"
  )

  if [[ -n "${PARTITION}" ]]; then
    sbatch_cmd+=("--partition=${PARTITION}")
  fi
  if [[ -n "${ACCOUNT}" ]]; then
    sbatch_cmd+=("--account=${ACCOUNT}")
  fi
  if [[ -n "${QOS}" ]]; then
    sbatch_cmd+=("--qos=${QOS}")
  fi
  if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
    read -r -a extra <<< "${SBATCH_EXTRA_ARGS}"
    sbatch_cmd+=("${extra[@]}")
  fi

  sbatch_cmd+=(
    "--export=ALL,PYTHON_BIN=${PYTHON_BIN},JOB_NAME=${JOB_NAME},LOG_DIR=${LOG_DIR},CONDA_BASE=${CONDA_BASE},CONDA_ENV_NAME=${CONDA_ENV_NAME},CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX},NODES=${NODES},GPUS_PER_NODE=${GPUS_PER_NODE},CPUS_PER_TASK=${CPUS_PER_TASK},TIME_LIMIT=${TIME_LIMIT},DATASET_ROOT=${DATASET_ROOT},MODEL_ROOT=${MODEL_ROOT},TARGET_FPS=${TARGET_FPS},HEIGHT=${HEIGHT},WIDTH=${WIDTH},VAE_DTYPE=${VAE_DTYPE},SAVE_DTYPE=${SAVE_DTYPE},PARTITION=${PARTITION},ACCOUNT=${ACCOUNT},QOS=${QOS},SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS},SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"
    "$0"
    "--inside-slurm"
    "$@"
  )

  printf 'Submitting job:'
  printf ' %q' "${sbatch_cmd[@]}"
  printf '\n'
  "${sbatch_cmd[@]}"
}

launch_job() {
  local total_tasks
  local -a srun_cmd
  total_tasks=$((SLURM_NNODES * GPUS_PER_NODE))
  setup_python_env

  srun_cmd=(
    srun
    "--nodes=${SLURM_NNODES}"
    "--ntasks=${total_tasks}"
    "--ntasks-per-node=${GPUS_PER_NODE}"
    "--gpus-per-task=1"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--kill-on-bad-exit=1"
    "--label"
  )

  if [[ -n "${SRUN_EXTRA_ARGS}" ]]; then
    read -r -a extra <<< "${SRUN_EXTRA_ARGS}"
    srun_cmd+=("${extra[@]}")
  fi

  srun_cmd+=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/script/extract_robocasa_latents.py"
    "--dataset-root" "${DATASET_ROOT}"
    "--model-root" "${MODEL_ROOT}"
    "--target-fps" "${TARGET_FPS}"
    "--height" "${HEIGHT}"
    "--width" "${WIDTH}"
    "--device" "auto"
    "--vae-dtype" "${VAE_DTYPE}"
    "--save-dtype" "${SAVE_DTYPE}"
    "$@"
  )

  printf 'Launching job in allocation:'
  printf ' %q' "${srun_cmd[@]}"
  printf '\n'

  cd "${REPO_ROOT}"
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  "${srun_cmd[@]}"
}

if [[ "${1:-}" == "--inside-slurm" ]]; then
  shift
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "[ERROR] Not inside a Slurm allocation and \`sbatch\` is unavailable."
    echo "        Either run this on the login node of a Slurm cluster,"
    echo "        or start an allocation manually and re-run the script."
    exit 1
  fi
  submit_job "$@"
  exit 0
fi

launch_job "$@"
