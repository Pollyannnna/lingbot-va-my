#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# SwanLab:
# 1) Recommended: run `swanlab login` once before training.
# 2) Optional: set SWANLAB_API_KEY for non-interactive login in train.py.
export ROBOTWIN_ENABLE_SWANLAB="${ROBOTWIN_ENABLE_SWANLAB:-1}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-Yeziyang}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-Lingbot-VA-Robotwin}"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
