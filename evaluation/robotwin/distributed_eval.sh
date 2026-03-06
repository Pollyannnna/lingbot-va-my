#!/bin/bash
###############################################################################
# distributed_eval.sh
#
# All-in-one distributed evaluation for ALL RoboTwin tasks.
# A single script runs on EVERY node — it auto-detects its own NODE_RANK
# from cluster environment variables and assigns tasks accordingly.
#
# ACP Usage (single launch command for 8-node cluster):
#   bash /data/250010187/yeziyang1/lingbot-va/evaluation/robotwin/distributed_eval.sh \
#       results/full_eval 100 0
#
# The script auto-detects NODE_RANK from these env vars (in order):
#   $NODE_RANK → $RANK → $SLURM_NODEID → $ACP_NODE_RANK → $OMPI_COMM_WORLD_RANK
# If none found, falls back to argument $1 if it looks like a number 0-7.
#
# What it does on each node:
#   1. Figures out which tasks belong to this node
#   2. Starts a LingBot-VA server per task (lingbotva env, 1 GPU each)
#   3. Waits for servers to be healthy (WebSocket probe)
#   4. Starts a RoboTwin client with logging per task (robotwin env, same GPU)
#   5. Waits for all clients, cleans up servers
###############################################################################
# Note: do NOT use set -e here — health check failures should be handled explicitly
set -uo pipefail

# ═══════════════════════════════════════════════════════════════════════
# 1. Auto-detect NODE_RANK from cluster environment
# ═══════════════════════════════════════════════════════════════════════
detect_node_rank() {
    # Try common cluster scheduler env vars
    for var in NODE_RANK RANK SLURM_NODEID ACP_NODE_RANK OMPI_COMM_WORLD_RANK INDEX; do
        val="${!var:-}"
        if [[ -n "$val" && "$val" =~ ^[0-9]+$ ]]; then
            echo "$val"
            return
        fi
    done
    # Fallback: first argument if it looks like a number
    if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ ]]; then
        echo "$1"
        return
    fi
    echo "ERROR: Cannot detect NODE_RANK. Set NODE_RANK env var or pass as first argument." >&2
    exit 1
}

MY_NODE_RANK=$(detect_node_rank "${1:-}")

# ═══════════════════════════════════════════════════════════════════════
# 2. Arguments (shift if first arg was used as NODE_RANK)
# ═══════════════════════════════════════════════════════════════════════
# If first arg was consumed as NODE_RANK, shift remaining args
if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ && "$1" == "$MY_NODE_RANK" ]]; then
    shift
fi

SAVE_ROOT=${1:-"results/chunk-aware-full_eval_$(date +%m%d)"}
TEST_NUM=${2:-100}
SEED=${3:-0}

# ═══════════════════════════════════════════════════════════════════════
# 3. Cluster Configuration (ADJUST FOR YOUR SETUP)
# ═══════════════════════════════════════════════════════════════════════
NUM_NODES=8
GPUS_PER_NODE=8

CONDA_BASE="/data/250010187/yeziyang1/miniconda3"
SERVER_PYTHON="${CONDA_BASE}/envs/lingbotva/bin/python"
CLIENT_PYTHON="${CONDA_BASE}/envs/robotwin/bin/python"

LINGBOT_VA_ROOT="/data/250010187/yeziyang1/lingbot-va"

BASE_WS_PORT=29556
BASE_MASTER_PORT=29700
MAX_WAIT_SECS=900       # 15 min: large models (20GB+) take 5-10 min to load
HEALTH_INTERVAL=30     # Print a dot every 30s to show progress

# ═══════════════════════════════════════════════════════════════════════
# 4. All 50 RoboTwin Tasks
# ═══════════════════════════════════════════════════════════════════════
ALL_TASKS=(
  stack_bowls_three handover_block hanging_mug scan_object
  lift_pot put_object_cabinet stack_blocks_three place_shoe
  adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad
  pick_dual_bottles shake_bottle place_fan turn_switch
  shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand
  put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket
  pick_diverse_bottles open_microwave beat_block_hammer press_stapler
  click_bell move_playingcard_away open_laptop move_can_pot
  stack_bowls_two place_a2b_right stamp_seal place_object_basket
  handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox
  click_alarmclock blocks_ranking_size place_phone_stand place_can_basket
  place_object_scale place_a2b_left grab_roller place_dual_shoes
  place_empty_cup blocks_ranking_rgb
)
TOTAL_TASKS=${#ALL_TASKS[@]}

# ═══════════════════════════════════════════════════════════════════════
# 5. Environment Setup (from test.sh)
# ═══════════════════════════════════════════════════════════════════════
export NVIDIA_DRIVER_CAPABILITIES=all
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so.1"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export DISPLAY=""
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}

# Install system deps quietly
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 librdmacm-dev libibverbs-dev \
    rdma-core infiniband-diags libvulkan1 mesa-vulkan-drivers vulkan-utils ffmpeg 2>/dev/null || true
mkdir -p /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d 2>/dev/null || true
${CLIENT_PYTHON} -m pip install -q "setuptools<81.0.0" "pillow<12.0.0" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════
# 6. Calculate Task Assignment
# ═══════════════════════════════════════════════════════════════════════
TASKS_PER_NODE=$(( (TOTAL_TASKS + NUM_NODES - 1) / NUM_NODES ))
START_IDX=$(( MY_NODE_RANK * TASKS_PER_NODE ))
END_IDX=$(( START_IDX + TASKS_PER_NODE ))
[ $END_IDX -gt $TOTAL_TASKS ] && END_IDX=$TOTAL_TASKS
if [ $START_IDX -ge $TOTAL_TASKS ]; then
    echo "[Node ${MY_NODE_RANK}] No tasks to run. Exiting."
    exit 0
fi
NUM_MY_TASKS=$(( END_IDX - START_IDX ))

echo "═══════════════════════════════════════════════════════════════"
echo "  Node ${MY_NODE_RANK}/${NUM_NODES} | Tasks ${START_IDX}..$(( END_IDX - 1 )) (${NUM_MY_TASKS} tasks)"
echo "  Save: ${SAVE_ROOT} | Episodes: ${TEST_NUM} | Seed: ${SEED}"
echo "  My tasks: ${ALL_TASKS[@]:$START_IDX:$NUM_MY_TASKS}"
echo "═══════════════════════════════════════════════════════════════"

LOG_DIR="${LINGBOT_VA_ROOT}/logs/node_${MY_NODE_RANK}"
mkdir -p "${LOG_DIR}"
BATCH_TIME=$(date +%Y%m%d_%H%M%S)

# ═══════════════════════════════════════════════════════════════════════
# 7. Helper: Wait for Server Health
# ═══════════════════════════════════════════════════════════════════════
wait_for_server() {
    local port=$1 task=$2 log_file=$3 elapsed=0
    echo "  [${task}] Waiting for server :${port} (max ${MAX_WAIT_SECS}s)..."
    while [ $elapsed -lt $MAX_WAIT_SECS ]; do
        if ${CLIENT_PYTHON} -c "
import websockets.sync.client, sys
ok = False
try:
    c=websockets.sync.client.connect('ws://127.0.0.1:${port}',compression=None,max_size=None,ping_interval=None,close_timeout=3)
    c.recv(timeout=5)
    c.close()
    ok = True
except Exception:
    pass
sys.exit(0 if ok else 1)
" 2>/dev/null; then
            echo "  [${task}] ✓ READY after ${elapsed}s"
            return 0
        fi
        sleep $HEALTH_INTERVAL
        elapsed=$(( elapsed + HEALTH_INTERVAL ))
        # Print recent server log to show it is still loading
        echo "  [${task}] still loading... (${elapsed}s) | last log: $(tail -1 ${log_file} 2>/dev/null | cut -c1-80)"
    done
    echo "  [${task}] ✗ TIMEOUT after ${MAX_WAIT_SECS}s. Last 5 lines of server log:"
    tail -5 "${log_file}" 2>/dev/null | sed "s/^/    /"
    return 1
}

# ═══════════════════════════════════════════════════════════════════════
# 8. Phase 1: Start Servers
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 1: Starting ${NUM_MY_TASKS} servers ─────────────────────"

cd "${LINGBOT_VA_ROOT}"
mkdir -p visualization

SERVER_PIDS=()
declare -A TASK_WS_PORT TASK_GPU

for local_i in $(seq 0 $(( NUM_MY_TASKS - 1 ))); do
    global_i=$(( START_IDX + local_i ))
    task="${ALL_TASKS[$global_i]}"
    gpu=$local_i
    ws_port=$(( BASE_WS_PORT + local_i ))
    master_port=$(( BASE_MASTER_PORT + local_i ))

    TASK_WS_PORT[$task]=$ws_port
    TASK_GPU[$task]=$gpu

    log="${LOG_DIR}/server_${task}_${BATCH_TIME}.log"
    echo "  [${task}] GPU=${gpu} PORT=${ws_port}"

    CUDA_VISIBLE_DEVICES=$gpu \
    nohup ${SERVER_PYTHON} -m torch.distributed.run \
        --nproc_per_node 1 --master_port $master_port \
        wan_va/wan_va_server.py --config-name robotwin \
        --port $ws_port --save_root visualization/ \
        > "$log" 2>&1 &
    SERVER_PIDS+=($!)
    sleep 2
done

echo ""
echo "── Phase 2: Health checks (parallel, max ${MAX_WAIT_SECS}s) ───────────────"

# Run all health checks in PARALLEL so we don't wait sequentially
HEALTH_PIDS=()
HEALTH_RESULTS_DIR="${LOG_DIR}/health_${BATCH_TIME}"
mkdir -p "${HEALTH_RESULTS_DIR}"

for local_i in $(seq 0 $(( NUM_MY_TASKS - 1 ))); do
    task="${ALL_TASKS[$(( START_IDX + local_i ))]}"
    log="${LOG_DIR}/server_${task}_${BATCH_TIME}.log"
    result_file="${HEALTH_RESULTS_DIR}/${task}.result"
    (
        if wait_for_server ${TASK_WS_PORT[$task]} $task $log; then
            echo "ok" > "$result_file"
        else
            echo "fail" > "$result_file"
        fi
    ) &
    HEALTH_PIDS+=($!)
done

# Wait for all parallel health checks to complete
for pid in "${HEALTH_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

# Count results
READY=0
for local_i in $(seq 0 $(( NUM_MY_TASKS - 1 ))); do
    task="${ALL_TASKS[$(( START_IDX + local_i ))]}"
    result_file="${HEALTH_RESULTS_DIR}/${task}.result"
    [ -f "$result_file" ] && [ "$(cat $result_file)" = "ok" ] && READY=$(( READY + 1 ))
done

echo "  ${READY}/${NUM_MY_TASKS} servers healthy."
if [ $READY -eq 0 ]; then
    echo "FATAL: No servers started! Check logs in ${LOG_DIR}/"
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════
# 10. Phase 3: Start Clients with Logging
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 3: Starting ${NUM_MY_TASKS} clients ─────────────────────"

CLIENT_PIDS=()
for local_i in $(seq 0 $(( NUM_MY_TASKS - 1 ))); do
    task="${ALL_TASKS[$(( START_IDX + local_i ))]}"
    gpu=${TASK_GPU[$task]}
    ws_port=${TASK_WS_PORT[$task]}
    log="${LOG_DIR}/client_${task}_${BATCH_TIME}.log"

    echo "  [${task}] GPU=${gpu} PORT=${ws_port}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    nohup ${CLIENT_PYTHON} -m evaluation.robotwin.eval_with_logging \
        --config policy/ACT/deploy_policy.yml --overrides \
        --task_name ${task} --task_config demo_clean \
        --train_config_name 0 --model_name 0 --ckpt_setting 0 \
        --seed ${SEED} --policy_name ACT \
        --save_root ${SAVE_ROOT} \
        --video_guidance_scale 5 --action_guidance_scale 1 \
        --test_num ${TEST_NUM} --port ${ws_port} \
        > "$log" 2>&1 &
    CLIENT_PIDS+=($!)
done

echo "${SERVER_PIDS[@]}" > "${LOG_DIR}/server_pids.txt"
echo "${CLIENT_PIDS[@]}" > "${LOG_DIR}/client_pids.txt"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Node ${MY_NODE_RANK}: ${NUM_MY_TASKS} tasks launched!"
echo "  Monitor: tail -f ${LOG_DIR}/client_*_${BATCH_TIME}.log"
echo "  Kill:    kill \$(cat ${LOG_DIR}/*_pids.txt)"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# 11. Phase 4: Wait for Completion + Cleanup
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 4: Waiting for clients to finish ────────────────────"

FAILS=0
for i in "${!CLIENT_PIDS[@]}"; do
    task="${ALL_TASKS[$(( START_IDX + i ))]}"
    if wait ${CLIENT_PIDS[$i]} 2>/dev/null; then
        echo "  ✓ ${task}"
    else
        echo "  ✗ ${task} (exit=$?)"
        FAILS=$(( FAILS + 1 ))
    fi
done

echo ""
echo "── Phase 5: Cleaning up servers ──────────────────────────────"
for pid in "${SERVER_PIDS[@]}"; do kill $pid 2>/dev/null || true; done
sleep 2
for pid in "${SERVER_PIDS[@]}"; do kill -9 $pid 2>/dev/null || true; done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Node ${MY_NODE_RANK} DONE! Tasks: ${NUM_MY_TASKS}, Failures: ${FAILS}"
echo "  Results: ${SAVE_ROOT}/"
echo "  Logs:    ${LOG_DIR}/"
echo "═══════════════════════════════════════════════════════════════"
