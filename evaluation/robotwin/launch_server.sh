START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
LINGBOT_VA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
save_root="${SAVE_ROOT:-${LINGBOT_VA_ROOT}/visualization/}"
mkdir -p "$save_root"
cd "$LINGBOT_VA_ROOT"

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "$START_PORT" \
    --save_root "$save_root"
