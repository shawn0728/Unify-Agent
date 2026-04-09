set -euo pipefail

export WANDB_API_KEY=${WANDB_API_KEY:?"Please set WANDB_API_KEY env var"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
SYNC_IFNAME=${SYNC_IFNAME:-bond1}
COMM_PROFILE=${COMM_PROFILE:-socket_safe}

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${SYNC_IFNAME}}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${SYNC_IFNAME}}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export GLOO_USE_IPV6=${GLOO_USE_IPV6:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}

case "${COMM_PROFILE}" in
    socket_safe)
        # Most robust profile: force pure TCP sockets.
        export NCCL_IB_DISABLE=1
        export NCCL_NET=Socket
        ;;
    ib_min)
        # Minimal IB profile: keep only essentials to reduce mismatch risk.
        export NCCL_IB_DISABLE=0
        export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
        export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
        ;;
    ib_perf)
        # Aggressive IB profile for peak throughput (requires homogeneous cluster).
        export NCCL_IB_DISABLE=0
        export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
        export NCCL_IB_SL=${NCCL_IB_SL:-3}
        export NCCL_CHECK_DISABLE=${NCCL_CHECK_DISABLE:-1}
        export NCCL_LL_THRESHOLD=${NCCL_LL_THRESHOLD:-16384}
        export NCCL_IB_CUDA_SUPPORT=${NCCL_IB_CUDA_SUPPORT:-1}
        export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond1}
        export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}
        export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
        export SHARP_COLL_ENABLE_SAT=${SHARP_COLL_ENABLE_SAT:-0}
        export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
        export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
        export NCCL_IB_TC=${NCCL_IB_TC:-160}
        export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-0}
        export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
        ;;
    *)
        echo "[launcher] invalid COMM_PROFILE=${COMM_PROFILE}, expected socket_safe|ib_min|ib_perf"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# W&B input logging controls
WANDB_LOG_INPUT_DATA=${WANDB_LOG_INPUT_DATA:-False}
WANDB_LOG_INPUT_EVERY=${WANDB_LOG_INPUT_EVERY:-100}
WANDB_INPUT_PREVIEW_TOKENS=${WANDB_INPUT_PREVIEW_TOKENS:-128}
WANDB_LOG_INPUT_IMAGES=${WANDB_LOG_INPUT_IMAGES:-False}
# -1 means log all reference/gen images in the batch
WANDB_MAX_LOGGED_IMAGES=${WANDB_MAX_LOGGED_IMAGES:--1}
WANDB_PROJECT=${WANDB_PROJECT:-unify-agent-sft}
WANDB_NAME=${WANDB_NAME:-sft_run}
WANDB_RUNID=${WANDB_RUNID:-1}
SPECIAL_TOKEN_CE_REWEIGHT=${SPECIAL_TOKEN_CE_REWEIGHT:-True}
SPECIAL_TOKEN_CE_WEIGHT=${SPECIAL_TOKEN_CE_WEIGHT:-3.0}
# DataLoader stability knobs (default: debug-stable mode)
NUM_WORKERS=${NUM_WORKERS:-1}
PIN_MEMORY=${PIN_MEMORY:-false}
KEEP_LAST_N=${KEEP_LAST_N:-8}
RESULTS_DIR=${RESULTS_DIR:-./outputs/sft}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./outputs/sft}
RESUME_FROM=${RESUME_FROM:?"Please set RESUME_FROM to your base model checkpoint path"}
AUTO_RESUME=${AUTO_RESUME:-True}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-True}
FINETUNE_FROM_EMA=${FINETUNE_FROM_EMA:-True}

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/train_sft_v1.sh <nnodes> <node_rank> <master_addr> <master_port>"
    exit 1
fi

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export no_proxy=localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}
LAUNCH_MODE=${LAUNCH_MODE:-static}
DIST_INIT_TIMEOUT_SEC=${DIST_INIT_TIMEOUT_SEC:-180}
TORCHELASTIC_MAX_RESTARTS=${TORCHELASTIC_MAX_RESTARTS:-3}
TORCHELASTIC_MONITOR_INTERVAL=${TORCHELASTIC_MONITOR_INTERVAL:-5}
TORCHELASTIC_RDZV_BACKEND=${TORCHELASTIC_RDZV_BACKEND:-c10d}
TORCHELASTIC_RDZV_ID=${TORCHELASTIC_RDZV_ID:-bagel_${MASTER_PORT}}
TORCHELASTIC_MIN_NODES=${TORCHELASTIC_MIN_NODES:-${NNODES}}
TORCHELASTIC_MAX_NODES=${TORCHELASTIC_MAX_NODES:-${NNODES}}

export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export DIST_INIT_TIMEOUT_SEC

if ! [[ "${NNODES}" =~ ^[0-9]+$ && "${NODE_RANK}" =~ ^[0-9]+$ && "${MASTER_PORT}" =~ ^[0-9]+$ ]]; then
    echo "[launcher] invalid numeric args: nnodes=${NNODES} node_rank=${NODE_RANK} master_port=${MASTER_PORT}"
    exit 1
fi

if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "[launcher] node_rank out of range: node_rank=${NODE_RANK}, nnodes=${NNODES}"
    exit 1
fi

if [[ "${LAUNCH_MODE}" != "elastic" && "${LAUNCH_MODE}" != "static" ]]; then
    echo "[launcher] invalid LAUNCH_MODE=${LAUNCH_MODE}, expected elastic|static"
    exit 1
fi

# Accept both checkpoint directory and direct safetensors file path.
if [[ -f "${RESUME_FROM}" ]]; then
    case "${RESUME_FROM}" in
        *.safetensors) RESUME_FROM="$(dirname "${RESUME_FROM}")" ;;
        *)
            echo "[launcher] RESUME_FROM file must be a .safetensors file: ${RESUME_FROM}"
            exit 1
            ;;
    esac
fi

if [[ ! -d "${RESUME_FROM}" ]]; then
    echo "[launcher] RESUME_FROM path not found or not a directory: ${RESUME_FROM}"
    exit 1
fi

CHECK_PORT=$((MASTER_PORT + 17))
if (( CHECK_PORT > 65535 )); then
    echo "[launcher] preflight check port overflow: master_port=${MASTER_PORT}, check_port=${CHECK_PORT}"
    exit 1
fi

echo "[launcher] $(date '+%F %T') host=$(hostname) nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[launcher] nproc_per_node=${NPROC_PER_NODE}"
echo "[launcher] launch_mode=${LAUNCH_MODE} dist_init_timeout_sec=${DIST_INIT_TIMEOUT_SEC}"
echo "[launcher] resume_from=${RESUME_FROM}"
echo "[launcher] auto_resume=${AUTO_RESUME} resume_model_only=${RESUME_MODEL_ONLY} finetune_from_ema=${FINETUNE_FROM_EMA}"
echo "[launcher] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[launcher] dataloader num_workers=${NUM_WORKERS} pin_memory=${PIN_MEMORY}"
echo "[launcher] comm_profile=${COMM_PROFILE}"
echo "[launcher] net ifname(sync)=${SYNC_IFNAME} nccl_ifname=${NCCL_SOCKET_IFNAME} gloo_ifname=${GLOO_SOCKET_IFNAME}"
if [[ "${LAUNCH_MODE}" == "elastic" ]]; then
    echo "[launcher] elastic rdzv_backend=${TORCHELASTIC_RDZV_BACKEND} rdzv_id=${TORCHELASTIC_RDZV_ID} nnodes=${TORCHELASTIC_MIN_NODES}:${TORCHELASTIC_MAX_NODES} max_restarts=${TORCHELASTIC_MAX_RESTARTS}"
fi

NNODES="${NNODES}" NODE_RANK="${NODE_RANK}" MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" python - <<'PY'
import os
import socket
import time

nnodes = int(os.environ["NNODES"])
node_rank = int(os.environ["NODE_RANK"])
master_addr = os.environ["MASTER_ADDR"]
master_port = int(os.environ["MASTER_PORT"])

print(
    f"[netcheck] host={socket.gethostname()} rank={node_rank}/{nnodes} master={master_addr}:{master_port}",
    flush=True,
)

try:
    resolved = socket.gethostbyname(master_addr)
    print(f"[netcheck] resolve {master_addr} -> {resolved}", flush=True)
except Exception as e:
    print(f"[netcheck] FAIL resolve {master_addr}: {e}", flush=True)
    raise SystemExit(2)

if node_rank != 0:
    ok = False
    last_err = ""
    for _ in range(20):
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect((master_addr, master_port))
            ok = True
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5)
        finally:
            try:
                s.close()
            except Exception:
                pass
    if ok:
        print(f"[netcheck] OK connect {master_addr}:{master_port}", flush=True)
    else:
        print(f"[netcheck] WARN cannot connect {master_addr}:{master_port}: {last_err}", flush=True)
else:
    print("[netcheck] rank0 skip connect probe.", flush=True)
PY

python scripts/dist_preflight_check.py \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --timeout-sec 60

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_BIN="$(command -v torchrun)"
else
    echo "[launcher] torchrun not found in PATH."
    echo "[launcher] Please activate your training env first or install PyTorch."
    exit 127
fi
echo "[launcher] using torchrun=${TORCHRUN_BIN}"

TRAIN_CMD=(
    train/pretrain_unified_navit.py
    --model_path "${RESUME_FROM}"
    --vit_path "${VIT_PATH:?"Please set VIT_PATH to your SigLIP checkpoint path"}"
    --layer_module Qwen2MoTDecoderLayer
    --max_latent_size 64
    --dataset_config_file ./data/configs/agent_data.yaml
    --auto_resume "${AUTO_RESUME}"
    --resume_model_only "${RESUME_MODEL_ONLY}"
    --finetune_from_hf True
    --finetune_from_ema "${FINETUNE_FROM_EMA}"
    --resume_from "${RESUME_FROM}"
    --results_dir "${RESULTS_DIR}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --num_workers "${NUM_WORKERS}"
    --pin_memory "${PIN_MEMORY}"
    --log_every 20
    --save_every 500
    --keep_last_n "${KEEP_LAST_N}"
    --num_shard $NPROC_PER_NODE
    --num_replicate $NNODES
    --lr 5e-5
    --warmup_steps 500
    --expected_num_tokens 40240
    --max_num_tokens 41520
    --max_num_tokens_per_sample 40240
    --wandb_project "${WANDB_PROJECT}"
    --wandb_name "${WANDB_NAME}"
    --wandb_runid "${WANDB_RUNID}"
    --wandb_offline False
    --wandb_log_input_data ${WANDB_LOG_INPUT_DATA}
    --wandb_log_input_every ${WANDB_LOG_INPUT_EVERY}
    --wandb_input_preview_tokens ${WANDB_INPUT_PREVIEW_TOKENS}
    --wandb_log_input_images ${WANDB_LOG_INPUT_IMAGES}
    --wandb_max_logged_images ${WANDB_MAX_LOGGED_IMAGES}
    --ce_loss_reweighting True
    --special_token_ce_reweight ${SPECIAL_TOKEN_CE_REWEIGHT}
    --special_token_ce_weight ${SPECIAL_TOKEN_CE_WEIGHT}
    --max_grad_norm 5.0
    --ce_weight 1.0
    --mse_weight 1.0
)

if [[ "${LAUNCH_MODE}" == "elastic" ]]; then
    "${TORCHRUN_BIN}" \
        --nnodes "${TORCHELASTIC_MIN_NODES}:${TORCHELASTIC_MAX_NODES}" \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --rdzv_backend "${TORCHELASTIC_RDZV_BACKEND}" \
        --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv_id "${TORCHELASTIC_RDZV_ID}" \
        --max_restarts "${TORCHELASTIC_MAX_RESTARTS}" \
        --monitor_interval "${TORCHELASTIC_MONITOR_INTERVAL}" \
        "${TRAIN_CMD[@]}"
else
    "${TORCHRUN_BIN}" \
        --nnodes "${NNODES}" \
        --node_rank "${NODE_RANK}" \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        "${TRAIN_CMD[@]}"
fi
