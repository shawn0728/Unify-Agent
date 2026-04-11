#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Unified entry point for Unify-Agent inference.
#   --num_gpus 1  →  single-GPU   (unify-agent_infer.py)
#   --num_gpus N  →  multi-GPU    (unify-agent_infer_thread.py)
#
# Usage examples:
#
#   # Single-GPU, minimal run (no tool execution)
#   bash run_infer.sh \
#       --ip_data examples/ip_data.json \
#       --output_dir ./output/test \
#       --allow_batch_ip
#
#   # Single-GPU with search tools enabled
#   export OPENAI_API_KEY="sk-..."
#   export SERPER_API_KEY="..."
#   bash run_infer.sh \
#       --ip_data examples/ip_data.json \
#       --output_dir ./output/test \
#       --execute_tools \
#       --allow_batch_ip
#
#   # Multi-GPU (4 GPUs)
#   bash run_infer.sh \
#       --ip_data examples/ip_data.json \
#       --output_dir ./output/test \
#       --num_gpus 4 \
#       --execute_tools \
#       --allow_batch_ip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse --num_gpus from argv (default: 1) ────────────────────────────
NUM_GPUS=1
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_gpus=*)
            NUM_GPUS="${1#*=}"
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Pre-flight checks ──────────────────────────────────────────────────
HAS_EXECUTE_TOOLS=false
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [[ "$arg" == "--execute_tools" ]]; then
        HAS_EXECUTE_TOOLS=true
        break
    fi
done

if $HAS_EXECUTE_TOOLS; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "ERROR: --execute_tools requires OPENAI_API_KEY to be set."
        exit 1
    fi
    if [[ -z "${SERPER_API_KEY:-}" ]]; then
        echo "ERROR: --execute_tools requires SERPER_API_KEY to be set."
        exit 1
    fi
fi

# ── Dispatch ────────────────────────────────────────────────────────────
if [[ "$NUM_GPUS" -le 1 ]]; then
    echo "==> Running single-GPU inference (unify-agent_infer.py)"
    python "${SCRIPT_DIR}/unify-agent_infer.py" "${PASSTHROUGH_ARGS[@]}"
else
    echo "==> Running multi-GPU inference with ${NUM_GPUS} GPUs (unify-agent_infer_thread.py)"
    python "${SCRIPT_DIR}/unify-agent_infer_thread.py" \
        --num_gpus "${NUM_GPUS}" \
        "${PASSTHROUGH_ARGS[@]}"
fi
