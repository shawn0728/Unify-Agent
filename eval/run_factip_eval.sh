#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# FactIP Benchmark: end-to-end evaluation pipeline.
#
# This script runs three stages:
#   1. Generate  – produce images from FactIP prompts using Bagel/Unify-Agent
#   2. Score     – evaluate each image against ground-truth via an MLLM judge
#   3. Calculate – aggregate per-category and overall scores (0-100 scale)
#
# Prerequisites:
#   - Download the FactIP dataset:
#       git clone https://huggingface.co/datasets/csfufu/FactIP
#     or use the HuggingFace CLI:
#       huggingface-cli download csfufu/FactIP --repo-type dataset --local-dir ./FactIP
#
#   - Set OPENAI_API_KEY (or OPENAI_BASE_URL for compatible endpoints)
#
# Usage:
#   bash eval/run_factip_eval.sh \
#       --model_path /path/to/model \
#       --prompt_json /path/to/FactIP/test.json \
#       --gt_dir /path/to/FactIP/images \
#       --output_dir ./results/my_model
#
#   # Run only scoring + calculation (skip generation):
#   bash eval/run_factip_eval.sh \
#       --skip_generate \
#       --gt_dir /path/to/FactIP/images \
#       --output_dir ./results/my_model
#
#   # Run only calculation (scoring already done):
#   bash eval/run_factip_eval.sh \
#       --only_calculate \
#       --output_dir ./results/my_model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Defaults ───
MODEL_PATH=""
PROMPT_JSON=""
GT_DIR=""
OUTPUT_DIR="./results/factip_eval"
NUM_GPUS=8
SEED=42
THINK=""
SKIP_GENERATE=false
ONLY_CALCULATE=false
SCORE_WORKERS=4
EVAL_MODEL=""

# ─── Parse arguments ───
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)     MODEL_PATH="$2";     shift 2 ;;
        --prompt_json)    PROMPT_JSON="$2";    shift 2 ;;
        --gt_dir)         GT_DIR="$2";         shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";       shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --think)          THINK="--think";     shift   ;;
        --skip_generate)  SKIP_GENERATE=true;  shift   ;;
        --only_calculate) ONLY_CALCULATE=true; shift   ;;
        --score_workers)  SCORE_WORKERS="$2";  shift 2 ;;
        --eval_model)     EVAL_MODEL="$2";     shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

# ─── Stage 1: Generate images ───
if [[ "${ONLY_CALCULATE}" == false && "${SKIP_GENERATE}" == false ]]; then
    if [[ -z "${MODEL_PATH}" ]]; then
        echo "Error: --model_path is required for generation."
        exit 1
    fi
    if [[ -z "${PROMPT_JSON}" ]]; then
        echo "Error: --prompt_json is required for generation."
        echo "Download FactIP first: huggingface-cli download csfufu/FactIP --repo-type dataset --local-dir ./FactIP"
        exit 1
    fi

    echo "============================================"
    echo "  Stage 1/3: Generating images"
    echo "  Model:      ${MODEL_PATH}"
    echo "  Prompts:    ${PROMPT_JSON}"
    echo "  Output:     ${OUTPUT_DIR}"
    echo "  GPUs:       ${NUM_GPUS}"
    echo "============================================"

    python "${SCRIPT_DIR}/bagel_infer_batch.py" \
        --model_path "${MODEL_PATH}" \
        --prompt_json "${PROMPT_JSON}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_gpus "${NUM_GPUS}" \
        --seed "${SEED}" \
        ${THINK}

    echo ""
    echo "Generation complete."
    echo ""
fi

# ─── Stage 2: Score with MLLM judge ───
if [[ "${ONLY_CALCULATE}" == false ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "Error: OPENAI_API_KEY is required for scoring."
        echo "Export it before running: export OPENAI_API_KEY=your_key"
        exit 1
    fi

    SCORE_ARGS=(
        --base-dir "${OUTPUT_DIR}"
        --workers "${SCORE_WORKERS}"
    )
    if [[ -n "${EVAL_MODEL}" ]]; then
        SCORE_ARGS+=(--model "${EVAL_MODEL}")
    fi

    if [[ -n "${GT_DIR}" ]]; then
        export FACTIP_GT_DIR="${GT_DIR}"
    fi

    echo "============================================"
    echo "  Stage 2/3: Scoring with MLLM judge"
    echo "  Results:    ${OUTPUT_DIR}"
    echo "  Workers:    ${SCORE_WORKERS}"
    echo "  Model:      ${EVAL_MODEL:-gpt-4o (default)}"
    echo "============================================"

    python "${SCRIPT_DIR}/score_factip.py" "${SCORE_ARGS[@]}"

    echo ""
    echo "Scoring complete."
    echo ""
fi

# ─── Stage 3: Calculate aggregate scores ───
echo "============================================"
echo "  Stage 3/3: Calculating aggregate scores"
echo "  Results:    ${OUTPUT_DIR}"
echo "============================================"

python "${SCRIPT_DIR}/calculate.py" --base_dir "${OUTPUT_DIR}"

echo ""
echo "Evaluation pipeline complete!"
echo "Results saved to: ${OUTPUT_DIR}/overall_score.txt"
