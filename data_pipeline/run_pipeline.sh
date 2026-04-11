#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Unified entry point for the three-stage data pipeline.
#
# Usage:
#   bash data_pipeline/run_pipeline.sh \
#       --source_ip celebrity \
#       --ip_data data/ip_prompts.json \
#       --output_dir ./output/pipeline
#
# All arguments are forwarded to pipeline_unify_data.py.
# This script performs pre-flight environment checks before launching.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Pre-flight checks ──────────────────────────────────────────────────────
info "Unify-Agent Data Pipeline"
echo "============================================================"

# Python
command -v python3 &>/dev/null || error "python3 not found in PATH"

# Required API keys
[[ -n "${OPENAI_API_KEY:-}" ]]  || error "OPENAI_API_KEY is not set"
[[ -n "${SERPER_API_KEY:-}" ]]  || warn  "SERPER_API_KEY is not set (text/image search will fail)"

# Optional keys
[[ -n "${JINA_API_KEY:-}" ]]    || warn  "JINA_API_KEY is not set (will fall back to snippets)"

info "Environment"
echo "  OPENAI_API_KEY   : ${OPENAI_API_KEY:0:8}..."
echo "  OPENAI_BASE_URL  : ${OPENAI_BASE_URL:-<default>}"
echo "  SERPER_API_KEY   : ${SERPER_API_KEY:+set}${SERPER_API_KEY:-<unset>}"
echo "  JINA_API_KEY     : ${JINA_API_KEY:+set}${JINA_API_KEY:-<unset>}"
echo ""
echo "  Stage 1 model    : ${STAGE1_MODEL:-gpt-4o}"
echo "  Stage 2 reasoning: ${AGENT_REASONING_MODEL:-gpt-4o}"
echo "  Stage 2 multiturn: ${AGENT_MULTITURN_MODEL:-gpt-4o}"
echo "  Stage 3 image gen: ${IMAGE_GEN_MODEL:-gpt-image-1}"
echo "============================================================"

# ── Launch ──────────────────────────────────────────────────────────────────
info "Launching pipeline_unify_data.py with args: $*"
exec python3 "${SCRIPT_DIR}/pipeline_unify_data.py" "$@"
