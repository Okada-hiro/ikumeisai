#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B-Instruct を vLLM (ROCm, native Linux) で起動 → /v1/chat/completions
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/vllm/bin/activate
# gfx1151 向けに調整した fused MoE の設定 (wsl_compat/tune_moe.py で生成)
export VLLM_TUNED_CONFIG_FOLDER="$PWD/moe_configs"

exec vllm serve "${OMNI_MODEL:-$HOME/models/Qwen3-Omni-30B-A3B-Instruct}" \
  --served-model-name qwen3-omni \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 \
  --limit-mm-per-prompt '{"audio": 1, "image": 0, "video": 0}' \
  --trust-remote-code
