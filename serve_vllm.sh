#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B-Instruct を vLLM (ROCm / WSL2 librocdxg) で起動 → /v1/chat/completions
set -euo pipefail
cd "$(dirname "$0")"
source wsl_compat/env.sh
source ~/venvs/vllm/bin/activate

exec vllm serve "$HOME/models/Qwen3-Omni-30B-A3B-Instruct" \
  --served-model-name qwen3-omni \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 \
  --limit-mm-per-prompt '{"audio": 1, "image": 0, "video": 0}' \
  --trust-remote-code
