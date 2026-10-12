#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B-Instruct を vLLM (ROCm, native Linux) で起動 → /v1/chat/completions
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/vllm/bin/activate
# gfx1151 向けに調整した fused MoE の設定 (wsl_compat/tune_moe.py で生成)
export VLLM_TUNED_CONFIG_FOLDER="$PWD/moe_configs"
# attention の qkv/o のデコード GEMM を int8 weight-only で実行 (wsl_compat/install_vllm_plugin.sh で
# vLLM プラグインを登録しておくこと)。MoE の router gate と lm_head は精度のため bf16 のまま。
export OMNI_DENSE_INT8=${OMNI_DENSE_INT8:-1}
# コンパイルキャッシュはモデルコードのハッシュで引かれ、上記の差し替えを反映しないため無効化
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}

exec vllm serve "${OMNI_MODEL:-$HOME/models/Qwen3-Omni-30B-A3B-Instruct-w8a16}" \
  --served-model-name qwen3-omni \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 \
  --limit-mm-per-prompt '{"audio": 1, "image": 0, "video": 0}' \
  --trust-remote-code
