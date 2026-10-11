#!/usr/bin/env bash
# Qwen3-TTS-12Hz-1.7B-Base (voice clone) を vLLM-Omni で起動 → /v1/audio/speech
set -euo pipefail
cd "$(dirname "$0")"
source wsl_compat/env.sh
source ~/venvs/vllm-omni/bin/activate
O=$(python -c "import vllm_omni,os;print(os.path.dirname(vllm_omni.__file__))")

exec vllm serve "$HOME/models/Qwen3-TTS-12Hz-1.7B-Base" \
  --omni \
  --deploy-config "$O/deploy/qwen3_tts.yaml" \
  --stage-overrides '{"0": {"gpu_memory_utilization": 0.06}, "1": {"gpu_memory_utilization": 0.06}}' \
  --served-model-name qwen3-tts \
  --host 127.0.0.1 --port 8001 \
  --trust-remote-code
