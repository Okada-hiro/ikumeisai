#!/usr/bin/env bash
# Qwen3-TTS-12Hz-1.7B-Base (voice clone) を vLLM-Omni で起動 → /v1/audio/speech
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/vllm-omni/bin/activate
# gfx1151: MIOpen が naive conv に落ちるため code2wav は ATen conv を使う (wsl_compat/apply_rocm_tts_patches.sh 適用前提)
export OMNI_DISABLE_MIOPEN_CONV=${OMNI_DISABLE_MIOPEN_CONV:-1}
# talker / code predictor のデコード GEMM を int8 weight-only (W8A16) で実行 (0 で bf16)
export OMNI_TALKER_INT8=${OMNI_TALKER_INT8:-1}
export OMNI_CP_INT8=${OMNI_CP_INT8:-1}
O=$(python -c "import vllm_omni,os;print(os.path.dirname(vllm_omni.__file__))")

exec vllm serve "$HOME/models/Qwen3-TTS-12Hz-1.7B-Base" \
  --omni \
  --deploy-config "${TTS_DEPLOY:-$PWD/deploy_rocm/qwen3_tts_rocm_fused.yaml}" \
  --stage-overrides '{"0": {"gpu_memory_utilization": 0.06}, "1": {"gpu_memory_utilization": 0.06}}' \
  --served-model-name qwen3-tts \
  --host 127.0.0.1 --port 8001 \
  --trust-remote-code
