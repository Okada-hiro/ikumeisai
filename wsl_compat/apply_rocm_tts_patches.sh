#!/usr/bin/env bash
# vLLM-Omni (Qwen3-TTS) の ROCm/gfx1151 向けパッチ。各ファイルは .orig にバックアップ。
#  1) code predictor の frame-local KV / fused predictor を ROCm でも有効化 (is_cuda ゲート)
#  2) code2wav: OMNI_DISABLE_MIOPEN_CONV=1 で MIOpen conv を無効化 (gfx1151 で naive_conv に落ちるため)
#  3) fused predictor の M<=5 GEMM を vLLM の wvSplitK (skinny GEMM) に置換
set -euo pipefail
D=$(~/venvs/vllm-omni/bin/python -c "import vllm_omni,os;print(os.path.dirname(vllm_omni.__file__))")/model_executor/models/qwen3_tts
cd "$D"
F=qwen3_tts_code_predictor_vllm.py; [ -f $F.orig ] || cp $F $F.orig; cp $F.orig $F
sed -i "s/current_omni_platform\.is_cuda()$/(current_omni_platform.is_cuda() or current_omni_platform.is_rocm())/" $F
F=qwen3_tts_code2wav.py; [ -f $F.orig ] || cp $F $F.orig
python3 - <<"PY"
p = "qwen3_tts_code2wav.py"; s = open(p + ".orig").read()
a = "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
assert s.count(a) == 1
s = s.replace(a, a + "        # ikumeisai: gfx1151 MIOpen picks naive_conv (double accum) for some decoder convs; use ATen native conv.\n"
    "        if os.environ.get(\"OMNI_DISABLE_MIOPEN_CONV\") == \"1\":\n            torch.backends.cudnn.enabled = False\n"
    "            logger.info(\"code2wav: MIOpen convolution disabled (ATen native conv)\")\n")
open(p, "w").write(s)
PY
python3 "$(dirname "$0")/patch_cp_skinny.py" 2>/dev/null || python3 ~/workspace/ikumeisai/wsl_compat/patch_cp_skinny.py
echo "patched $D"
