"""Install int8_gemv.py into vllm_omni qwen3_tts and hook it (env-gated). Run from the qwen3_tts dir after patch_cp_skinny.py.

OMNI_TALKER_INT8=1: talker Qwen3Model linears use int8 weights for decode (<=5 rows).
OMNI_CP_INT8=1:     fused code predictor GEMMs use int8 weights.
"""
import os
import shutil

here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "int8_gemv.py"), "int8_gemv.py")

# talker: hook at the end of load_weights
p = "qwen3_tts_talker.py"
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
s = open(p + ".orig").read()
a = '''            logger.info("Qwen3-TTS Talker stream decode enabled (%d slots)", slots)
        return loaded
'''
assert s.count(a) == 1
s = s.replace(a, '''            logger.info("Qwen3-TTS Talker stream decode enabled (%d slots)", slots)
        if __import__("os").environ.get("OMNI_TALKER_INT8") == "1":
            from .int8_gemv import quantize_vllm_linears

            quantize_vllm_linears(self.model, logger)
        return loaded
''')
open(p, "w").write(s)

# fused code predictor: int8 branch inside _mm (file already patched by patch_cp_skinny.py)
p = "fused_code_predictor.py"
s = open(p).read()
assert "_SKINNY" in s, "run patch_cp_skinny.py first"
a = '''def _mm(x, w_kn, w_nk, bias=None):
    if _SKINNY and x.shape[0] <= 5 and x.is_contiguous():
'''
assert s.count(a) == 1
s = s.replace(a, '''_INT8 = __import__("os").environ.get("OMNI_CP_INT8") == "1"
_Q8: dict = {}


def _mm(x, w_kn, w_nk, bias=None):
    if _INT8 and x.shape[0] <= 5:
        q = _Q8.get(w_nk.data_ptr())
        if q is None:
            from .int8_gemv import quantize, supported

            q = _Q8[w_nk.data_ptr()] = quantize(w_nk) if supported(w_nk) else False
        if q is not False:
            from .int8_gemv import gemv

            out = gemv(x.contiguous(), q[0], q[1])
            return out + bias if bias is not None else out
    if _SKINNY and x.shape[0] <= 5 and x.is_contiguous():
''')
open(p, "w").write(s)
print("patched talker + fused predictor for int8")
