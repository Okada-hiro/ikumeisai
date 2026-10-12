"""Patch vllm_omni fused_code_predictor.py: ROCm skinny GEMM (wvSplitK) for M<=5 GEMMs. Run from the qwen3_tts dir."""
import shutil, os

p = "fused_code_predictor.py"
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
s = open(p + ".orig").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)


rep("from .tokenizer_12hz.streaming_decoder import _silu_mul_kernel\n",
    '''from .tokenizer_12hz.streaming_decoder import _silu_mul_kernel

# ikumeisai: on ROCm, M<=5 GEMMs via hipBLASLt (MT128x128 tiles) reach only ~80-190 GB/s on gfx1151;
# vLLM's skinny GEMM (wvSplitK) reaches ~250 GB/s. Weights are also kept as [N, K] for it.
_SKINNY = bool(torch.version.hip) and __import__("os").environ.get("OMNI_CP_SKINNY", "1") == "1"
_CU = torch.cuda.get_device_properties(0).multi_processor_count if _SKINNY else 0
if _SKINNY:
    from vllm import _custom_ops as _ops


def _mm(x, w_kn, w_nk, bias=None):
    if _SKINNY and x.shape[0] <= 5 and x.is_contiguous():
        return _ops.wvSplitK(w_nk, x, _CU, bias)
    return torch.addmm(bias, x, w_kn) if bias is not None else torch.mm(x, w_kn)
''')
rep("        self.proj_wt = c(pw.t()) if pw is not None else None\n",
    "        self.proj_wt = c(pw.t()) if pw is not None else None\n"
    "        self.proj_w = c(pw) if pw is not None else None\n")
rep('''                    down=c(f32(layer.mlp.down_proj.weight).t()),
                )''', '''                    down=c(f32(layer.mlp.down_proj.weight).t()),
                    qkv_nk=c(f32(a.qkv_proj.weight)),
                    o_nk=c(f32(a.o_proj.weight)),
                    gu_nk=c(f32(layer.mlp.gate_up_proj.weight)),
                    down_nk=c(f32(layer.mlp.down_proj.weight)),
                )''')
rep("        self.lm_heads = [c(f32(h.weight).t()) for h in predictor.lm_head]\n",
    "        self.lm_heads = [c(f32(h.weight).t()) for h in predictor.lm_head]\n"
    "        self.lm_heads_nk = [c(f32(h.weight)) for h in predictor.lm_head]\n")
rep('        qkv = torch.mm(h, L["qkv"])\n', '        qkv = _mm(h, L["qkv"], L["qkv_nk"])\n')
rep('        x.addmm_(att, L["o"])\n',
    '        if _SKINNY and rows <= 5:\n'
    '            x.add_(_mm(att, L["o"], L["o_nk"]))\n'
    '        else:\n'
    '            x.addmm_(att, L["o"])\n')
rep('        gu = torch.mm(h, L["gu"])\n', '        gu = _mm(h, L["gu"], L["gu_nk"])\n')
rep('        x.addmm_(a, L["down"])\n',
    '        if _SKINNY and rows <= 5:\n'
    '            x.add_(_mm(a, L["down"], L["down_nk"]))\n'
    '        else:\n'
    '            x.addmm_(a, L["down"])\n')
rep("        x = torch.addmm(self.proj_b, inp, self.proj_wt) if self.proj_wt is not None else inp.clone()\n",
    "        x = _mm(inp.contiguous(), self.proj_wt, self.proj_w, self.proj_b) if self.proj_wt is not None else inp.clone()\n")
rep("            logits = torch.mm(hf, self.lm_heads[step - 1])\n",
    "            logits = _mm(hf, self.lm_heads[step - 1], self.lm_heads_nk[step - 1])\n")
open(p, "w").write(s)
print("patched")
