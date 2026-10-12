"""Tune vLLM Triton fused-MoE configs for Qwen3-Omni-30B-A3B int8 experts on gfx1151.

    --mode w8a8   : int8 weights (per-channel) x int8 activations (per-token, dynamic)
    --mode w8a16  : int8 weights (group 128) x bf16 activations
Writes E=128,N=768,device_name=AMD_Radeon_8060S,dtype=int8_<mode>.json into moe_configs/.
"""
import argparse, itertools, json, time
from pathlib import Path
import torch
from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe import config as C

E, H, I, TOPK, G = 128, 2048, 768, 8, 128
OUT_DIR = Path(__file__).resolve().parent.parent / "moe_configs"

def space(m, mode):
    tokens = m * TOPK
    bm = [16, 32] if tokens <= 64 else [16, 32, 64] if tokens <= 1024 else [32, 64, 128]
    bk = [32, 64, 128, 256] if mode == "w8a8" else [64, 128]  # w8a16: BLOCK_K must tile the 128 group
    for BM, BN, BK, Gm, W, WPE in itertools.product(bm, [16, 32, 64, 128], bk, [1, 8], [1, 2, 4, 8], [0, 2, 4]):
        yield {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": Gm, "SPLIT_K": 1,
               "num_warps": W, "num_stages": 2, "waves_per_eu": WPE}

def bench(cfg, x, w1, w2, tw, ti, qc, iters):
    with override_config(cfg):
        fused_experts(x, w1, w2, tw, ti, quant_config=qc); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fused_experts(x, w1, w2, tw, ti, quant_config=qc)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["w8a8", "w8a16"], required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    a = ap.parse_args()
    w1 = torch.randint(-127, 128, (E, 2 * I, H), dtype=torch.int8, device="cuda")
    w2 = torch.randint(-127, 128, (E, H, I), dtype=torch.int8, device="cuda")
    if a.mode == "w8a8":
        qc = C.int8_w8a8_moe_quant_config(torch.rand(E, 2 * I, 1, device="cuda") * 1e-3,
                                          torch.rand(E, H, 1, device="cuda") * 1e-3, None, None, per_act_token_quant=True)
    else:
        qc = C.int8_w8a16_moe_quant_config(torch.rand(E, 2 * I, H // G, device="cuda", dtype=torch.bfloat16) * 1e-3,
                                           torch.rand(E, H, I // G, device="cuda", dtype=torch.bfloat16) * 1e-3,
                                           block_shape=[0, G])
    out = OUT_DIR / f"E={E},N={I},device_name=AMD_Radeon_8060S,dtype=int8_{a.mode}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for m in a.batch_sizes:
        x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
        tw, ti, _ = fused_topk(x, torch.randn(m, E, device="cuda"), TOPK, renormalize=True)
        best, best_us, t0 = None, float("inf"), time.time()
        for cfg in space(m, a.mode):
            try: us = bench(cfg, x, w1, w2, tw, ti, qc, 20 if m <= 256 else 5)
            except Exception: continue
            if us < best_us: best, best_us = cfg, us
        res[str(m)] = best
        out.write_text(json.dumps(dict(sorted(res.items(), key=lambda kv: int(kv[0]))), indent=4))
        print(f"M={m}: best {best_us:.1f} us/layer ({time.time()-t0:.0f}s) {best}", flush=True)
    print("wrote", out)

if __name__ == "__main__":
    main()
