"""Tune vLLM's Triton fused-MoE kernel for Qwen3-Omni-30B-A3B (bf16) on gfx1151.

vLLM ships no bf16 config for E=128,N=768 on Radeon 8060S and falls back to a default
that reaches only ~59 GB/s at decode (M=1). This searches an RDNA-sized space (no MFMA
knobs) per batch size and writes the vLLM config JSON.

Usage (vllm venv, wsl_compat/env.sh sourced):
    python wsl_compat/tune_moe.py [--batch-sizes 1 2 4 ...]
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import vllm
from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

E, H, I, TOPK = 128, 2048, 768, 8  # Qwen3-Omni-30B-A3B thinker
DEVICE_NAME = "AMD_Radeon_8060S"  # must match current_platform.get_device_name()
OUT = (Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
       / f"E={E},N={I},device_name={DEVICE_NAME}.json")


def space(m: int):
    tokens = m * TOPK
    bm = [16, 32] if tokens <= 64 else [16, 32, 64] if tokens <= 1024 else [32, 64, 128]
    for BM, BN, BK, G, W, WPE in itertools.product(
        bm, [16, 32, 64, 128], [32, 64, 128, 256], [1, 8], [1, 2, 4, 8], [0, 2, 4]
    ):
        yield {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK,
               "GROUP_SIZE_M": G, "SPLIT_K": 1, "num_warps": W, "num_stages": 2,
               "waves_per_eu": WPE}


def bench(cfg, x, w1, w2, tw, ti, iters):
    with override_config(cfg):
        fused_experts(x, w1, w2, tw, ti)  # compile / warm
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fused_experts(x, w1, w2, tw, ti)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    args = ap.parse_args()

    dt = torch.bfloat16
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=dt) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=dt) * 0.02
    result = json.loads(OUT.read_text()) if OUT.exists() else {}

    for m in args.batch_sizes:
        x = torch.randn(m, H, device="cuda", dtype=dt)
        tw, ti, _ = fused_topk(x, torch.randn(m, E, device="cuda"), TOPK, renormalize=True)
        iters = 20 if m <= 256 else 5
        best, best_us, t0 = None, float("inf"), time.time()
        for cfg in space(m):
            try:
                us = bench(cfg, x, w1, w2, tw, ti, iters)
            except Exception:
                continue
            if us < best_us:
                best, best_us = cfg, us
        result[str(m)] = best
        OUT.write_text(json.dumps(dict(sorted(result.items(), key=lambda kv: int(kv[0]))), indent=4))
        print(f"M={m}: best {best_us:.1f} us/layer ({time.time() - t0:.0f}s search) {best}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
