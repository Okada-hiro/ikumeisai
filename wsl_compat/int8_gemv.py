# ikumeisai: int8 weight-only (W8A16, per-output-channel scale) GEMV for decode on gfx1151.
# Decode GEMMs (<= 5 rows) are memory-bound; reading int8 weights halves the bytes. Larger M
# (prefill) keeps the bf16 weight and F.linear. Configs from a sweep with cache-cold weights.
from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

MAX_ROWS = 5


@triton.jit
def _w8a16_gemv_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, BN: tl.constexpr, BK: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = rn < N
    s = tl.load(s_ptr + rn, mask=nmask, other=0.0)
    for m in range(M):
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], mask=nmask[:, None], other=0)
            x = tl.load(x_ptr + m * K + rk)
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
        tl.store(out_ptr + m * N + rn, (acc * s).to(out_ptr.dtype.element_ty), mask=nmask)


def _cfg(n: int, k: int) -> tuple[int, int, int] | None:
    if k % 2048 == 0:
        return (4 if n >= 8192 else 2), 2048, (8 if k >= 6144 else 4)
    if k % 1024 == 0:
        return 8, 1024, 4
    if k % 256 == 0:
        return 16, 256, 4
    return None


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] float weight -> (int8 [N, K], fp32 per-row scale [N])."""
    wf = w.detach().float()
    s = (wf.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
    q = (wf / s[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
    return q, s.contiguous()


def supported(w: torch.Tensor) -> bool:
    return w.dim() == 2 and _cfg(w.shape[0], w.shape[1]) is not None


def gemv(x2d: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, k = x2d.shape
    n = q.shape[0]
    bn, bk, warps = _cfg(n, k)
    out = torch.empty(m, n, device=x2d.device, dtype=x2d.dtype)
    _w8a16_gemv_kernel[(triton.cdiv(n, bn),)](x2d, q, s, out, m, n, k, BN=bn, BK=bk, num_warps=warps)
    return out


@torch.library.custom_op("ikumeisai::linear_w8", mutates_args=())
def linear_w8(x: torch.Tensor, w: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.shape[0] <= MAX_ROWS:
        out = gemv(x2d.contiguous(), q, s)
    else:
        out = F.linear(x2d, w)
    return out.reshape(*x.shape[:-1], w.shape[0])


@linear_w8.register_fake
def _(x, w, q, s):
    return x.new_empty(*x.shape[:-1], w.shape[0])


def quantize_vllm_linears(root: torch.nn.Module, logger=None) -> int:
    """Attach int8 copies to every unquantized vLLM LinearBase under ``root`` and route decode through them."""
    from vllm.model_executor.layers.linear import LinearBase

    count = 0
    for name, mod in root.named_modules():
        if not isinstance(mod, LinearBase) or type(mod.quant_method).__name__ != "UnquantizedLinearMethod":
            continue
        w = getattr(mod, "weight", None)
        if w is None or not supported(w):
            continue
        q, s = quantize(w)
        mod.register_buffer("w8_q", q, persistent=False)
        mod.register_buffer("w8_s", s, persistent=False)

        def apply(layer, x, bias=None):
            out = torch.ops.ikumeisai.linear_w8(x, layer.weight, layer.w8_q, layer.w8_s)
            return out + bias if bias is not None else out

        mod.quant_method.apply = apply
        count += 1
    if logger is not None:
        logger.info("ikumeisai: int8 weight-only decode GEMV on %d linear layers", count)
    return count
