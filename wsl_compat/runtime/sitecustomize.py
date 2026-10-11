# Loaded automatically by every Python process when this directory is on PYTHONPATH
# (set by wsl_compat/env.sh).
#
# MIOpen (torch.backends.cudnn on ROCm) runs a kernel search on the first call for every
# new conv input shape: ~29 s per shape on gfx1151. vLLM's startup profiling feeds the
# Qwen3-Omni audio encoder many shapes, so startup never finishes in practice.
# PyTorch's native conv is ~0.03-0.16 s for the same shapes, so disable MIOpen conv.
import os

if os.environ.get("IKU_DISABLE_MIOPEN_CONV") == "1":
    try:
        import torch

        torch.backends.cudnn.enabled = False
    except Exception:
        pass
