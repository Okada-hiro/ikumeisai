# Source before running anything from ~/venvs/vllm on WSL2 + librocdxg.
#   HSA_ENABLE_DXG_DETECTION   : use /dev/dxg (librocdxg) instead of /dev/kfd
#   ROCPROFILER_REGISTER_ENABLED=0 : rocprofiler-sdk aborts without /sys/class/kfd
#   ~/.local/lib/vllm-compat   : libdw.so.1 (from libdw1t64 deb) and a stub
#                                libmpi_cxx.so.40 (OpenMPI 4 C++ bindings, absent in
#                                Ubuntu 26.04's OpenMPI 5) required by vLLM's torch
export HSA_ENABLE_DXG_DETECTION=1
export ROCPROFILER_REGISTER_ENABLED=0
export LD_LIBRARY_PATH="$HOME/.local/lib/vllm-compat:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
#   IKU_DISABLE_MIOPEN_CONV=1  : runtime/sitecustomize.py turns off MIOpen conv (slow
#                                per-shape kernel search on gfx1151); see that file
export IKU_DISABLE_MIOPEN_CONV=1
_IKU_RUNTIME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runtime"
export PYTHONPATH="${_IKU_RUNTIME}${PYTHONPATH:+:$PYTHONPATH}"
