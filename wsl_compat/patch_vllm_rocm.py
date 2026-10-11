"""WSL2 (librocdxg) has no amdgpu kernel driver, so amdsmi_init() fails with
AMDSMI_STATUS_DRIVER_NOT_LOADED. Patch vllm/platforms/rocm.py so that every
@with_amdsmi_context function falls back to torch.cuda / safe defaults.

Usage: python wsl_compat/patch_vllm_rocm.py   (run inside the vllm venv; idempotent)
"""
import importlib.util
import shutil
from pathlib import Path

MARK = "# --- WSL amdsmi fallback (ikumeisai) ---"

OLD = '''def with_amdsmi_context(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper
'''

NEW = MARK + '''
def _wsl_amdsmi_fallback(name, args):
    device_id = args[1] if len(args) > 1 else 0
    if name == "get_device_name":
        # same name amdsmi would give via _ROCM_DEVICE_ID_NAME_MAP, so that
        # fused_moe/configs/*device_name=AMD_Radeon_8060S*.json are found
        if "gfx1151" in torch.cuda.get_device_properties(device_id).gcnArchName:
            return "AMD_Radeon_8060S"
        return torch.cuda.get_device_name(device_id)
    if name == "get_device_uuid":
        return str(torch.cuda.get_device_properties(device_id).uuid)
    if name == "is_fully_connected":
        return True
    if name == "get_all_device_numa_nodes":
        return None
    raise RuntimeError(f"amdsmi unavailable (WSL) for {name}")


def with_amdsmi_context(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            amdsmi_init()
        except Exception:
            return _wsl_amdsmi_fallback(fn.__name__, args)
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper
'''


def main():
    path = Path(importlib.util.find_spec("vllm").origin).parent / "platforms" / "rocm.py"
    orig = path.with_suffix(".py.orig")
    src = path.read_text()
    if NEW in src:
        print(f"already patched: {path}")
        return
    if MARK in src:  # older version of this patch: re-apply from the original
        src = orig.read_text()
    if OLD not in src:
        raise SystemExit(f"unexpected rocm.py layout, not patched: {path}")
    if not orig.exists():
        shutil.copy2(path, orig)
    path.write_text(src.replace(OLD, NEW))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
