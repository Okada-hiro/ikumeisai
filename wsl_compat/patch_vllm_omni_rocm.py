"""vllm_omni detects ROCm only through amdsmi, which fails on WSL2 (no amdgpu
driver). Patch the platform plugin to fall back to torch's HIP build check.

Usage: python wsl_compat/patch_vllm_omni_rocm.py   (run inside the vllm-omni venv; idempotent)
"""
import importlib.util
import shutil
from pathlib import Path

MARK = "# --- WSL ROCm fallback (ikumeisai) ---"

OLD = '''    except Exception as e:
        logger.debug("ROCm OmniPlatform is not available because: %s", str(e))

    return "vllm_omni.platforms.rocm.platform.RocmOmniPlatform" if is_rocm else None
'''

NEW = '''    except Exception as e:
        logger.debug("ROCm OmniPlatform is not available because: %s", str(e))
        ''' + MARK + '''
        try:
            import torch

            is_rocm = torch.version.hip is not None and torch.cuda.is_available()
        except Exception:
            pass

    return "vllm_omni.platforms.rocm.platform.RocmOmniPlatform" if is_rocm else None
'''


def main():
    path = Path(importlib.util.find_spec("vllm_omni").origin).parent / "platforms" / "__init__.py"
    src = path.read_text()
    if MARK in src:
        print(f"already patched: {path}")
        return
    if OLD not in src:
        raise SystemExit(f"unexpected layout, not patched: {path}")
    shutil.copy2(path, path.with_suffix(".py.orig"))
    path.write_text(src.replace(OLD, NEW))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
