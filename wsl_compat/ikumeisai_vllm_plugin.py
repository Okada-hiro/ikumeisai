"""vLLM general plugin (ikumeisai): OMNI_DENSE_INT8=1 runs decode GEMMs of unquantized dense linears and the
lm_head with int8 weights (W8A16 Triton GEMV from wsl_compat/int8_gemv.py) on gfx1151."""
import importlib.util
import os
import sys

INT8_GEMV = os.path.expanduser("~/workspace/ikumeisai/wsl_compat/int8_gemv.py")


def register():
    if os.environ.get("OMNI_DENSE_INT8") != "1":
        return
    from vllm.logger import init_logger
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    logger = init_logger("ikumeisai_vllm_plugin")
    if getattr(BaseModelLoader.load_model, "_ikumeisai", False):
        return
    orig = BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = orig(self, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("ikumeisai_int8_gemv", INT8_GEMV)
        mod = sys.modules.get("ikumeisai_int8_gemv")
        if mod is None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["ikumeisai_int8_gemv"] = mod  # Dynamo re-imports traced functions' modules by name
            spec.loader.exec_module(mod)
        n = mod.quantize_vllm_linears(model, logger, include_lm_head=os.environ.get("OMNI_DENSE_INT8_LMHEAD", "0") == "1",
            # MoE router logits pick the experts; keep them bf16 unless asked.
            skip_suffixes=() if os.environ.get("OMNI_DENSE_INT8_GATE") == "1" else (".mlp.gate",))
        print(f"[ikumeisai] int8 decode GEMV on {n} layers of {type(model).__name__}", file=sys.stderr, flush=True)
        if os.environ.get("OMNI_DENSE_INT8_DUMP"):
            import collections

            from vllm.model_executor.layers.linear import LinearBase

            c = collections.Counter()
            for name, m in model.named_modules():
                if isinstance(m, LinearBase):
                    key = name.split(".layers.")[0] + " " + name.rsplit(".", 1)[-1] + " " + type(m.quant_method).__name__
                    c[(key, hasattr(m, "w8_q"))] += 1
            for (key, q8), v in sorted(c.items()):
                print(f"[ikumeisai] {v:4d} x {key} int8={q8}", file=sys.stderr, flush=True)
        return model

    load_model._ikumeisai = True
    BaseModelLoader.load_model = load_model
    print(f"[ikumeisai] plugin registered in pid {os.getpid()}", file=sys.stderr, flush=True)
