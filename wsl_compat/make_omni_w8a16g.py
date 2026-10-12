"""Make ~/models/Qwen3-Omni-30B-A3B-Instruct-w8a16 (serve_vllm.sh default): thinker only, MoE experts int8
weight-only, symmetric group 128, compressed-tensors pack-quantized (W8A16). Other weights stay bf16.
Run: ~/venvs/vllm/bin/python wsl_compat/make_omni_w8a16g.py  (uses the GPU; ~10 min)"""
import json, re, shutil, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
G = 128
SRC = Path.home() / "models/Qwen3-Omni-30B-A3B-Instruct"
OUT = Path.home() / "models/Qwen3-Omni-30B-A3B-Instruct-w8a16"
EXPERT = re.compile(r"^thinker\..*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")
shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
files = sorted({f for k, f in wm.items() if k.startswith("thinker.")})
index, buf, size, ns, nq, err = {}, {}, 0, 0, 0, 0.0
def flush():
    global buf, size, ns
    if not buf: return
    ns += 1; name = f"model-{ns:05d}.safetensors"; save_file(buf, str(OUT / name), metadata={"format": "pt"})
    for k in buf: index[k] = name
    buf, size = {}, 0
t0 = time.time()
for f in files:
    with safe_open(str(SRC / f), "pt") as fh:
        for k in fh.keys():
            if not k.startswith("thinker."): continue
            t = fh.get_tensor(k)
            if EXPERT.match(k):
                w = t.to("cuda", torch.float32); o, i = w.shape
                wg = w.view(o, i // G, G)
                s = (wg.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12).to(torch.bfloat16).float()
                q = torch.round(wg / s).clamp_(-127, 127).to(torch.int8)
                err = max(err, ((q.float() * s - wg).abs().max() / w.abs().max().clamp_min(1e-12)).item())
                b = k[:-len("weight")]
                buf[b + "weight_packed"] = pack_to_int32(q.view(o, i), 8).cpu()
                buf[b + "weight_scale"] = s.view(o, i // G).to(torch.bfloat16).cpu()
                buf[b + "weight_shape"] = torch.tensor([o, i]); nq += 1; size += o * i
            else:
                buf[k] = t; size += t.numel() * t.element_size()
            if size >= 4 * 1024**3: flush()
    print(f, f"{time.time()-t0:.0f}s", flush=True)
flush()
(OUT / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
for p in SRC.iterdir():
    if p.suffix in (".json", ".txt", ".jinja") and p.name != "model.safetensors.index.json": shutil.copy2(p, OUT / p.name)
cfg = json.loads((OUT / "config.json").read_text())
cfg["quantization_config"] = {"quant_method": "compressed-tensors", "format": "pack-quantized", "version": "0.17.0",
    "quantization_status": "compressed", "kv_cache_scheme": None, "ignore": [],
    "config_groups": {"group_0": {"targets": ["RoutedExperts"], "input_activations": None, "output_activations": None,
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "group", "group_size": G,
                    "observer": "minmax", "observer_kwargs": {}, "dynamic": False, "actorder": None, "block_structure": None}}}}
cfg["enable_audio_output"] = False
(OUT / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
print(f"quantized {nq}, max rel err {err:.5f}, {time.time()-t0:.0f}s")
