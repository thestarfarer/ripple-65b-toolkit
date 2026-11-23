#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into Ripple's base weights while llama.cpp converts them to GGUF.

The base checkpoint is streamed tensor by tensor through llama.cpp's convert_hf_to_gguf. Every
tensor the adapter targets gets  W + scale * (B @ A)  added in fp32 and is cast back to the
base dtype before it is written, so no merged Hugging Face copy is ever stored on disk.
scale = alpha / sqrt(r) for rsLoRA adapters, alpha / r otherwise.

Usage:
  python merge_lora_to_gguf.py --llama-cpp LLAMA_CPP_DIR --adapter ADAPTER_DIR \
      BASE_MODEL_DIR --outtype q8_0 --no-lazy --outfile merged-Q8_0.gguf [--split-max-size 45G]

Every argument after the two wrapper flags is passed to convert_hf_to_gguf unchanged.
The run ends with a MERGE_OK line only if every adapter tensor pair was applied.
"""
import argparse
import atexit
import json
import math
import os
import re
import sys

wrapper = argparse.ArgumentParser(add_help=False)
wrapper.add_argument("--llama-cpp", required=True, help="llama.cpp checkout with RicursiveForCausalLM support")
wrapper.add_argument("--adapter", required=True, help="PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
args, convert_argv = wrapper.parse_known_args()

llama_cpp = os.path.abspath(args.llama_cpp)
sys.path.insert(0, llama_cpp)
sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))

from safetensors import safe_open  # noqa: E402

cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
r, alpha = cfg["r"], cfg["lora_alpha"]
SCALE = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r

lora = safe_open(os.path.join(args.adapter, "adapter_model.safetensors"), "pt")
pairs = {}
for key in lora.keys():
    m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight$", key)
    if not m:
        raise SystemExit(f"unexpected adapter key {key!r}")
    pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = key
if any(set(v) != {"A", "B"} for v in pairs.values()):
    raise SystemExit("adapter has an unpaired lora_A / lora_B tensor")

merged = []

import conversion.base as base  # noqa: E402

_original_get_tensors = base.ModelBase.get_tensors


def _merged_get_tensors(self):
    for name, tensor in _original_get_tensors(self):
        if name in pairs:
            a = lora.get_tensor(pairs[name]["A"]).float()
            b = lora.get_tensor(pairs[name]["B"]).float()
            if (b.shape[0], a.shape[1]) != tuple(tensor.shape):
                raise SystemExit(f"shape mismatch on {name}: W {tuple(tensor.shape)} vs B@A {(b.shape[0], a.shape[1])}")
            dtype = tensor.dtype
            tensor = (tensor.float() + SCALE * (b @ a)).to(dtype)
            merged.append(name)
        yield name, tensor


base.ModelBase.get_tensors = _merged_get_tensors


@atexit.register
def _report():
    status = "MERGE_OK" if len(merged) == len(pairs) else "MERGE_INCOMPLETE"
    print(f"{status} merged={len(merged)} expected={len(pairs)} scale={SCALE:.6f} r={r} alpha={alpha} "
          f"rslora={bool(cfg.get('use_rslora'))}", flush=True)


sys.argv = [os.path.join(llama_cpp, "convert_hf_to_gguf.py")] + convert_argv
import convert_hf_to_gguf  # noqa: E402

convert_hf_to_gguf.main()
