#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter (rsLoRA or plain) to a llama.cpp LoRA GGUF that loads with a plain
`--lora` at the trained scale.

llama.cpp scales LoRA by alpha / r, while rsLoRA trains at alpha / sqrt(r). This folds the trained
scale into an effective alpha (alpha_eff = scale * r) in a temporary copy of the adapter config, then
runs llama.cpp's convert_lora_to_gguf.py. The adapter weights themselves are untouched.

Usage:
  python lora_to_gguf.py --llama-cpp path/to/llama.cpp --base path/to/Ripple-65B-Base-Preview \
      --adapter path/to/adapter --outfile adapter-f16.gguf [--outtype f16]
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--llama-cpp", required=True)
ap.add_argument("--base", required=True, help="base model directory (HF layout)")
ap.add_argument("--adapter", required=True, help="PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
ap.add_argument("--outfile", required=True)
ap.add_argument("--outtype", default="f16")
a = ap.parse_args()

src, out = os.path.abspath(a.adapter), os.path.abspath(a.outfile)
cfg = json.load(open(os.path.join(src, "adapter_config.json")))
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
cfg["lora_alpha"], cfg["use_rslora"] = scale * r, False
tmp = tempfile.mkdtemp(dir=os.path.dirname(out) or ".")
try:
    os.symlink(os.path.join(src, "adapter_model.safetensors"), os.path.join(tmp, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(tmp, "adapter_config.json"), "w"))
    rc = subprocess.run([sys.executable, os.path.join(os.path.abspath(a.llama_cpp), "convert_lora_to_gguf.py"),
                         "--base", os.path.abspath(a.base), "--outtype", a.outtype, "--outfile", out, tmp])
finally:
    shutil.rmtree(tmp)
if rc.returncode != 0 or not os.path.exists(out):
    sys.exit("conversion failed")
print(f"LORA_GGUF_OK {out} r={r} alpha={alpha} scale={scale:.6f} alpha_eff={scale * r:.4f}")
