#!/usr/bin/env python3
"""Stage 2 of the LoRA plugin certification (run in the pinned HF/PEFT env).

Recomputes the same per-token logprobs through transformers+PEFT and compares
the ADAPTER DELTA (lora minus base logprob, per token) between the two
engines. Comparing deltas cancels engine-systematic drift, so a packed-module
slice misalignment cannot hide. PASS requires pearson r >= 0.97 and slope in
[0.9, 1.1] on the deltas.

Usage:
  python validate_hf_stage.py --model ricursiveAI/Ripple-65B-Base-Preview \
      --revision <pin> --adapter /path/to/adapter --vllm-lp vllm_lp.json
"""
import argparse, json
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--revision", default=None)
p.add_argument("--adapter", required=True)
p.add_argument("--vllm-lp", default="vllm_lp.json")
a = p.parse_args()
v = json.load(open(a.vllm_lp))

base = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16,
        device_map="balanced", trust_remote_code=True, attn_implementation="sdpa").eval()

@torch.no_grad()
def lps(model):
    out = []
    for x in v["ids"]:
        ids = torch.tensor([x], device=model.device)
        lg = model(input_ids=ids).logits.float().log_softmax(-1)
        out.append([None] + lg[0, :-1].gather(-1, ids[0, 1:, None])[:, 0].tolist())
    return out

hb = lps(base)
model = PeftModel.from_pretrained(base, a.adapter, is_trainable=False).eval()
hl = lps(model)
dv, dh, eb = [], [], []
for i in range(len(v["ids"])):
    for j in range(1, len(v["ids"][i])):
        dv.append(v["lora"][i][j] - v["base"][i][j])
        dh.append(hl[i][j] - hb[i][j])
        eb.append(abs(v["base"][i][j] - hb[i][j]))
n = len(dv)
mv, mh = sum(dv) / n, sum(dh) / n
cov = sum((x - mv) * (y - mh) for x, y in zip(dv, dh))
r = cov / (sum((y - mh) ** 2 for y in dh) * sum((x - mv) ** 2 for x in dv)) ** 0.5
slope = cov / sum((y - mh) ** 2 for y in dh)
ad = sorted(abs(x - y) for x, y in zip(dv, dh))
print("n=%d  pearson_r %.4f  slope %.4f  median|dv-dh| %.4f  p95 %.4f  base engine gap %.4f"
      % (n, r, slope, ad[n // 2], ad[int(n * 0.95)], sum(eb) / n))
print("LORA_PLUGIN_VALID" if r >= 0.97 and 0.9 <= slope <= 1.1 else "LORA_PLUGIN_INVALID")
