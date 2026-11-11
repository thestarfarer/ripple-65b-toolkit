#!/usr/bin/env python3
"""Stage 1 of the LoRA plugin certification (run in the vLLM environment).

Computes teacher-forced per-token logprobs for the BASE model and for
BASE+adapter through vLLM with ripple_lora_plugin registered, on texts you
supply. Writes them to a JSON file that stage 2 (HF/PEFT environment)
compares against its own reference. Two stages exist because vLLM and the
pinned PEFT/transformers stack generally cannot share one environment.

Usage:
  python validate_vllm_stage.py --model ricursiveAI/Ripple-65B-Base-Preview \
      --revision <pin> --adapter /path/to/adapter --texts texts.jsonl \
      --out vllm_lp.json [--tp 2] [--max-lora-rank 64] [--max-texts 8]

--texts: a .jsonl with a "text" field per line, or a plain .txt
(one document per line). ~500 tokens per text is plenty.
"""
import argparse, json
import ripple_lora_plugin; ripple_lora_plugin.register()
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--revision", default=None)
p.add_argument("--adapter", required=True)
p.add_argument("--texts", required=True)
p.add_argument("--out", default="vllm_lp.json")
p.add_argument("--tp", type=int, default=2)
p.add_argument("--max-lora-rank", type=int, default=64)
p.add_argument("--max-texts", type=int, default=8)
p.add_argument("--max-tokens-per-text", type=int, default=480)
a = p.parse_args()

if a.texts.endswith(".jsonl"):
    texts = [json.loads(l)["text"] for l in open(a.texts)][: a.max_texts]
else:
    texts = [l.rstrip("\n") for l in open(a.texts) if l.strip()][: a.max_texts]

llm = LLM(model=a.model, revision=a.revision, tensor_parallel_size=a.tp,
          max_model_len=4608, gpu_memory_utilization=0.92, trust_remote_code=True,
          enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=1)
tok = llm.get_tokenizer()
ids = [tok.encode(t, add_special_tokens=False)[: a.max_tokens_per_text] for t in texts]
sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
res = {"ids": ids}
for tag, lreq in (("base", None), ("lora", LoRARequest("a", 1, a.adapter))):
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, lora_request=lreq)
    res[tag] = [[None] + [o.prompt_logprobs[j][x[j]].logprob for j in range(1, len(x))]
                for x, o in zip(ids, outs)]
json.dump(res, open(a.out, "w"))
print("VLLM_STAGE_DONE ->", a.out)
