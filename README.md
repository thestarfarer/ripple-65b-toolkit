# ripple-65b-toolkit

Tooling for training and serving LoRA adapters on
[ricursiveAI/Ripple-65B-Base-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Base-Preview).

## Contents

| file | description |
|---|---|
| `ripple_fsdp.py`, `train_rank_probe.py` | FSDP LoRA trainer for 2+ GPUs. rsLoRA with alpha = sqrt(512*r), fp32 LoRA master weights and optimizer state, resumable, dataset and topology checks on resume. |
| `packed_data.py` | Packed-window collator: block-diagonal 4D attention masks, per-segment position ids, first-token label masking. |
| `ripple_lora_plugin.py` | Registers a `SupportsLoRA` subclass of vLLM's ricursive model at runtime. No vLLM rebuild required. |
| `validate_vllm_stage.py`, `validate_hf_stage.py` | Correctness check for the plugin: compares per-token adapter deltas (logprob with adapter minus without) between vLLM and transformers+PEFT. |
| `merge_lora_to_gguf.py` | Merges an adapter into the base weights inside llama.cpp's HF-to-GGUF conversion, streaming tensor by tensor; no merged HF checkpoint is written. |

## Versions

Training: `torch==2.9.1` (cu128), `transformers==4.57.1`, `peft==0.18.0` (`requirements.txt`).
Newer transformers/peft versions break the adapter save/load roundtrip with this trainer.

Serving: vLLM >= 0.34.0rc1 (first version with ricursive support).

Use separate virtualenvs for training and vLLM; their transformers requirements conflict.
The validator is split into two stages for the same reason.

## Training

```
torchrun --standalone --nproc_per_node=2 train_rank_probe.py \
  --train-file data.jsonl --rank 32 --lr 3.7e-7 --max-length 544 \
  --global-batch 16 --micro-batch 4 --eval-batch 16 --warmup 35 \
  --max-steps 400 --save-steps 0 --output-dir out/r32
```

`--rank {32,64,128}`. Measured on 2x A100-80 at sequence length 544: r32 peaks
at ~72.7 GiB per GPU, r64 at ~74.5 GiB. For r64 use `--eval-batch 8`: evals
after the first optimizer step allocate on top of optimizer state and OOM at
batch 16. r128 does not fit on 2x80.

## Serving

```python
import ripple_lora_plugin; ripple_lora_plugin.register()  # before constructing LLM
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="ricursiveAI/Ripple-65B-Base-Preview", tensor_parallel_size=2,
          trust_remote_code=True, enable_lora=True, max_lora_rank=64)
out = llm.generate(prompts, params, lora_request=LoRARequest("a", 1, "out/r32/final"))
```

## Validation

Run stage 1 in the vLLM env, stage 2 in the training env:

```
python validate_vllm_stage.py --model <base> --adapter <dir> --texts texts.jsonl --out vllm_lp.json
python validate_hf_stage.py   --model <base> --adapter <dir> --vllm-lp vllm_lp.json
```

Verified configuration: single adapter, TP=2, bf16, rank 32, targets
q/k/v/o/gate/up/down (`g_proj` absent from the packed `qkg` module). Result on
3,431 tokens: pearson r 0.998, slope 0.998, median per-token disagreement
0.013 (the base models alone differ by 0.018 between the two engines).
Other configurations (multiple adapters, other TP degrees, ranks above 64,
embedding or bias LoRAs) are untested; run the validator before relying on them.

## GGUF

`merge_lora_to_gguf.py` wraps llama.cpp's `convert_hf_to_gguf.py`. While the converter streams the
base checkpoint, every tensor the adapter targets gets `W + scale * (B @ A)` added in fp32 and is cast
back to the base dtype (`scale = alpha / sqrt(r)` for rsLoRA adapters, `alpha / r` otherwise).
Requirements: a llama.cpp checkout with `RicursiveForCausalLM` support and its converter dependencies
(`requirements/requirements-convert_hf_to_gguf.txt`), plus `safetensors`.

```
python merge_lora_to_gguf.py --llama-cpp path/to/llama.cpp --adapter path/to/adapter \
  path/to/Ripple-65B-Base-Preview --outtype q8_0 --split-max-size 45G --outfile merged-Q8_0.gguf

path/to/llama.cpp/build/bin/llama-quantize --allow-requantize \
  merged-Q8_0-00001-of-00002.gguf merged-Q4_K_M.gguf Q4_K_M
```

All arguments other than `--llama-cpp` and `--adapter` go to the converter unchanged
(`--split-max-size 45G` keeps each file under the Hugging Face 50 GB limit; `--no-lazy` converts faster
but needs RAM for the whole model). The run must end with a `MERGE_OK` line, which is printed only when
every adapter tensor pair was applied. A cheap end-to-end check is `llama-perplexity` on held-out text:
the merged file must reproduce the adapter's loss drop against the base.
