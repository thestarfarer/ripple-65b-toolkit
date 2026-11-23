# ripple-65b-toolkit

Tooling for training and serving LoRA adapters on
[ricursiveAI/Ripple-65B-Base-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Base-Preview).

## Contents

| file | description |
|---|---|
| `ripple_fsdp.py`, `train_rank_probe.py`, `train_2048.py` | FSDP LoRA trainer for 2+ GPUs. rsLoRA with alpha = sqrt(512*r), fp32 LoRA master weights and optimizer state, resumable, dataset and topology checks on resume. |
| `packed_data.py` | Packed-window collator: block-diagonal 4D attention masks, per-segment position ids, first-token label masking. |
| `ripple_lora_plugin.py` | Registers a `SupportsLoRA` subclass of vLLM's ricursive model at runtime. No vLLM rebuild required. |
| `validate_vllm_stage.py`, `validate_hf_stage.py` | Correctness check for the plugin: compares per-token adapter deltas (logprob with adapter minus without) between vLLM and transformers+PEFT. |
| `lora_to_gguf.py` | Converts an adapter to a llama.cpp LoRA GGUF, folding the rsLoRA scale so plain `--lora` applies it at the trained strength. |
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

2048 stage, packed rows (`{"packed": [segment, ...]}`; each segment attends only to itself):

```
torchrun --standalone --nproc_per_node=4 train_2048.py --fresh-adapter \
  --train-file mix_2048.jsonl --data-format packed --max-length 2048 \
  --rank 128 --lr 1e-6 --warmup 100 --global-batch 16 --micro-batch 4 --eval-batch 8 \
  --forward-prefetch --no-append-eos --save-steps 250 --output-dir out/r128
```

Measured on 4x H100-80 SXM (NVLink), r128: peak 52.9 GiB per GPU at micro-batch 4 on full
2,048-token rows (eval batch 8 included), ~835 tokens/s, ~38 s per optimizer step of 16 rows.
For comparison, 8x A100-80 SXM ran the same r128 2048 stage at 831 tokens/s.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set for that run.

Plain-text rows get an EOS appended by default; pass `--no-append-eos` for continuation-style
training. No BOS is ever added. Resumable checkpoints at r128 are ~29 GB (4 optimizer shards + adapter).

## Serving

```python
import ripple_lora_plugin; ripple_lora_plugin.register()  # before constructing LLM
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="ricursiveAI/Ripple-65B-Base-Preview", tensor_parallel_size=2,
          trust_remote_code=True, enable_lora=True, max_lora_rank=64)
out = llm.generate(prompts, params, lora_request=LoRARequest("a", 1, "out/r32/final"))
```

vLLM starts tensor-parallel workers with `spawn`, which re-imports your script in every worker:
call `register()` at module level (the workers need it too) and put the generation itself under
`if __name__ == "__main__":`.

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
  path/to/Ripple-65B-Base-Preview --outtype q8_0 --no-lazy --split-max-size 45G --outfile merged-Q8_0.gguf

path/to/llama.cpp/build/bin/llama-quantize --allow-requantize \
  merged-Q8_0-00001-of-00002.gguf merged-Q4_K_M.gguf Q4_K_M
```

All arguments other than `--llama-cpp` and `--adapter` go to the converter unchanged
(`--split-max-size 45G` keeps each file under the Hugging Face 50 GB limit). Pass `--no-lazy`: current
llama.cpp converters load tensors lazily on the meta device, and adding the CPU LoRA delta to them fails
(`Tensor on device cpu is not on the expected device meta!`, `MERGE_INCOMPLETE merged=0`). `--no-lazy`
needs RAM for the whole model; a 65B r128 merge to Q8_0 took 13 min on a 2x Xeon 8480+ box. The run must end with a `MERGE_OK` line, which is printed only when
every adapter tensor pair was applied. A cheap end-to-end check is `llama-perplexity` on held-out text:
the merged file must reproduce the adapter's loss drop against the base.

To keep the adapter separate instead of merging (one base GGUF, many adapters):

```
python lora_to_gguf.py --llama-cpp path/to/llama.cpp --base path/to/Ripple-65B-Base-Preview \
  --adapter path/to/adapter --outfile adapter-f16.gguf
llama-server -m base-f16.gguf --lora adapter-f16.gguf
```

llama.cpp applies LoRA at `alpha / r`; rsLoRA adapters train at `alpha / sqrt(r)`, so converting the
config as-is would load them about sqrt(r) times too weak. `lora_to_gguf.py` writes the trained scale
into the GGUF's alpha. (In current llama.cpp, a manual scale is passed as `--lora-scaled FILE:SCALE`, one argument.)
